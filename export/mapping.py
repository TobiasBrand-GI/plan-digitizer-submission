"""
Turns the pipeline's vector outputs into export.model objects:

    linked_areas.geojson  -> BaugebietsTeilFlaeche  (geometry + parsed text attributes)
    shp/polygon3.shp      -> UeberbaubareGrundstuecksFlaeche  (geometry only)
    umring GeoJSON        -> raeumlicherGeltungsbereich / geltungsbereich

The structuring pass (texthandling/structure.py) produces free-form keys,
so every key goes through an explicit alias -> parser -> model-field path.
Anything that can't be mapped becomes an Issue instead of being dropped
silently -- the caller writes those to the export report for manual review.

Geometries are cloned but not repaired here; that's geometry.py's job.
"""

import re
from dataclasses import dataclass

from osgeo import ogr

from . import model, parse
from .parse import ParseError

ogr.UseExceptions()

# Properties that describe the text detection itself, not plan content.
METADATA_KEYS = {"text", "label", "confidence", "conflict1", "conflict2"}

# Height keys: <kuerzel>[min|max], e.g. "FHmax", "THmax", "GHmin", "OK".
# Keys are the plan's kuerzel; values the XP_ArtHoehenbezugspunkt member
# (they differ for GH -> GBH and HA -> HBA, see XP_ArtHoehenbezugspunkt.xml).
_HEIGHT_KUERZEL = {
    "TH": "TH", "FH": "FH", "OK": "OK", "LH": "LH", "SH": "SH", "EFH": "EFH",
    "HA": "HBA", "UK": "UK", "GH": "GBH", "WH": "WH", "GOK": "GOK",
}
_HEIGHT_KEY = re.compile(r"^(" + "|".join(sorted(_HEIGHT_KUERZEL, key=len, reverse=True)) + r")(min|max)?$")

# Roof pitch without a roof form. Keys that are a BP_Dachform kuerzel
# (e.g. "SD" = Satteldach) carry the form as well -- see map_area.
_ROOF_PITCH_KEYS = {"DN", "D"}

_STOREY_KEYS = {"Z", "Zmin", "Zmax", "Zzwingend"}


@dataclass
class Issue:
    source: str        # e.g. "linked_areas.geojson"
    feature_index: int
    key: str | None
    raw_value: str | None
    reason: str


def _parse_joined(raw, parser):
    """texthandling/structure.py joins a key that occurred more than once in
    a block with "; ". Identical readings collapse to one value; differing
    readings can't be told apart, so the value is rejected."""
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    if not parts:
        raise ParseError(f"empty value {raw!r}")
    values = [parser(part) for part in parts]
    if any(value != values[0] for value in values[1:]):
        raise ParseError(f"joined values differ and can't be told apart: {raw!r}")
    return values[0]


def _parse_dachformen(raw):
    """'SD' -> [SD], 'SD/WD' or 'SD, WD' -> [SD, WD]"""
    kuerzel = [k.strip() for k in re.split(r"[/,]", raw) if k.strip()]
    try:
        return [model.Dachform[k] for k in kuerzel]
    except KeyError as e:
        raise ParseError(f"unknown roof form {e.args[0]!r}") from None


def _parse_pitch(raw):
    """Like parse.parse_range, but a value without any digit (e.g. 'SD':
    'Satteldach') yields None -- the roof form alone is still usable."""
    if not re.search(r"\d", raw):
        return None
    return parse.parse_range(raw)


def map_area(properties, geometry, source="linked_areas.geojson", feature_index=0):
    """
    properties : one linked_areas feature's attribute dict.
    geometry : its ogr.Geometry (cloned by the caller).

    Returns (BaugebietsTeilFlaeche, list[Issue]).
    """
    issues = []
    fields = {}
    hoehenangaben = []
    dachgestaltungen = []

    def issue(key, raw, reason):
        issues.append(Issue(source, feature_index, key, raw, reason))

    for key, raw in properties.items():
        if key in METADATA_KEYS or raw in (None, ""):
            continue
        raw = str(raw)
        try:
            if key == "classification":
                abbreviation, storeys = _parse_joined(raw, parse.split_nutzung_geschosse)
                try:
                    fields["besondereArtDerBaulNutzung"] = model.BesondereArtDerBaulNutzung[abbreviation]
                except KeyError:
                    issue(key, raw, f"unknown use type {abbreviation!r}")
                if storeys is not None:
                    # a plain numeral is a maximum (Z); circled = Zzwingend is not
                    # distinguishable from OCR text
                    fields["Z"] = storeys

            elif key in ("GRZ", "GFZ"):
                fields[key] = _parse_joined(raw, parse.to_float)

            elif key in _STOREY_KEYS:
                fields[key] = _parse_joined(raw, parse.roman_to_int)

            elif key in model.Dachform.__members__:
                pitch = _parse_joined(raw, _parse_pitch)
                dachgestaltungen.append(model.Dachgestaltung(
                    dachform=[model.Dachform[key]],
                    **({} if pitch is None else dict(DNmin=pitch.min, DNmax=pitch.max, DN=pitch.exact)),
                ))

            elif key in _ROOF_PITCH_KEYS:
                pitch = _parse_joined(raw, parse.parse_range)
                dachgestaltungen.append(model.Dachgestaltung(DNmin=pitch.min, DNmax=pitch.max, DN=pitch.exact))

            elif key == "dachform":
                dachgestaltungen.append(model.Dachgestaltung(dachform=_parse_joined(raw, _parse_dachformen)))

            elif (match := _HEIGHT_KEY.match(key)):
                kuerzel, bound = match.groups()
                bezugspunkt = model.ArtHoehenbezugspunkt[_HEIGHT_KUERZEL[kuerzel]]
                if bound == "max":
                    hoehenangaben.append(model.Hoehenangabe(
                        bezugspunkt=bezugspunkt, hMax=_parse_joined(raw, parse.to_float)))
                elif bound == "min":
                    hoehenangaben.append(model.Hoehenangabe(
                        bezugspunkt=bezugspunkt, hMin=_parse_joined(raw, parse.to_float)))
                else:
                    span = _parse_joined(raw, parse.parse_range)
                    hoehenangaben.append(model.Hoehenangabe(
                        bezugspunkt=bezugspunkt, hMin=span.min, hMax=span.max, h=span.exact))

            else:
                issue(key, raw, "no XPlanung mapping for this key")

        except ParseError as e:
            issue(key, raw, str(e))

    # Linked text that produced neither an attribute nor a failure (e.g. a
    # bare "private Zufahrt" note) would otherwise vanish without a trace.
    text = properties.get("text")
    if text and not (fields or hoehenangaben or dachgestaltungen or issues):
        issue("text", str(text), "linked text produced no XPlanung attributes")

    area = model.BaugebietsTeilFlaeche(
        position=geometry,
        hoehenangabe=hoehenangaben,
        dachgestaltung=dachgestaltungen,
        **fields,
    )
    return area, issues


def read_areas(linked_areas_path):
    """Returns (list[BaugebietsTeilFlaeche], list[Issue]) for every feature
    in linked_areas.geojson -- areas with no linked text are kept too, as
    plain polygons without attributes."""
    source = "linked_areas.geojson"
    ds = ogr.Open(linked_areas_path)
    layer = ds.GetLayer(0)
    areas, issues = [], []
    for index, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        if geom is None:
            issues.append(Issue(source, index, None, None, "feature has no geometry; skipped"))
            continue
        area, area_issues = map_area(feature.items(), geom.Clone(), source, index)
        areas.append(area)
        issues.extend(area_issues)
    ds = None
    return areas, issues


def read_baufenster(polygon_shp_path):
    """Building windows from the red-mask polygonization: only HA = 255
    (the mask value) -- polygonizing.polygonize already deletes the rest,
    filtered again here so a stale file can't leak background polygons."""
    ds = ogr.Open(polygon_shp_path)
    layer = ds.GetLayer(0)
    layer.SetAttributeFilter("HA = 255")
    windows = [
        model.UeberbaubareGrundstuecksFlaeche(position=feature.GetGeometryRef().Clone())
        for feature in layer
        if feature.GetGeometryRef() is not None
    ]
    ds = None
    return windows


def read_geltungsbereich(umring_path):
    """Union of every polygon in the boundary file -- usually exactly one."""
    ds = ogr.Open(umring_path)
    layer = ds.GetLayer(0)
    union = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        union = geom.Clone() if union is None else union.Union(geom)
    ds = None
    if union is None:
        raise ValueError(f"no geometry in {umring_path}")
    return union
