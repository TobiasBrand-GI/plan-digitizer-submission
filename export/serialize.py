from enum import IntEnum

from lxml import etree

from . import geometry

XPLAN_NS = "http://www.xplanung.de/xplangml/6/1"
GML_NS = geometry.GML_NS
XLINK_NS = "http://www.w3.org/1999/xlink"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
NSMAP = {"xplan": XPLAN_NS, "gml": GML_NS, "xlink": XLINK_NS, "xsi": XSI_NS}
SCHEMA_LOCATION = (f"{XPLAN_NS} "
                   "https://registry.gdi-de.org/schemas/de.xleitstelle.xplanung/6.1/XPlanung-Operationen.xsd")

# uom values for gml:LengthType / gml:AngleType (gml:MeasureType requires uom)
UOM_LENGTH = "m"
UOM_ANGLE = "grad"


def _x(tag):
    """Returns tag qualified with the xplan namespace, in lxml's {uri}tag form."""
    return f"{{{XPLAN_NS}}}{tag}"


def _g(tag):
    """Returns tag qualified with the gml namespace, in lxml's {uri}tag form."""
    return f"{{{GML_NS}}}{tag}"


def _format(value):
    """Formats a Python value as the XML text XPlanGML expects for it."""
    if isinstance(value, bool):  # before int -- bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, IntEnum):  # code list values are written as their code
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _text(parent, tag, value, **attrib):
    """Appends <xplan:tag>value</xplan:tag>; skipped when value is None."""
    if value is None:
        return None
    element = etree.SubElement(parent, _x(tag), attrib)
    element.text = _format(value)
    return element


def _ref(parent, tag, target_id):
    """Appends <xplan:tag xlink:href="#target_id"/>, an xlink reference to
    another feature's gml:id."""
    etree.SubElement(parent, _x(tag), {f"{{{XLINK_NS}}}href": f"#{target_id}"})


def _geometry(parent, tag, geom, gml_id):
    """Appends <xplan:tag> wrapping geom's GML representation under gml_id."""
    etree.SubElement(parent, _x(tag)).append(geometry.to_gml(geom, gml_id))


def _hoehenangabe(parent, hoehenangabe):
    """Appends an XP_Hoehenangabe element with its bezugspunkt and height bounds."""
    element = etree.SubElement(etree.SubElement(parent, _x("hoehenangabe")), _x("XP_Hoehenangabe"))
    _text(element, "bezugspunkt", hoehenangabe.bezugspunkt)
    for name in ("hMin", "hMax", "hZwingend", "h"):
        _text(element, name, getattr(hoehenangabe, name), uom=UOM_LENGTH)


def _dachgestaltung(parent, dachgestaltung):
    """Appends a BP_Dachgestaltung element with its pitch bounds and roof forms."""
    element = etree.SubElement(etree.SubElement(parent, _x("dachgestaltung")), _x("BP_Dachgestaltung"))
    for name in ("DNmin", "DNmax", "DN", "DNzwingend"):
        _text(element, name, getattr(dachgestaltung, name), uom=UOM_ANGLE)
    for form in dachgestaltung.dachform:
        _text(element, "dachform", form)


def _flaechenobjekt(member, tag, obj, bereich_id):
    """Opens an area feature and writes the part every BP_Flaechenobjekt
    shares (XP_Objekt + BP_Flaechenobjekt), in XSD order. Returns the
    feature element so the caller can append its own properties."""
    element = etree.SubElement(member, _x(tag), {_g("id"): obj.gml_id})
    for hoehenangabe in getattr(obj, "hoehenangabe", []):
        _hoehenangabe(element, hoehenangabe)
    _ref(element, "gehoertZuBereich", bereich_id)
    _text(element, "rechtscharakter", obj.rechtscharakter)
    _geometry(element, "position", obj.position, f"{obj.gml_id}_geom")
    _text(element, "flaechenschluss", obj.flaechenschluss)
    return element


def _baugebiet(member, area, bereich_id):
    """Writes a BP_BaugebietsTeilFlaeche feature: the shared flaechenobjekt
    part plus roof and use-type properties specific to this class."""
    element = _flaechenobjekt(member, "BP_BaugebietsTeilFlaeche", area, bereich_id)
    for dachgestaltung in area.dachgestaltung:
        _dachgestaltung(element, dachgestaltung)
    for name in ("GFZ", "GRZ", "Zmin", "Zmax", "Zzwingend", "Z",
                 "allgArtDerBaulNutzung", "besondereArtDerBaulNutzung"):
        _text(element, name, getattr(area, name))


def _plan(member, plan, bereich_id):
    """Writes the BP_Plan feature: name, boundary, municipalities, plan type,
    and a reference to its Bereich."""
    element = etree.SubElement(member, _x("BP_Plan"), {_g("id"): plan.gml_id})
    _text(element, "name", plan.name)
    _geometry(element, "raeumlicherGeltungsbereich", plan.raeumlicherGeltungsbereich, f"{plan.gml_id}_geom")
    for gemeinde in plan.gemeinde:
        xp_gemeinde = etree.SubElement(etree.SubElement(element, _x("gemeinde")), _x("XP_Gemeinde"))
        _text(xp_gemeinde, "ags", gemeinde.ags)
        _text(xp_gemeinde, "gemeindeName", gemeinde.gemeindeName)
    for plan_art in plan.planArt:
        _text(element, "planArt", plan_art)
    _ref(element, "bereich", bereich_id)


def _bereich(member, bereich, plan_id, planinhalt_ids):
    """Writes the BP_Bereich feature: its boundary and references to its
    plan and the feature objects it contains."""
    element = etree.SubElement(member, _x("BP_Bereich"), {_g("id"): bereich.gml_id})
    _text(element, "nummer", bereich.nummer)
    if bereich.geltungsbereich is not None:
        _geometry(element, "geltungsbereich", bereich.geltungsbereich, f"{bereich.gml_id}_geom")
    for planinhalt_id in planinhalt_ids:
        _ref(element, "planinhalt", planinhalt_id)
    _ref(element, "gehoertZuPlan", plan_id)


def build(auszug):
    """Returns the XPlanAuszug root element."""
    root = etree.Element(_x("XPlanAuszug"),
                         {_g("id"): auszug.gml_id, f"{{{XSI_NS}}}schemaLocation": SCHEMA_LOCATION},
                         nsmap=NSMAP)

    fachobjekte = [*auszug.baugebiete, *auszug.baufenster]
    all_geoms = [auszug.plan.raeumlicherGeltungsbereich, *(o.position for o in fachobjekte)]
    etree.SubElement(root, _g("boundedBy")).append(geometry.envelope(all_geoms))

    def feature_member():
        """Appends and returns a new gml:featureMember element under root."""
        return etree.SubElement(root, _g("featureMember"))

    bereich_id = auszug.bereich.gml_id
    _plan(feature_member(), auszug.plan, bereich_id)
    _bereich(feature_member(), auszug.bereich, auszug.plan.gml_id, [o.gml_id for o in fachobjekte])
    for area in auszug.baugebiete:
        _baugebiet(feature_member(), area, bereich_id)
    for window in auszug.baufenster:
        _flaechenobjekt(feature_member(), "BP_UeberbaubareGrundstuecksFlaeche", window, bereich_id)

    # to_gml() elements each declare xmlns:gml themselves -- drop the duplicates
    etree.cleanup_namespaces(root)
    return root


def write(auszug, path):
    """Builds auszug's XML tree and writes it to path as UTF-8 GML. Returns path."""
    etree.ElementTree(build(auszug)).write(path, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return path
