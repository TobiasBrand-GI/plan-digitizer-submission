"""Final step: merge nearby CONFIRMED text lines (dissolve.py / reconcile.py
output) into coherent multi-line "block" features for downstream
processing -- e.g. a WA2 zone's full spec (header + GRZ + GFZ + FHmax +
SD/KW rows) as ONE polygon, not one tiny polygon per line.

This is deliberately a SEPARATE, LATER step from group.py's word-level
merging, not folded back into it: merging across lines destroys the
positional information needed to correctly pair a label with its own value
(see conversation/testing -- this is why group.py only merges words on the
SAME physical line). Once every line has its own final, correct text
(after dissolve.py + reconcile.py), merging complete lines together purely
for GEOMETRY consolidation is safe -- each line's text is kept intact and
simply joined, never re-flattened word by word.
"""

from osgeo import ogr
import networkx as nx

# Line-to-line spacing (full line height plus inter-line whitespace, often
# with a divider between a block's "header" and "body" sub-sections) is
# larger than the word-to-word spacing group.py's identically-named
# constant was tuned for -- reusing that value here under-merged real
# multi-line blocks (see conversation/testing). A reduction to 6.0m was
# tried to curb a real over-merging problem (dozens of unrelated, scattered
# elevation spot-heights getting swept into one block purely by falling
# within reach of each other or of a real zoning line -- see
# merge_blocks()'s own docstring), but confirmed on real data to
# under-merge genuine multi-line blocks instead (less of the desired text
# survived as coherent blocks) -- reverted back to 10.0m. The scattered-
# elevation over-merging problem is real and still unsolved; fixing it
# needs something more targeted than this single distance knob (see
# merge_blocks()'s docstring for why a blanket label exclusion isn't it
# either) -- tune max_gap_meters per-document via extract_text_features in
# the meantime if a specific plan's layout needs it.
MAX_GAP_METERS = 10.0


def _bbox(geom):
    """Axis-aligned bounding box of `geom`, as a Polygon."""
    minx, maxx, miny, maxy = geom.GetEnvelope()
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def merge_blocks(records, max_gap_meters=MAX_GAP_METERS):
    """
    records : dissolve.dissolve_duplicates() (+ reconcile.reconcile())
        output -- list of {"geom": ogr.Geometry, "text": str, "label": str,
        "confidence": int, ...}, already in the raster's CRS.

    Returns a list of the same shape plus a "lines" key: records within
    max_gap_meters (straight-line distance, ogr.Geometry.Distance()) of
    each other are merged into one, with geometry as the BOUNDING BOX of
    their combined extent (not the raw union -- text lines don't touch
    each other, so a plain Union() is a MultiPolygon full of gaps between
    lines, not a single usable shape for downstream spatial joins), "lines"
    as the ordered (top-to-bottom) list of each individual member's own
    (already-correct) text, and "text" as that same list joined with "; "
    for simple/default consumers. structure.py uses "lines" directly rather
    than re-splitting "text", since a value could legitimately contain
    "; " itself. conflict1/conflict2 are dropped for merged blocks (several
    lines' own conflicts don't reduce to one "alternate reading" the way a
    single line's does) but kept as-is for records that don't merge with
    anything -- those also get a trivial "lines": [text] and their geometry
    passed through the same bbox conversion, for a consistent output shape
    either way.

    NOTE: elevation_value records DO participate in merging here (no label
    exclusion) -- content relevant to a zone-spec block (e.g. max. building
    height) can get classified as elevation_value too, so excluding that
    label split apart lines that actually belong in the same block, leaving
    overlapping polygons instead of one clean merged block (see
    conversation/testing). This does mean genuinely unrelated, scattered
    elevation spot-heights that happen to sit within max_gap_meters of each
    other can still get swept into one block -- if that becomes a problem
    again, filtering it out needs to happen in a way that doesn't also
    split real blocks, e.g. distinguishing "isolated point far from other
    content" from "line adjacent to zoning-code lines", not a blanket label
    exclusion.
    """
    n = len(records)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if records[i]["geom"].Distance(records[j]["geom"]) <= max_gap_meters:
                graph.add_edge(i, j)

    result = []
    for component in nx.connected_components(graph):
        members = [records[i] for i in component]
        if len(members) == 1:
            r = members[0]
            result.append({**r, "geom": _bbox(r["geom"]), "lines": [r["text"]]})
            continue

        # Top-to-bottom reading order -- Y increases northward in a
        # standard projected CRS, so descending Y is "top first".
        members.sort(key=lambda r: r["geom"].Centroid().GetY(), reverse=True)

        geom = members[0]["geom"]
        for m in members[1:]:
            geom = geom.Union(m["geom"])

        labels = list(dict.fromkeys(m["label"] for m in members if m.get("label")))
        lines = [m["text"] for m in members]

        result.append({
            "geom": _bbox(geom),
            "text": "; ".join(lines),
            "lines": lines,
            "label": "; ".join(labels) if labels else None,
            "confidence": min(m["confidence"] for m in members),
            "conflict1": None,
            "conflict2": None,
        })
    return result
