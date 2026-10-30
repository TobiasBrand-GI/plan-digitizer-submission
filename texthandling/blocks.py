
from osgeo import ogr
import networkx as nx

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
    Merges records (dissolve.dissolve_duplicates() + reconcile.reconcile()
    output) within max_gap_meters of each other into one, with geometry as
    the bounding box of their combined extent. Adds "lines" (ordered
    top-to-bottom text per member) and "text" (joined with "; "). Merged
    records drop conflict1/conflict2; unmerged ones pass through unchanged.
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
