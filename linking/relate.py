"""Match candidate connector-line paths (see lines.py) to (area polygon,
text block) pairs.

Both sides of a candidate pair use the same relation: `geom.Intersects(
endpoint.Buffer(tolerance))`, i.e. "the endpoint lies within `tolerance`
of this geometry" -- the polygon-side endpoint is NOT required to fall
inside the polygon, only to be reachably close to it. An earlier version
of this module used `polygon.Contains(endpoint.Buffer(tolerance))` for the
polygon side specifically to reject decoy lines that merely graze or cross
the boundary. Real project data overturned that assumption: tracing an
actual, correctly-detected 45m connector line on real plan data (see
conversation/testing) showed its own drafted endpoint lands ~0.36m OUTSIDE
the polygon boundary, not safely inside it -- drafters commonly terminate
a connector right at a boundary vertex/edge rather than inside the fill.
Contains() rejects that endpoint permanently regardless of tolerance
(Contains(endpoint.Buffer(t)) requires the buffered disc's own CENTER, the
raw endpoint, to already lie inside the polygon -- growing t only makes
the test harder to satisfy, never easier), so a genuine connector of this
(apparently common) shape could never match under the old rule.

This does reopen the risk the old rule specifically closed: a line that
merely grazes or crosses through a polygon on its way to a text block can
now match too, since Intersects(buffer) accepts a point on either side of
the boundary. POLYGON_BUFFER_TOLERANCE_METERS is kept small to limit that
blast radius to near-boundary cases rather than opening the door to
arbitrary nearby lines -- but "small" is calibrated from two real,
independently-traced connectors, not the original single 0.5m guess: the
45m connector above measured ~0.36m outside the boundary, and a second
real connector (a "WA1"-zone leader, traced on the same plan) measured
~0.73m outside -- both boundary-termination artifacts of the same kind,
not two different phenomena. 0.5m rejected the second one outright, so
the tolerance was raised to 1.0m, comfortably covering both measured
cases with a small margin while staying an order of magnitude below
TEXT_BUFFER_TOLERANCE_METERS.

NO RETRY PASS: an unmatched text block is never retried with a wider
tolerance. A text block that doesn't match even at TEXT_BUFFER_TOLERANCE_METERS
stays unmatched -- there is no distance-based fallback guess.

TEXT_BUFFER_TOLERANCE_METERS is deliberately larger than the polygon-side
tolerance: real project data shows a genuine connector line's detected
endpoint can land several meters short of the text block it's pointing at
(measured: a typical observed gap of 5-6m on real plan data -- see
conversation/testing).
"""

from osgeo import ogr

POLYGON_BUFFER_TOLERANCE_METERS = 1.0
TEXT_BUFFER_TOLERANCE_METERS = 6.0


def _point(xy):
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(*xy)
    return pt


def _buffered_intersects(geom, point_xy, tolerance):
    return geom.Intersects(_point(point_xy).Buffer(tolerance))


def _text_blocks_touching_any_area(polygons, text_blocks):
    """
    Indices of text blocks whose bbox intersects ANY area polygon -- never
    valid connector targets, no matter which polygon a candidate pair is
    being tested against. A genuine spec block always sits outside every
    polygon it might describe, connected by a line (see module docstring).
    Checking only the specific candidate polygon in a pair (as an earlier
    version of this module did) misses a text block that overlaps a
    DIFFERENT polygon than the one under test, letting it wrongly match
    some other, non-overlapping polygon instead -- this is a general
    pre-filter, applied once before any path/polygon matching, so that
    vulnerability can't reappear on a per-pair basis.
    """
    return {
        ti for ti, tb in enumerate(text_blocks)
        if any(tb.Intersects(poly) for poly in polygons)
    }


def _candidates_for_path(path, polygons, text_blocks, polygon_tolerance, text_tolerance,
                          invalid_text_indices):
    """
    Test every ordered pair of DISTINCT loose-end points in `path` against
    the polygon (Intersects-buffer, polygon_tolerance) and text-block
    (Intersects-buffer, text_tolerance) tests, in both directions -- which
    end is the polygon side and which is the text-block side isn't known
    upfront. Text blocks in `invalid_text_indices` (see
    _text_blocks_touching_any_area) are excluded entirely, regardless of
    which polygon is being tested. Returns a list of (polygon_index,
    text_index) candidate pairs.
    """
    poly_hits_per_point = [
        [pi for pi, poly in enumerate(polygons) if _buffered_intersects(poly, pt, polygon_tolerance)]
        for pt in path
    ]
    text_hits_per_point = [
        [ti for ti, tb in enumerate(text_blocks)
         if ti not in invalid_text_indices and _buffered_intersects(tb, pt, text_tolerance)]
        for pt in path
    ]

    candidates = []
    for ai in range(len(path)):
        for bi in range(len(path)):
            if ai == bi:
                continue
            for pi in poly_hits_per_point[ai]:
                for ti in text_hits_per_point[bi]:
                    candidates.append((pi, ti))
    return candidates


def _closest_polygon_index(poly_indices, polygons, reference_geom):
    return min(poly_indices, key=lambda pi: reference_geom.Distance(polygons[pi]))


def _resolve_match(poly_indices, polygons, text_geom):
    poly_indices = sorted(poly_indices)
    if len(poly_indices) == 1:
        return {"polygon_index": poly_indices[0], "conflict_polygon_indices": []}
    winner = _closest_polygon_index(poly_indices, polygons, text_geom)
    conflicts = [pi for pi in poly_indices if pi != winner]
    return {"polygon_index": winner, "conflict_polygon_indices": conflicts}


def match_text_to_polygons(paths, polygons, text_blocks,
                            polygon_tolerance_meters=POLYGON_BUFFER_TOLERANCE_METERS,
                            text_tolerance_meters=TEXT_BUFFER_TOLERANCE_METERS):
    """
    paths : list of candidate paths from lines.merge_segments_into_paths
        (each a list of (x, y) loose-end points).
    polygons, text_blocks : list of ogr.Geometry.
    polygon_tolerance_meters : Intersects-buffer tolerance for the polygon
        side -- kept small, since it exists only to forgive a connector's
        endpoint landing just outside the boundary rather than to permit
        arbitrary nearby lines (see module docstring).
    text_tolerance_meters : Intersects-buffer tolerance for the text-block
        side -- deliberately larger, since widening this one genuinely
        does help (see module docstring).

    Returns (matches, unmatched_indices):
      matches : {text_index: {"polygon_index": int, "conflict_polygon_indices": [int, ...]}}
      unmatched_indices : sorted list of text_index with no match found on
          the single strict pass -- never guessed by distance-based
          fallback, and never retried with a wider tolerance (see module
          docstring for why a retry pass was removed).

    A text block is excluded from matching entirely -- against every
    polygon, not just whichever one a given candidate pair happens to test
    -- if its bbox intersects ANY area polygon (see
    _text_blocks_touching_any_area). A genuine spec block never overlaps
    the polygon it describes.

    An endpoint whose buffer satisfies the polygon/text-block test for more
    than one candidate is resolved by picking the geometrically closest,
    with the runner-up(s) recorded in conflict_polygon_indices rather than
    silently discarded.
    """
    invalid_text_indices = _text_blocks_touching_any_area(polygons, text_blocks)

    candidates_by_text = {}
    for path in paths:
        for pi, ti in _candidates_for_path(path, polygons, text_blocks,
                                            polygon_tolerance_meters, text_tolerance_meters,
                                            invalid_text_indices):
            candidates_by_text.setdefault(ti, set()).add(pi)

    matches = {
        ti: _resolve_match(poly_set, polygons, text_blocks[ti])
        for ti, poly_set in candidates_by_text.items()
    }

    all_text_indices = set(range(len(text_blocks)))
    unmatched_indices = sorted(all_text_indices - set(matches.keys()))

    ambiguous_count = sum(1 for m in matches.values() if m["conflict_polygon_indices"])
    print(f"relate.match_text_to_polygons: {len(paths)} candidate path(s), "
          f"{len(text_blocks)} text block(s), {len(matches)} matched "
          f"({ambiguous_count} ambiguous), {len(unmatched_indices)} unmatched")
    if unmatched_indices:
        print(f"  unmatched text block indices: {unmatched_indices}")

    return matches, unmatched_indices

