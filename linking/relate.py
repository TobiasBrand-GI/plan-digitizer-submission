from osgeo import ogr

POLYGON_BUFFER_TOLERANCE_METERS = 1.0
TEXT_BUFFER_TOLERANCE_METERS = 6.0


def _point(xy):
    """Returns an ogr.Geometry Point at xy."""
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(*xy)
    return pt


def _buffered_intersects(geom, point_xy, tolerance):
    """True if geom intersects a disc of radius tolerance centered at point_xy."""
    return geom.Intersects(_point(point_xy).Buffer(tolerance))


def _text_blocks_touching_any_area(polygons, text_blocks):
    """
    Indices of text blocks whose bbox intersects ANY area polygon -- never
    valid connector targets, since a genuine spec block always sits outside
    every polygon it describes. Applied as a pre-filter before any
    path/polygon matching, so a block overlapping a different polygon than
    the one under test can't match elsewhere.
    """
    return {
        ti for ti, tb in enumerate(text_blocks)
        if any(tb.Intersects(poly) for poly in polygons)
    }


def _candidates_for_path(path, polygons, text_blocks, polygon_tolerance, text_tolerance,
                          invalid_text_indices):
    """
    Tests every ordered pair of distinct loose-end points in path against
    the polygon and text-block buffer-intersects tests, in both directions,
    since which end is which isn't known upfront. Text blocks in
    invalid_text_indices are excluded entirely. Returns a list of
    (polygon_index, text_index) candidate pairs.
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
    """Returns the index into poly_indices of the polygon closest to reference_geom."""
    return min(poly_indices, key=lambda pi: reference_geom.Distance(polygons[pi]))


def _resolve_match(poly_indices, polygons, text_geom):
    """Resolves one text block's candidate polygon indices to a single
    polygon_index (nearest to text_geom if ambiguous), recording the rest
    as conflict_polygon_indices."""
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
    Matches candidate paths (from lines.merge_segments_into_paths) to
    (polygon, text block) pairs via buffered intersection tests.
    polygon_tolerance_meters/text_tolerance_meters set each side's buffer.
    Text blocks overlapping any polygon are excluded. Returns (matches,
    unmatched_indices): matches maps text_index to {"polygon_index",
    "conflict_polygon_indices"} (ties resolved by nearest polygon);
    unmatched_indices lists text blocks with no match, never retried.
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

