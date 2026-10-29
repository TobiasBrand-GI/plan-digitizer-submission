"""Orchestrates text-block-to-area-polygon linking end to end: loads area
polygons and text blocks (already produced by processing/ and
texthandling/), detects and merges connector-line candidates from the
source GeoTIFF, matches them, and writes three outputs -- linked area
polygons, any text blocks that couldn't be matched, and every candidate
line the detection+merge stage produced (not just the ones that ended up
matched), for visualizing/debugging what lines.py actually recognized and
processed.

File-path-in, file-path-out, matching every other orchestration function
in this codebase (processing.polygonizing.intersect_usage_types_by_lines,
texthandling.extract_text_features) rather than requiring the areas/
text-handling pipelines to share in-memory state.
"""

import rasterio
from osgeo import ogr

from . import lines, relate


def _load_polygons(path):
    ds = ogr.Open(path, 0)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    lyr = ds.GetLayer(0)
    srs = lyr.GetSpatialRef()
    lyr.ResetReading()
    geoms = [feat.GetGeometryRef().Clone() for feat in lyr]
    ds = None
    return geoms, srs


def _load_text_blocks(path):
    ds = ogr.Open(path, 0)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    field_names = [defn.GetFieldDefn(i).GetNameRef() for i in range(defn.GetFieldCount())]

    geoms, attrs = [], []
    lyr.ResetReading()
    for feat in lyr:
        geoms.append(feat.GetGeometryRef().Clone())
        attrs.append({name: feat.GetField(name) for name in field_names})
    ds = None
    return geoms, attrs, field_names


def _create_output_datasource(path):
    driver = ogr.GetDriverByName("GeoJSON")
    try:
        if driver.Open(path) is not None:
            driver.DeleteDataSource(path)  # avoid mixing in a stale/old output file
    except RuntimeError:
        pass
    return driver.CreateDataSource(path)


def link_text_to_areas(areas_path, text_features_path, geotiff_path,
                        out_path, unmatched_out_path, detected_lines_out_path,
                        min_length_meters=lines.MIN_LENGTH_METERS,
                        polygon_tolerance_meters=relate.POLYGON_BUFFER_TOLERANCE_METERS,
                        text_tolerance_meters=relate.TEXT_BUFFER_TOLERANCE_METERS):
    """
    areas_path : GeoJSON of area polygons (e.g. processing's areas.geojson,
        which currently has empty properties -- this join populates them
        for the first time, it does not merge with existing attributes).
    text_features_path : GeoJSON of structured text blocks (e.g.
        texthandling's text_features.geojson).
    geotiff_path : source raster to detect connector lines in.
    out_path : written with one feature per area polygon -- ALL of them,
        never dropped. A matched text block's fields are populated;
        link_conflict is set if the match was ambiguous; fields are empty
        if the polygon was never matched. If more than one text block
        matches the same polygon, their field values are combined
        (semicolon-joined per field) rather than the later one silently
        overwriting the earlier -- same never-silently-drop principle as
        texthandling/structure.py's same-key-twice handling.
    unmatched_out_path : any text block that never matched a polygon on the
        single matching pass (see relate.py -- there is no retry pass),
        kept as a standalone feature for manual follow-up -- never guessed
        by distance-based fallback.
    detected_lines_out_path : one LineString feature per candidate path
        lines.merge_segments_into_paths() produced -- the ENTIRE detection
        + aggregation result, not just the handful that ended up matched
        (see relate.py). For visualizing/debugging what lines.py actually
        recognized and merged, independent of whether relate.py's
        relational filter later accepted or rejected it.

    Returns (out_path, unmatched_out_path, detected_lines_out_path).
    """
    polygons, srs = _load_polygons(areas_path)
    text_geoms, text_attrs, text_field_names = _load_text_blocks(text_features_path)

    with rasterio.open(geotiff_path) as dataset:
        segments = lines.detect_line_segments(dataset, min_length_meters=min_length_meters)
    segments = lines.merge_parallel_contour_pairs(segments)
    segments = lines.bridge_collinear_gaps(segments)
    paths = lines.merge_segments_into_paths(segments)

    matches, unmatched_indices = relate.match_text_to_polygons(
        paths, polygons, text_geoms,
        polygon_tolerance_meters=polygon_tolerance_meters,
        text_tolerance_meters=text_tolerance_meters,
    )

    matches_by_polygon = {}
    for text_idx, m in matches.items():
        matches_by_polygon.setdefault(m["polygon_index"], []).append(text_idx)

    out_ds = _create_output_datasource(out_path)
    out_lyr = out_ds.CreateLayer("linked_areas", srs=srs, geom_type=ogr.wkbPolygon)
    for name in text_field_names:
        out_lyr.CreateField(ogr.FieldDefn(name, ogr.OFTString))
    out_lyr.CreateField(ogr.FieldDefn("link_conflict", ogr.OFTString))
    out_defn = out_lyr.GetLayerDefn()

    for poly_idx, polygon in enumerate(polygons):
        feat = ogr.Feature(out_defn)
        feat.SetGeometry(polygon)

        merged_fields = {}
        conflict_notes = []
        for text_idx in matches_by_polygon.get(poly_idx, []):
            for name in text_field_names:
                value = text_attrs[text_idx].get(name)
                if value is None:
                    continue
                merged_fields[name] = f"{merged_fields[name]}; {value}" if name in merged_fields else value
            conflict_indices = matches[text_idx]["conflict_polygon_indices"]
            if conflict_indices:
                conflict_notes.append(
                    f"text block {text_idx} ambiguous, also near area polygon(s) {conflict_indices}")

        for name, value in merged_fields.items():
            feat.SetField(name, value)
        if conflict_notes:
            feat.SetField("link_conflict", "; ".join(conflict_notes))

        out_lyr.CreateFeature(feat)
    out_ds = None

    unmatched_ds = _create_output_datasource(unmatched_out_path)
    unmatched_lyr = unmatched_ds.CreateLayer("unmatched_text", srs=srs, geom_type=ogr.wkbPolygon)
    for name in text_field_names:
        unmatched_lyr.CreateField(ogr.FieldDefn(name, ogr.OFTString))
    unmatched_defn = unmatched_lyr.GetLayerDefn()

    for text_idx in unmatched_indices:
        feat = ogr.Feature(unmatched_defn)
        feat.SetGeometry(text_geoms[text_idx])
        for name in text_field_names:
            value = text_attrs[text_idx].get(name)
            if value is not None:
                feat.SetField(name, value)
        unmatched_lyr.CreateFeature(feat)
    unmatched_ds = None

    detected_lines_ds = _create_output_datasource(detected_lines_out_path)
    detected_lines_lyr = detected_lines_ds.CreateLayer(
        "detected_lines", srs=srs, geom_type=ogr.wkbLineString)
    detected_lines_lyr.CreateField(ogr.FieldDefn("path_index", ogr.OFTInteger))
    detected_lines_lyr.CreateField(ogr.FieldDefn("num_points", ogr.OFTInteger))
    detected_lines_defn = detected_lines_lyr.GetLayerDefn()

    skipped_degenerate = 0
    for path_idx, path in enumerate(paths):
        if len(path) < 2:
            # A path lines.py couldn't resolve to at least 2 usable points
            # (see merge_segments_into_paths' own degenerate-path fallback
            # -- this should be rare/none in practice, but geometry with
            # under 2 points can't form a line at all).
            skipped_degenerate += 1
            continue
        line_geom = ogr.Geometry(ogr.wkbLineString)
        for x, y in path:
            line_geom.AddPoint(x, y)

        feat = ogr.Feature(detected_lines_defn)
        feat.SetGeometry(line_geom)
        feat.SetField("path_index", path_idx)
        feat.SetField("num_points", len(path))
        detected_lines_lyr.CreateFeature(feat)
    detected_lines_ds = None
    if skipped_degenerate:
        print(f"  {skipped_degenerate} candidate path(s) had under 2 points, skipped in export")

    print(f"link_text_to_areas: {len(polygons)} area polygon(s), "
          f"{len(matches_by_polygon)} with a matched text block, "
          f"{len(unmatched_indices)} text block(s) unmatched, "
          f"{len(paths)} detected line(s) -> {unmatched_out_path}, {detected_lines_out_path}")

    return out_path, unmatched_out_path, detected_lines_out_path
