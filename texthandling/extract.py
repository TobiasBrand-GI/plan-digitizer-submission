"""Top-level entry point: GeoTIFF -> GeoJSON of extracted text regions.

Four-stage pipeline:
  1. Tesseract (ocr.py) measures precise per-word pixel bboxes, per tile.
  2. A first VLM pass (mistral_client.classify_regions) verifies/corrects
     each word's text and adds a semantic label, referring to each by a
     numbered box (annotate.py) rather than re-localizing it itself -- VLM
     spatial grounding was measurably noisier than a dedicated CV text
     detector, while a CV text detector alone can't tell a parcel number
     from a street name (see conversation/testing).
  3. Words not already known to be atomic single-value labels (see
     EXCLUDED_LABELS) are spatially merged into coherent multi-word groups
     (group.py) and sent through a second VLM pass, so e.g. "GRZ" and "0,4"
     -- two separate Tesseract words -- get classified together as
     "GRZ 0,4" instead of as two disconnected fragments.
  4. Once every tile has been processed, dissolve.py merges duplicate
     detections of the same real-world text that got independently found
     in 2+ overlapping tiles (see tiling.py's overlap) into one feature.
     This can only happen after the full tile loop, so results are
     accumulated in memory rather than streamed straight to the output
     layer.
  5. reconcile.py resolves whatever conflicts dissolve.py couldn't settle
     locally, using one final text-only VLM call that sees the rest of the
     extraction's confirmed results as context plus general zoning-plan
     domain knowledge.
  6. blocks.py merges nearby CONFIRMED lines into one polygon per block
     (e.g. a full zone specification), for downstream processing that
     needs one feature per zone rather than one per line -- geometry only,
     each line's own text is kept intact and joined, never re-flattened.
  7. structure.py parses each block's lines into key/value attributes
     (e.g. "GRZ 0,4" -> GRZ=0,4) via one more text-only VLM call, and the
     output schema is built dynamically from whatever keys are actually
     discovered, since a Bebauungsplan's attribute vocabulary isn't known
     ahead of time.
"""

import rasterio
from osgeo import ogr, osr

from processing.utils import dataset_crs

from . import annotate, blocks, dissolve, geocode, group, mistral_client, ocr, reconcile, structure, tiling
from .schema import TEXT_FIELDS

# Labels from the first classification pass that are atomic single-value
# labels and should be (a) dropped from the output and (b) never spatially
# merged with a neighbor. NOTE: elevation_value is deliberately NOT in this
# set -- some relevant content (e.g. max. building height) gets classified
# as elevation_value too, so it can't be blindly discarded here and needs a
# more careful filter downstream.
EXCLUDED_LABELS = {"parcel_number"}


def extract_text_features(geotiff_path, out_path, api_key=None,
                           model=mistral_client.DEFAULT_MODEL,
                           tile_size=1024, overlap=64, prompt=None,
                           ocr_upscale=ocr.UPSCALE, ocr_min_confidence=ocr.MIN_CONFIDENCE,
                           block_max_gap_meters=blocks.MAX_GAP_METERS):
    """
    Run OCR + two-pass VLM classification over a GeoTIFF, dissolve
    cross-tile duplicates, merge nearby confirmed lines into per-block
    polygons, and write the results to `out_path` as GeoJSON -- one
    feature per block:
      - geometry: the bounding box of that block's member lines' combined
        extent, in the raster's CRS
      - block_max_gap_meters: how far apart (straight-line distance) two
        lines can be and still count as the same block -- tune per
        document if lines that should merge don't, or ones that shouldn't
        do (see blocks.py)
      - properties: TEXT_FIELDS (joined text, label, OCR confidence, and
        conflict1/conflict2 if a merged-away line disagreed with another
        -- see dissolve.py) PLUS one field per attribute key discovered by
        structure.py (e.g. GRZ, GFZ, THmax -- varies per document, empty
        for blocks that don't have that particular attribute)

    The raster is processed in overlapping tiles (see tiling.py) since a
    GeoTIFF is typically far larger than a VLM's usable input resolution.
    Tiles with no OCR-detected words skip both VLM calls entirely.
    """
    client = mistral_client.get_client(api_key)

    with rasterio.open(geotiff_path) as dataset:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset_crs(dataset).to_wkt())

        records = []  # accumulated across all tiles; dissolved once the loop finishes

        def record(bbox_px, text, label, confidence, member_words):
            if label in EXCLUDED_LABELS:
                return
            records.append({
                "geom": geocode.bbox_to_geom(dataset, window, bbox_px),
                "text": text,
                "label": label,
                "confidence": confidence,
                "word_count": len(member_words),
                "word_geoms": [w["geom"] for w in member_words],
            })

        for window, bgr in tiling.iter_tiles(dataset, tile_size=tile_size, overlap=overlap):
            words = ocr.detect_words(bgr, upscale=ocr_upscale, min_confidence=ocr_min_confidence)
            if not words:
                continue
            for w in words:
                w["geom"] = geocode.bbox_to_geom(dataset, window, w["bbox"])

            annotated = annotate.draw_numbered_boxes(bgr, words)
            image_b64 = tiling.encode_png_b64(annotated)
            pass1 = {
                c["index"]: c
                for c in mistral_client.classify_regions(client, image_b64, words, model=model, prompt=prompt)
            }

            candidates = []  # eligible for grouping: real text, not an atomic no-merge label
            for i, word in enumerate(words):
                classification = pass1.get(i)
                if classification is None or not classification.get("is_text", True):
                    continue
                label = classification.get("label")
                text = classification.get("corrected_text") or word["text"]
                if label in EXCLUDED_LABELS:
                    continue  # never grouped, never written out
                candidates.append({**word, "text": text, "label": label})

            if not candidates:
                continue

            groups = group.group_words(candidates, dataset.res[0])
            singles = [g for g in groups if len(g["words"]) == 1]
            merged = [g for g in groups if len(g["words"]) > 1]

            for g in singles:
                w = g["words"][0]
                record(w["bbox"], w["text"], w["label"], w["confidence"], member_words=g["words"])

            if merged:
                merged_annotated = annotate.draw_numbered_boxes(bgr, merged)
                merged_b64 = tiling.encode_png_b64(merged_annotated)
                pass2 = {
                    c["index"]: c
                    for c in mistral_client.classify_regions(client, merged_b64, merged, model=model, prompt=prompt)
                }
                for i, g in enumerate(merged):
                    classification = pass2.get(i)
                    if classification is None or not classification.get("is_text", True):
                        continue
                    label = classification.get("label")
                    text = classification.get("corrected_text") or g["text"]
                    confidence = min(w["confidence"] for w in g["words"])
                    record(g["bbox"], text, label, confidence, member_words=g["words"])

        resolved = dissolve.dissolve_duplicates(records)
        resolved = reconcile.reconcile(client, resolved, model=model)
        merged_blocks = blocks.merge_blocks(resolved, max_gap_meters=block_max_gap_meters)
        merged_blocks, attribute_keys = structure.structure(client, merged_blocks, model=model)
        attribute_keys = [k for k in attribute_keys if k not in TEXT_FIELDS]  # avoid clashing with the base schema

        driver = ogr.GetDriverByName("GeoJSON")
        try:
            if driver.Open(out_path) is not None:
                driver.DeleteDataSource(out_path)  # avoid mixing in a stale/old output file
        except RuntimeError:
            pass
        out_ds = driver.CreateDataSource(out_path)
        out_lyr = out_ds.CreateLayer("text_regions", srs=srs, geom_type=ogr.wkbPolygon)
        for field in TEXT_FIELDS:
            out_lyr.CreateField(ogr.FieldDefn(field, ogr.OFTString))
        for key in attribute_keys:
            out_lyr.CreateField(ogr.FieldDefn(key, ogr.OFTString))
        out_defn = out_lyr.GetLayerDefn()

        for r in merged_blocks:
            feat = ogr.Feature(out_defn)
            feat.SetGeometry(r["geom"])
            feat.SetField("text", r["text"])
            if r["label"]:
                feat.SetField("label", r["label"])
            feat.SetField("confidence", str(r["confidence"]))
            if r["conflict1"]:
                feat.SetField("conflict1", r["conflict1"])
            if r["conflict2"]:
                feat.SetField("conflict2", r["conflict2"])
            for key, value in r["fields"].items():
                feat.SetField(key, value)
            out_lyr.CreateFeature(feat)

        out_ds = None  # flush/close

    return out_path
