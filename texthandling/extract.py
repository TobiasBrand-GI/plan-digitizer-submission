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
    Runs OCR + two-pass VLM classification over geotiff_path, dissolves
    cross-tile duplicates, merges lines into per-block polygons, and writes
    one feature per block to out_path as GeoJSON (geometry = block bbox;
    properties = TEXT_FIELDS plus structure.py's discovered keys). Tiles
    with no OCR words skip both VLM calls.
    """
    client = mistral_client.get_client(api_key)

    with rasterio.open(geotiff_path) as dataset:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset_crs(dataset).to_wkt())

        records = []  # accumulated across all tiles; dissolved once the loop finishes

        def record(bbox_px, text, label, confidence, member_words):
            """Appends a record for this detection to the enclosing records
            list; skipped entirely if label is excluded."""
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
