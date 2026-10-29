"""Top-level entry point: mask the red-drawn area outlines in a GeoTIFF,
trace them into polygons, and clip/dissolve them against allowed ALKIS
land-use types -- producing areas.geojson, the polygon layer
linking.join.link_text_to_areas later attaches text-block attributes to.

File-path-in, file-path-out, matching texthandling.extract_text_features
and linking.join.link_text_to_areas rather than requiring callers to share
in-memory state. Every intermediate and final file is written under
`out_dir`, so a caller can point two different runs at two different
directories without one overwriting the other's outputs.
"""

import os

import numpy as np
import rasterio
from osgeo import gdal, ogr

from . import masking, polygonizing, utils

DEFAULT_ALLOWED_VALUES = [
    "Wohnbaufläche",
    "Fläche gemischter Nutzung",
    "Fläche besonderer funktionaler Prägung",
]


def segment_areas(tif_path, umring_path, vectors_path, out_dir,
                   allowed_values=DEFAULT_ALLOWED_VALUES):
    """
    tif_path : source GeoTIFF to mask/segment.
    umring_path : boundary GeoJSON clipping the ALKIS usage-type layer.
    vectors_path : ALKIS usage-type shapefile (nutzungFlurstueck_kgt.shp).
    out_dir : directory every intermediate and final output is written
        into (created if missing): rot_maske_3.tif, shp/polygon3.shp,
        dots.tif, lines.geojson, buffers.geojson, dots_contours.tif, and
        finally areas.geojson.
    allowed_values : ALKIS "nutzart" values kept when clipping/dissolving
        (see polygonizing.intersect_usage_types_by_lines).

    Returns the path to areas.geojson.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("converting to rgb")
    dataset = utils.open_rgb(tif_path, out_dir)
    height = dataset.height
    width = dataset.width
    meta = dataset.meta.copy()

    print("masking red pixels")
    red_mask = masking.red(dataset)

    print("saving masked pixels")
    red_mask_path = os.path.join(out_dir, "rot_maske_3.tif")
    with rasterio.open(red_mask_path, "w", **meta) as dst:
        dst.write(red_mask, 1)

    src_ds = gdal.Open(red_mask_path)

    shp_dir = os.path.join(out_dir, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(os.path.join(shp_dir, "polygon3.shp"))

    mask_polygons = polygonizing.polygonize(src_ds, dst_ds, utils.dataset_crs(dataset).to_wkt())

    spatial_res = dataset.res[0]
    dots_img, dots, line_mask = masking.dots(dataset, height, width, spatial_res)
    print("saving dots")
    meta.update(count=1, dtype=np.uint8)
    with rasterio.open(os.path.join(out_dir, "dots.tif"), "w", **meta) as dst:
        dst.write(dots_img[np.newaxis, :, :])  # (H, W) -> (1, H, W)

    # Written right away (not at the end) so the debug image exists even
    # when a later step fails
    with rasterio.open(os.path.join(out_dir, "dots_contours.tif"), "w", **meta) as dst:
        dst.write(line_mask[np.newaxis, :, :])  # (H, W) -> (1, H, W)

    lines_path = os.path.join(out_dir, "lines.geojson")
    polygonizing.dots_to_lines_gdal(dataset, dots, lines_path, spatial_res)

    lines_ds = ogr.Open(lines_path)
    line_count = lines_ds.GetLayer(0).GetFeatureCount()
    lines_ds = None
    if line_count == 0:
        raise RuntimeError(
            f"no dotted boundary lines found ({len(dots)} dot(s) detected, none chained "
            f"into a line) -- check dots.tif / dots_contours.tif in {out_dir}")

    buffers_path = os.path.join(out_dir, "buffers.geojson")
    polygonizing.postprocess_dotted_lines(lines_path, buffers_path)

    areas_path = os.path.join(out_dir, "areas.geojson")
    polygonizing.intersect_usage_types_by_lines(
        umring_path, vectors_path, buffers_path, areas_path,
        mask_polygons, allowed_values=allowed_values)

    return areas_path
