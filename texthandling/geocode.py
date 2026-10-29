"""Reproject a tile-local pixel bounding box into the raster's CRS as an
OGR polygon."""

import rasterio.transform
from osgeo import ogr


def bbox_to_geom(dataset, window, bbox_px):
    """
    bbox_px : (x0, y0, x1, y1), absolute pixel coordinates local to the tile
        that was read from `window` (i.e. exactly what ocr.detect_words()
        measured).
    window : the rasterio.windows.Window the tile was read from -- used to
        offset the tile-local pixel coords back into full-raster pixel space.

    Returns an ogr.Geometry Polygon in `dataset`'s CRS.
    """
    x0, y0, x1, y1 = bbox_px
    corners_px = [
        (window.col_off + x0, window.row_off + y0),
        (window.col_off + x1, window.row_off + y0),
        (window.col_off + x1, window.row_off + y1),
        (window.col_off + x0, window.row_off + y1),
    ]

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for px, py in corners_px:
        gx, gy = rasterio.transform.xy(dataset.transform, py, px)
        ring.AddPoint(gx, gy)
    ring.CloseRings()

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly
