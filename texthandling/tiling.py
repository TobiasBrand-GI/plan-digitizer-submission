"""Split a rasterio dataset into VLM-sized image tiles."""

import base64

import cv2
import numpy as np
from rasterio.windows import Window


def iter_tiles(dataset, tile_size=1024, overlap=64):
    """
    Yield (window, bgr) pairs covering `dataset`, in tile_size x tile_size
    pixel windows with `overlap` pixels shared between adjacent tiles so
    text straddling a tile boundary isn't cut off mid-token. bgr is an
    OpenCV-order (BGR) uint8 array -- callers annotate/encode it as needed
    (see annotate.py / encode_png_b64).

    a text region sitting in the overlap strip can be reported twice, once
    per tile. Fine for a first pass; revisit once real output is inspected.
    """
    stride = tile_size - overlap
    for row_off in range(0, dataset.height, stride):
        for col_off in range(0, dataset.width, stride):
            width = min(tile_size, dataset.width - col_off)
            height = min(tile_size, dataset.height - row_off)
            window = Window(col_off, row_off, width, height)
            yield window, read_bgr(dataset, window)


def read_bgr(dataset, window):
    """Read `window` from `dataset` as an OpenCV-order (BGR) uint8 array."""
    bands = min(3, dataset.count)
    data = dataset.read(indexes=list(range(1, bands + 1)), window=window)
    rgb = np.moveaxis(data, 0, -1)  # (bands, H, W) -> (H, W, bands)
    if bands == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_png_b64(bgr):
    """Encode an OpenCV-order (BGR) array as a base64 PNG."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode tile as PNG")
    return base64.b64encode(buf).decode("ascii")
