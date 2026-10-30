import base64

import cv2
import numpy as np
from rasterio.windows import Window


def iter_tiles(dataset, tile_size=1024, overlap=64):
    """
    Yields (window, bgr) pairs covering dataset in tile_size x tile_size
    pixel windows, with overlap pixels shared between adjacent tiles so text
    straddling a boundary isn't cut off mid-token. bgr is an OpenCV BGR
    uint8 array. A text region in the overlap strip can be reported twice,
    once per tile.
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
