"""Experiment: replace open_rgb()'s gdal.Translate(rgbExpand="rgb") call.

On tif/z_206.tif, gdal.Translate(...) -- and even the plainer
driver.CreateCopy() it wraps internally, with no rgbExpand at all --
segfaults (confirmed via direct testing: exit code 139/SIGSEGV, reproduced
both with and without rgbExpand). That's a native crash in GDAL 3.10.0's
GTiff driver, not a Python exception open_rgb()'s existing try/except
RuntimeError can catch -- it kills the process/thread outright, which is
why the API call "gets stuck" with no error: the worker handling the
request dies mid-call and never replies.

Reading the same file works fine both via rasterio and via GDAL's own
ReadAsArray -- the crash is isolated to GDAL's copy-to-new-file path, not
reading. This script avoids that path entirely: read the palette-indexed
band and its colormap via rasterio, expand to RGB with a numpy LUT (same
technique linking/lines.py's detect_line_segments() already uses for
palette rasters, just producing 3 bands instead of 1 luminance band), and
write the result with rasterio's own writer.

Run directly to try it against z_206.tif:
    python3 processing/open_rgb_experiment.py
"""

import os

import numpy as np
import rasterio


def open_rgb_experiment(path, out_dir):
    """Drop-in experimental replacement for processing.utils.open_rgb --
    same signature and return value (an open rasterio dataset, RGB if the
    source was single-band, passed through unchanged otherwise)."""
    dataset_tmp = rasterio.open(path)

    if dataset_tmp.count != 1:
        return dataset_tmp

    filename = os.path.basename(path)
    rgb_path = os.path.join(out_dir, "rgb_" + filename)
    print(rgb_path)

    print("expanding palette to rgb (numpy LUT, no gdal.Translate)")
    idx = dataset_tmp.read(1)
    colormap = dataset_tmp.colormap(1)
    lut_size = max(colormap.keys()) + 1
    lut = np.zeros((lut_size, 3), dtype=np.uint8)
    for i, rgba in colormap.items():
        lut[i] = rgba[:3]
    rgb = lut[idx]  # (H, W, 3)

    meta = dataset_tmp.meta.copy()
    meta.update(count=3, dtype="uint8", photometric="RGB")
    meta.pop("nodata", None)
    with rasterio.open(rgb_path, "w", **meta) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)

    print("translated")
    return rasterio.open(rgb_path)


if __name__ == "__main__":
    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tif_path = os.path.join(dirname, "tif", "z_206.tif")
    out_dir = "/tmp/open_rgb_experiment"
    os.makedirs(out_dir, exist_ok=True)

    ds = open_rgb_experiment(tif_path, out_dir)
    print("success:", ds.width, ds.height, ds.count, ds.dtypes)
