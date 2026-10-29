"""Tesseract-based text detection: precise per-word pixel bounding boxes.

Used as a preprocessing stage ahead of the VLM (see mistral_client.py /
extract.py) because a dedicated CV text detector measures geometry far more
precisely than VLM spatial grounding does (see conversation/testing).

Requires the `tesseract-ocr` system package (e.g. `apt-get install
tesseract-ocr`) in addition to the `pytesseract` Python binding -- there is
no pure-Python fallback here.

Raw Tesseract badly under-detects text on this cluttered CAD-plan style
content and produces significant noise from misread drawing linework;
upscaling + adaptive thresholding first (the defaults below) is what makes
it usable (see conversation/testing).

Tesseract's LSTM engine parallelizes its matrix ops across OpenMP threads,
and floating-point summation order isn't guaranteed identical across thread
interleavings -- this produces tiny run-to-run jitter in confidence scores
for marginal detections, which can flip a word across MIN_CONFIDENCE between
otherwise-identical runs (see conversation/testing: same input, different
detected words). Forcing single-threaded execution below removes that
nondeterminism at the source, at the cost of some throughput.
"""

import os

import cv2
import pytesseract

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

UPSCALE = 4
MIN_CONFIDENCE = 5
TESSERACT_CONFIG = "--psm 11"  # sparse text, no assumed layout/orientation


def detect_words(bgr, upscale=UPSCALE, min_confidence=MIN_CONFIDENCE):
    """
    bgr : a tile image in OpenCV BGR order (e.g. from tiling.read_bgr()).

    Returns a list of {"bbox": (x0, y0, x1, y1), "text": str, "confidence": int},
    ordered as Tesseract returned them. bbox is in the ORIGINAL (non-upscaled)
    tile-local pixel space, ready for geocode.bbox_to_geom().
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(
        big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT,
                                      config=TESSERACT_CONFIG)

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        confidence = int(data["conf"][i])
        if confidence < min_confidence:
            continue
        left, top, width, height = (data["left"][i], data["top"][i],
                                     data["width"][i], data["height"][i])
        words.append({
            "bbox": (left / upscale, top / upscale,
                     (left + width) / upscale, (top + height) / upscale),
            "text": text,
            "confidence": confidence,
        })
    return words
