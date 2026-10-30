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
