"""Draw numbered reference boxes onto a tile image, so the VLM's
classification pass (mistral_client.classify_regions) can refer to each
Tesseract-detected word by index instead of having to re-localize it."""

import cv2

_BOX_COLOR_BGR = (0, 0, 255)  # red -- distinct from typical B/W plan content
_FONT_SCALE = 0.4


def draw_numbered_boxes(bgr, words):
    """Returns a COPY of bgr with each words[i]'s bbox outlined and labeled
    with its index i (matching the order classify_regions expects back)."""
    annotated = bgr.copy()
    for i, word in enumerate(words):
        x0, y0, x1, y1 = (round(v) for v in word["bbox"])
        cv2.rectangle(annotated, (x0, y0), (x1, y1), _BOX_COLOR_BGR, 1)
        cv2.putText(annotated, str(i), (x0, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _BOX_COLOR_BGR, 1, cv2.LINE_AA)
    return annotated
