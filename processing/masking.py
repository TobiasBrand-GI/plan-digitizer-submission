import numpy as np
import cv2
from numpy.ma.core import floor, ceil

from processing import utils

def red(dataset):
    """Returns a binary mask (morphologically closed) of red-colored pixels
    in dataset, covering both ends of the red hue wraparound in HSV."""
    hsv_image, meta, rgb_image = utils.transform_to_hsv(dataset)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 4. Masken erstellen und kombinieren
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)  # Kombiniert beide Rot-Bereiche

    # 5. Maske auf das Originalbild anwenden (optional: zeigt nur rote Bereiche)
    #masked_output = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)
    k = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k)
    meta.update({
        'count': 1,  # Nur noch ein Band statt drei
        'dtype': 'uint8',  # Maske ist 8-Bit unsigned Integer
        'nodata': 0  # Optional: Nicht-rote Bereiche als NoData definieren
    })

    return closed

def edges(dataset):
    """Returns (Canny edge mask, updated raster meta) from a Gaussian-blurred
    version of dataset's RGB image."""
    hsv_image, meta, rgb_image = utils.transform_to_hsv(dataset)

    blurred_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)
    retrieved_edges = cv2.Canny(image=blurred_image, threshold1=100, threshold2=200)

    meta.update({
        'count': 1,  # Nur noch ein Band statt drei
        'dtype': 'uint8',  # Maske ist 8-Bit unsigned Integer
        'nodata': 0  # Optional: Nicht-rote Bereiche als NoData definieren
    })

    return retrieved_edges, meta

def dots(dataset, height, width, spatial_res):
    """Detects small circular dot contours (dotted boundary-line markers) in
    dataset via contour area/circularity filtering. Returns (dot_img marking
    dot centers, list of (x, y, radius) dots, line_mask of contour outlines)."""
    img = dataset.read()
    img = np.squeeze(img)
    img = np.moveaxis(img, 0, -1)  # (H, W, bands)
    print(height, width)
    # Handle band count safely (1-band, 3-band RGB, or 4-band RGBA)
    if img.ndim == 2:
        gray = img
    elif img.shape[-1] == 4:
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold to isolate dark (black) pixels
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Light morphological opening to detach dots from any touching
    # line segments (dots are "fatter" than the thin boundary line)
    kernel_size = 0
    iterations = 0
    if 1 / spatial_res < 10 :
        kernel_size = 3
        iterations = 1
    else:
        kernel_size = 5
        iterations = 2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)

    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        contours = []
    else:
        contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]

    dot_img = np.zeros((height, width), dtype=np.uint8)
    dots1 = []  # store (x, y, radius) like the Hough output, for downstream use

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        dots_size_factor = 1.5 / spatial_res
        if area < ((floor(dots_size_factor)-4)**2.0) or area > ((ceil(dots_size_factor)+4)**2.0):
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Observed range 0.79-0.85 — allow a bit of headroom either side
        if circularity < 0.70 or circularity > 0.99:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)

        dots1.append((cx, cy, radius))
        #cv2.circle(dot_img, (cx, cy), max(radius, 2), 255, 2)
        cv2.circle(dot_img, (cx, cy), 2, 255, 3)


    # Draw contours directly onto the line mask
    line_mask = np.zeros((height, width), dtype=np.uint8)
    for cnt in contours:
            cv2.drawContours(line_mask, [cnt], -1, color=255, thickness=1)

    return dot_img, dots1, line_mask

def debug_bboxes(dataset, height, width):
    """Debug helper: detects contours in dataset and draws their bounding
    boxes with area/circularity labels on a BGR copy. Returns (annotated
    image, list of per-contour metric dicts)."""
    img = dataset.read()
    img = np.squeeze(img)
    img = np.moveaxis(img, 0, -1)  # (H, W, bands)

    if img.ndim == 2:
        gray = img
    elif img.shape[-1] == 4:
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Color debug image so boxes/text are visible (BGR)
    debug_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    stats = []  # collect metrics per contour for inspection outside the image too

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area == 0 or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        stats.append({
            "id": i,
            "x": x, "y": y, "w": w, "h": h,
            "area": round(area, 1),
            "circularity": round(circularity, 2),
            "aspect_ratio": round(aspect_ratio, 2),
        })

        # Draw every bbox in red so nothing is hidden by filtering
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Small label above the box: index, area, circularity
        label = f"{i}:a{int(area)} c{circularity:.2f}"
        cv2.putText(debug_img, label, (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    return debug_img, stats