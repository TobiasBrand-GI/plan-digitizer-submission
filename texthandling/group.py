"""Merge nearby OCR word-boxes into coherent multi-word groups ahead of a
second VLM classification pass -- e.g. so "GRZ" and "0,4" (two separate
Tesseract words) get seen and classified together as "GRZ 0,4", instead of
as two disconnected fragments with no indication they belong together.

Only meant to run on words NOT already known to be atomic single-value
labels that should never merge with a neighbor (parcel_number -- see
extract.py's first classification pass and EXCLUDED_LABELS).

Candidate-neighbor search uses a cKDTree, same as
processing/polygonizing.py's dotted-line reconstruction
(build_graph_greedy_matching / dots_to_lines_gdal), and the same
spatial-resolution-aware distance convention: a real-world MAX_GAP_METERS
threshold converted to pixels via the raster's spatial_res (meters/pixel,
e.g. rasterio Dataset.res[0]) instead of a fixed or box-relative pixel gap
-- so grouping behaves consistently regardless of the source image's
resolution.

Unlike build_graph_greedy_matching (which over-fetches a fixed k=6 nearest
neighbors per point before filtering by distance -- fine there, since dot
chains are locally sparse), this uses an uncapped RADIUS query
(query_pairs) instead of a k nearest neighbor cap. A k-cap is the wrong
tool for a pure distance threshold: under high local candidate density
(e.g. many low-OCR-confidence detections crowding one area), a word's true
partner can silently fall outside the top-k by center distance and never
even get gap-tested -- not because the gap check rejects it, but because
the check never runs on that pair at all (see conversation/testing: this
caused legitimate multi-line groups to split once MIN_CONFIDENCE was
lowered and candidate volume roughly doubled).

Merging is restricted to words on the SAME PHYSICAL LINE (their Y-ranges
overlap) -- isotropic 2D proximity used to also bridge separate lines
whenever a document's line-spacing was tight (true for this content: e.g.
"SD/KW 10-35 mit THmax." on one line and "SD/KW 35-45 mit THmax." on the
line below got merged into one flat string). Once words from different
lines are joined into a single space-separated string, the positional
information needed to correctly re-pair each label with its own value is
gone for good -- no later VLM prompt can reliably reconstruct which value
belonged with which label (see conversation/testing: this produced wrong
cross-line pairings like a THmax label ending up next to a completely
different line's SD/KW range). Restricting to same-line merging keeps each
physical line intact as its own group; any cross-line association a
downstream consumer needs happens on those clean per-line readings, not by
flattening them together here.
"""

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

MAX_GAP_METERS = 5.5
MIN_LINE_OVERLAP_FRACTION = 0.4  # of the shorter box's height, to count as "same line"


def _same_line(a, b):
    """True if two boxes' vertical extents overlap enough to plausibly be
    the same physical text line (not just two nearby but separate lines)."""
    ay0, ay1 = a[1], a[3]
    by0, by1 = b[1], b[3]
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap <= 0:
        return False
    return overlap >= MIN_LINE_OVERLAP_FRACTION * min(ay1 - ay0, by1 - by0)


def _gap(a, b):
    """Euclidean gap between two axis-aligned boxes (0 if they touch/overlap)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def group_words(words, spatial_res, max_gap_meters=MAX_GAP_METERS):
    """
    words : list of {"bbox": (x0, y0, x1, y1), "text": str, ...}.
    spatial_res : the raster's ground resolution in meters/pixel (e.g.
        rasterio Dataset.res[0]). max_gap_meters is converted to a pixel
        distance from this, matching dots_to_lines_gdal's convention.

    Returns a list of groups, each:
        {"bbox": merged (x0, y0, x1, y1), "words": [original word dicts, in
         reading order], "text": concatenated text}
    A word with no same-line neighbor within range becomes its own
    single-item group.
    """
    n = len(words)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    if n >= 2:
        max_gap_px = max_gap_meters / spatial_res
        boxes = [w["bbox"] for w in words]
        centers = np.array([((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes])
        diagonals = [((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 for x0, y0, x1, y1 in boxes]

        # Candidate pairs by CENTER distance, over-fetched with a generous
        # radius (max_gap_px plus the largest box diagonal in play) so no
        # pair whose true edge-to-edge gap qualifies can be missed -- center
        # distance is always >= edge gap, never less. No k-cap: every
        # candidate this radius could plausibly include gets the exact
        # checks below, regardless of how many candidates that is.
        radius = max_gap_px + max(diagonals)
        for i, j in cKDTree(centers).query_pairs(r=radius):
            if _same_line(boxes[i], boxes[j]) and _gap(boxes[i], boxes[j]) <= max_gap_px:
                graph.add_edge(i, j)

    groups = []
    for component in nx.connected_components(graph):
        # Reading order within a group is left-to-right (X) only -- now that
        # merging requires the same physical line, Y is no longer a
        # meaningful primary sort key, and using it as one let small
        # per-glyph baseline variation (descenders, cap-height differences)
        # scramble the order (see conversation/testing).
        member_idxs = sorted(component, key=lambda i: words[i]["bbox"][0])
        members = [words[i] for i in member_idxs]
        groups.append({
            "bbox": (
                min(w["bbox"][0] for w in members),
                min(w["bbox"][1] for w in members),
                max(w["bbox"][2] for w in members),
                max(w["bbox"][3] for w in members),
            ),
            "words": members,
            "text": " ".join(w["text"] for w in members),
        })
    return groups
