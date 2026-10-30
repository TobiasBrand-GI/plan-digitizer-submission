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
    words : list of {"bbox", "text", ...} dicts. spatial_res : raster ground
    resolution (meters/pixel), used to convert max_gap_meters to pixels.
    Returns a list of groups, each {"bbox": merged box, "words": member word
    dicts in reading order, "text": concatenated text}; an unmatched word
    becomes its own single-item group.
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
