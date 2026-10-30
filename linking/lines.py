import itertools
import math

import cv2
import networkx as nx
import numpy as np
import rasterio.transform
from rasterio.enums import ColorInterp
from scipy.spatial import cKDTree

# A genuine connector line runs across meaningful map distance; hatch-fill
# strokes and dot/symbol marks are typically much shorter. Starting point
# for tuning against real plan data -- not yet empirically calibrated (same
# status as ocr.py's MIN_CONFIDENCE / blocks.py's MAX_GAP_METERS were
# before real-data tuning).
MIN_LENGTH_METERS = 1.0

# How close two segment endpoints must be (geo-units) to count as the same
# joint when bridging fragments back into one path.
ENDPOINT_SNAP_TOLERANCE_METERS = 0.3

# How close to parallel (degrees) two segments meeting at a joint must be
# to be treated as a continuation of the same line, rather than a genuine
# crossing/branch of two unrelated lines that happen to touch at a point
# (e.g. a hatch stroke crossing the connector, or two parcel boundaries
# meeting at a corner).
ANGLE_TOLERANCE_DEG = 8.0

# Two contour edges of one straight stroke are precisely parallel -- much
# tighter than ANGLE_TOLERANCE_DEG's allowance for jitter across a bridged
# gap between separate stroke fragments.
PARALLEL_ANGLE_TOLERANCE_DEG = 3.0

# Generous upper bound on real-world stroke thickness for two segments to
# plausibly be the two edges of the same stroke, not two genuinely distinct
# nearby parallel lines (e.g. two separate hatch strokes, two parcel
# boundaries running alongside each other).
PARALLEL_MAX_OFFSET_METERS = 1.0

# How much of the shorter segment's length (projected onto the other's own
# direction) must run alongside it to count as "the same stroke's edges"
# rather than two lines that merely glance past each other near one end
# (more likely a genuine crossing of two unrelated lines).
PARALLEL_MIN_OVERLAP_FRACTION = 0.7

# Tighter than ANGLE_TOLERANCE_DEG -- a gap bridge asserts "this is the
# SAME straight line with a real interruption", not merely "a plausible
# continuation after a bridged gap", so it holds gap candidates to the same
# precision as PARALLEL_ANGLE_TOLERANCE_DEG.
COLLINEAR_GAP_ANGLE_TOLERANCE_DEG = 3.0

# How far a candidate's endpoints may sit from the other segment's own
# infinite line and still count as the same line rather than a separate,
# merely near-parallel one -- jitter-scale, matching
# ENDPOINT_SNAP_TOLERANCE_METERS rather than PARALLEL_MAX_OFFSET_METERS's
# stroke-width allowance, since this is "the same centerline", not "the
# other edge of the same stroke".
COLLINEAR_GAP_MAX_PERPENDICULAR_OFFSET_METERS = 0.3

# Upper bound on a genuine crossing-junction interruption (see module
# docstring) -- a starting point matching this project's existing
# real-interruption gap conventions (texthandling/blocks.py's
# MAX_GAP_METERS=10.0, texthandling/group.py's MAX_GAP_METERS=5.5), not yet
# tuned against more real data.
MAX_COLLINEAR_GAP_METERS = 8.0


def detect_line_segments(dataset, min_length_meters=MIN_LENGTH_METERS):
    """
    Run LSD over the whole raster and return candidate line segments as
    ((gx0, gy0), (gx1, gy1)) pairs in the raster's CRS, filtered to at
    least min_length_meters long.

    dataset : an open rasterio dataset.
    """
    if dataset.count == 1:
        band = dataset.read(1)
        if dataset.colorinterp[0] == ColorInterp.palette:
            # A single-band PALETTE raster's values are indices into a
            # color table, not luminance -- reading them directly (as if
            # already grayscale) is only coincidentally close to correct
            # for a palette that happens to be roughly luminance-ordered,
            # and silently wrong for any other palette ordering. Expand
            # each index to its true RGB via the dataset's own colormap,
            # then reduce to luminance with the same weights cv2 uses for
            # RGB2GRAY, rather than reading the raw index as gray.
            colormap = dataset.colormap(1)
            lut_size = max(colormap.keys()) + 1
            lut = np.zeros(lut_size, dtype=np.uint8)
            for idx, rgba in colormap.items():
                r, g, b = rgba[0], rgba[1], rgba[2]
                lut[idx] = round(0.299 * r + 0.587 * g + 0.114 * b)
            gray = lut[np.clip(band, 0, lut_size - 1)]
        else:
            gray = band
    else:
        rgb = np.moveaxis(dataset.read([1, 2, 3]), 0, -1)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw_lines, _, _, _ = lsd.detect(gray)
    if raw_lines is None:
        return []

    min_length_px = min_length_meters / dataset.res[0]
    segments = []
    for line in raw_lines:
        x0, y0, x1, y1 = line[0]
        if math.hypot(x1 - x0, y1 - y0) < min_length_px:
            continue
        gx0, gy0 = rasterio.transform.xy(dataset.transform, y0, x0)
        gx1, gy1 = rasterio.transform.xy(dataset.transform, y1, x1)
        segments.append(((gx0, gy0), (gx1, gy1)))
    return segments


def _angle_deg(segment):
    """Segment orientation in degrees, folded to [0, 180) since a line has
    no inherent direction."""
    (x0, y0), (x1, y1) = segment
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180


def _angle_diff_deg(a, b):
    """Smallest angular difference between two 0-180deg orientations, in degrees."""
    diff = abs(a - b)
    return min(diff, 180 - diff)


def _segment_length(segment):
    """Euclidean length of a two-point ((x0, y0), (x1, y1)) segment."""
    (x0, y0), (x1, y1) = segment
    return math.hypot(x1 - x0, y1 - y0)


def _perpendicular_offset_and_overlap(a, b):
    """
    Returns (perp_offset, overlap_fraction) describing how b relates to a's
    direction: perp_offset is b's average perpendicular distance from a's
    line; overlap_fraction is how much of the shorter segment overlaps the
    other's projected extent. High overlap with near-zero offset means the
    same stroke's two edges.
    """
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    adx, ady = ax1 - ax0, ay1 - ay0
    alen = math.hypot(adx, ady)
    if alen == 0:
        return None
    ux, uy = adx / alen, ady / alen

    def project(px, py):
        """Projects (px, py) onto a's direction; returns (position along a,
        signed perpendicular distance from a)."""
        vx, vy = px - ax0, py - ay0
        t = vx * ux + vy * uy  # position along a's direction
        perp = vx * -uy + vy * ux  # signed perpendicular distance from a
        return t, perp

    t0, perp0 = project(bx0, by0)
    t1, perp1 = project(bx1, by1)
    perp_offset = (abs(perp0) + abs(perp1)) / 2

    overlap = max(0.0, min(alen, max(t0, t1)) - max(0.0, min(t0, t1)))
    shorter_len = min(alen, abs(t1 - t0))
    overlap_fraction = overlap / shorter_len if shorter_len > 0 else 0.0

    return perp_offset, overlap_fraction


def _centerline(a, b):
    """Endpoint-averaged synthesized centerline between two parallel
    segments -- pairs each of a's endpoints with whichever of b's is
    closer, so it's correct regardless of which direction b happens to
    run in (LSD gives no guarantee the two edges of one stroke are
    reported in the same direction)."""
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    d_same = math.hypot(ax0 - bx0, ay0 - by0) + math.hypot(ax1 - bx1, ay1 - by1)
    d_swapped = math.hypot(ax0 - bx1, ay0 - by1) + math.hypot(ax1 - bx0, ay1 - by0)
    if d_swapped < d_same:
        bx0, by0, bx1, by1 = bx1, by1, bx0, by0
    return (
        ((ax0 + bx0) / 2, (ay0 + by0) / 2),
        ((ax1 + bx1) / 2, (ay1 + by1) / 2),
    )


def merge_parallel_contour_pairs(segments, angle_tolerance_deg=PARALLEL_ANGLE_TOLERANCE_DEG,
                                  max_offset_meters=PARALLEL_MAX_OFFSET_METERS,
                                  min_overlap_fraction=PARALLEL_MIN_OVERLAP_FRACTION):
    """
    Merges segment pairs that are really the two contour edges of one
    stroke (parallel, similar length, small offset, mostly overlapping)
    into a single centerline segment; unpaired segments are returned as-is.
    Run on detect_line_segments()'s raw output before
    merge_segments_into_paths, to avoid duplicate segments downstream.
    O(n^2) worst case.
    """
    n = len(segments)
    used = [False] * n
    angles = [_angle_deg(s) for s in segments]
    lengths = [_segment_length(s) for s in segments]

    result = []
    for i in range(n):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, n):
            if used[j]:
                continue
            if _angle_diff_deg(angles[i], angles[j]) > angle_tolerance_deg:
                continue
            longer, shorter = max(lengths[i], lengths[j]), min(lengths[i], lengths[j])
            if longer == 0 or shorter / longer < 0.5:
                continue  # too different in length to be edges of the same stroke
            offset_overlap = _perpendicular_offset_and_overlap(segments[i], segments[j])
            if offset_overlap is None:
                continue
            perp_offset, overlap_fraction = offset_overlap
            if perp_offset > max_offset_meters or overlap_fraction < min_overlap_fraction:
                continue
            partner = j
            break
        if partner is not None:
            used[i] = used[partner] = True
            result.append(_centerline(segments[i], segments[partner]))
        else:
            result.append(segments[i])

    merged_count = (n - len(result))
    if merged_count:
        print(f"merge_parallel_contour_pairs: {merged_count} contour-edge pair(s) merged into "
              f"centerlines ({n} segments -> {len(result)})")

    return result


def _endpoint_gap(a, b):
    """Smallest distance between any endpoint of a and any endpoint of b."""
    return min(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa in a for pb in b)


def _bridge_segment(a, b):
    """Spanning segment across two collinear, non-overlapping segments --
    the farthest-apart pair among all four endpoints, i.e. the full extent
    of both combined (same farthest-own-endpoints reasoning as
    merge_segments_into_paths' own degenerate-path fallback)."""
    points = list(a) + list(b)
    return max(
        itertools.combinations(points, 2),
        key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
    )


def bridge_collinear_gaps(segments, angle_tolerance_deg=COLLINEAR_GAP_ANGLE_TOLERANCE_DEG,
                           max_perpendicular_offset_meters=COLLINEAR_GAP_MAX_PERPENDICULAR_OFFSET_METERS,
                           max_gap_meters=MAX_COLLINEAR_GAP_METERS):
    """
    Bridges segment pairs that are really one straight line interrupted by
    a crossing feature (collinear, negligible offset, NOT overlapping) into
    one segment spanning their combined extent; unpaired segments are
    returned as-is. Run after merge_parallel_contour_pairs and before
    merge_segments_into_paths, so only small jitter gaps remain for the
    latter's bridging. O(n^2) worst case.
    """
    n = len(segments)
    angles = [_angle_deg(s) for s in segments]
    used = [False] * n

    result = []
    for i in range(n):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, n):
            if used[j]:
                continue
            if _angle_diff_deg(angles[i], angles[j]) > angle_tolerance_deg:
                continue
            offset_overlap = _perpendicular_offset_and_overlap(segments[i], segments[j])
            if offset_overlap is None:
                continue
            perp_offset, overlap_fraction = offset_overlap
            if perp_offset > max_perpendicular_offset_meters or overlap_fraction > 0:
                continue
            if _endpoint_gap(segments[i], segments[j]) > max_gap_meters:
                continue
            partner = j
            break
        if partner is not None:
            used[i] = used[partner] = True
            result.append(_bridge_segment(segments[i], segments[partner]))
        else:
            result.append(segments[i])

    bridged_count = n - len(result)
    if bridged_count:
        print(f"bridge_collinear_gaps: {bridged_count} collinear gap(s) bridged "
              f"({n} segments -> {len(result)})")

    return result


def merge_segments_into_paths(segments, endpoint_tolerance_meters=ENDPOINT_SNAP_TOLERANCE_METERS,
                               angle_tolerance_deg=ANGLE_TOLERANCE_DEG):
    """
    Merges fragmented, endpoint-adjacent, nearly-collinear segments into
    candidate paths. Returns a list of paths; each path is a list of (x, y)
    "loose end" points -- the true ends of the (possibly multi-fragment)
    line, touched by exactly one of that path's own segments. relate.py
    tests these points directly, without path traversal.
    """
    n = len(segments)
    if n == 0:
        return []

    # Snap nearby endpoints (from any segment) into shared "point nodes" --
    # same cKDTree.query_pairs + networkx connected-components pattern
    # already used in group.py for word-box proximity clustering.
    all_points = []
    for p0, p1 in segments:
        all_points.append(p0)
        all_points.append(p1)
    points_arr = np.array(all_points)

    snap_graph = nx.Graph()
    snap_graph.add_nodes_from(range(len(all_points)))
    for i, j in cKDTree(points_arr).query_pairs(r=endpoint_tolerance_meters):
        snap_graph.add_edge(i, j)

    point_node_id = {}
    for node_id, component in enumerate(nx.connected_components(snap_graph)):
        for idx in component:
            point_node_id[idx] = node_id

    node_ids_of_segment = [
        (point_node_id[2 * i], point_node_id[2 * i + 1]) for i in range(n)
    ]

    # Two segments become part of the same path only if they share a
    # snapped point AND are nearly collinear -- sharing a point alone
    # would also merge unrelated lines that simply cross at that point,
    # which is not a "continuation" of either line.
    path_graph = nx.Graph()
    path_graph.add_nodes_from(range(n))
    angles = [_angle_deg(s) for s in segments]
    for i in range(n):
        for j in range(i + 1, n):
            if not set(node_ids_of_segment[i]) & set(node_ids_of_segment[j]):
                continue
            if _angle_diff_deg(angles[i], angles[j]) <= angle_tolerance_deg:
                path_graph.add_edge(i, j)

    paths = []
    degenerate_fallback_count = 0
    for component in nx.connected_components(path_graph):
        # Group THIS path's own segment-endpoint instances by their
        # snapped point-node id -- scoped to `component` only, so an
        # unrelated segment from a DIFFERENT path that happens to snap to
        # the same global point-node (e.g. two genuinely different lines
        # crossing at one location) can't contaminate this path's own
        # loose-end computation.
        own_points_by_node = {}
        for seg_idx in component:
            for end in (0, 1):
                pt = segments[seg_idx][end]
                node_id = node_ids_of_segment[seg_idx][end]
                own_points_by_node.setdefault(node_id, []).append(pt)

        loose_ends = [pts[0] for pts in own_points_by_node.values() if len(pts) == 1]

        if len(loose_ends) < 2:
            # Degenerate case: every one of this component's snapped point
            # nodes has incidence >= 2, so the degree-1 test above yields 0
            # or 1 usable points. This happens, for example, when LSD
            # detects a single drawn stroke as two near-parallel segments
            # (one per edge of the stroke -- normal LSD behavior) whose
            # endpoints snap together at BOTH ends. Falling through with
            # an empty/singleton path would silently drop a real, valid
            # detected line from all downstream matching (relate.py has
            # nothing to test against). Fall back to the two of this
            # component's OWN endpoint instances that are farthest apart
            # from each other -- for a merged stroke/line, these are its
            # true two ends by construction.
            component_points = [
                segments[seg_idx][end] for seg_idx in component for end in (0, 1)
            ]
            if len(component_points) >= 2:
                loose_ends = list(max(
                    itertools.combinations(component_points, 2),
                    key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
                ))
                degenerate_fallback_count += 1

        paths.append(loose_ends)

    if degenerate_fallback_count:
        print(f"merge_segments_into_paths: {degenerate_fallback_count} of {len(paths)} "
              f"path(s) had fewer than 2 loose ends from degree-1 endpoints (e.g. a "
              f"single stroke detected as two snapped-together near-parallel segments) "
              f"-- fell back to each path's farthest-apart own endpoints instead")

    return paths
