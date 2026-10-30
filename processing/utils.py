import numpy as np
import cv2
from osgeo import gdal, ogr, osr
import rasterio
from rasterio.crs import CRS
import math
from collections import defaultdict
import networkx as nx
from processing import fix_topology, open_rgb_experiment


GRID_SIZE = 0.001  # match units of your data, tune as needed

FALLBACK_EPSG = 25832


def dataset_crs(dataset):
    """The raster's CRS, or EPSG:25832 if it has none -- assumes the
    raster's transform is in UTM 32N coordinates."""
    if dataset.crs is not None:
        return dataset.crs
    print(f"WARNING: no CRS in {dataset.name}, assuming EPSG:{FALLBACK_EPSG}")
    return CRS.from_epsg(FALLBACK_EPSG)

def transform_to_hsv(dataset):
    """Returns (hsv_image, raster meta, rgb_image) built from dataset's
    first three bands."""
    r = dataset.read(1)
    g = dataset.read(2)
    b = dataset.read(3)
    rgb_image = np.dstack((r, g, b))
    meta = dataset.meta.copy()

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    return hsv_image, meta, rgb_image

def export_lines_to_vector(lines, transform, crs_wkt, output_path="lines.geojson"):
    """Writes lines (pixel-coordinate (x1, y1)-(x2, y2) pairs) as
    georeferenced LineString features to output_path, converting pixel to
    geo coordinates via transform and using crs_wkt as the layer's SRS."""
    # Setup driver and datasource
    driver = ogr.GetDriverByName("GeoJSON")
    datasource = driver.CreateDataSource(output_path)

    # Setup spatial reference from rasterio CRS
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)

    # Create layer
    layer = datasource.CreateLayer("lines", srs=srs, geom_type=ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

    # Add lines
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]

        # Convert pixel coords to geo coords using affine transform
        lon1, lat1 = transform * (x1, y1)
        lon2, lat2 = transform * (x2, y2)

        # Create geometry
        geom = ogr.Geometry(ogr.wkbLineString)
        geom.AddPoint(lon1, lat1)
        geom.AddPoint(lon2, lat2)

        # Create feature
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("id", i)
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)

        feature = None  # dereference

    datasource = None  # flush and close

def open_rgb(path, out_dir):
    """Opens path with rasterio; if single-band, expands it to RGB via
    gdal.Translate (written under out_dir) and returns that dataset,
    otherwise returns the original dataset unchanged."""
    print("opening ori tif")
    dataset_tmp = rasterio.open(path)

    print("checking rgb status")
    if dataset_tmp.count == 1:
        filename = path.split("/")[-1]
        rgb_path = out_dir+"/rgb_"+filename
        print(rgb_path)

        try:
            print("translating tif")
            gdal.Translate(rgb_path, path, rgbExpand="rgb")
        except RuntimeError as e:
            print("test")
            print(f"GDAL Translate failed: {e}", flush=True)
            raise RuntimeError(f"GDAL Translate failed: {e}")
        print("translated")
        dataset = rasterio.open(rgb_path)
        return dataset
        #return open_rgb_experiment.open_rgb_experiment(path, out_dir)
    else:
        return dataset_tmp


def remove_spikes(ordered_points, angle_threshold_deg=25, max_iterations=5):
    """
    Removes vertices that create sharp spikes (near-reversal turns) in
    ordered_points, an Nx2 array of path-ordered (x, y) points. A vertex is
    a spike if its interior angle is below angle_threshold_deg (lower =
    more aggressive). Repeats up to max_iterations times, since removing
    one spike can reveal another.
    """
    points = np.array(ordered_points, dtype=float)

    for _ in range(max_iterations):
        if len(points) < 3:
            break

        keep_mask = np.ones(len(points), dtype=bool)

        for i in range(1, len(points) - 1):
            p_prev = points[i - 1]
            p_curr = points[i]
            p_next = points[i + 1]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue  # duplicate point, leave for separate cleanup

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # guard fp rounding
            angle_deg = np.degrees(np.arccos(cos_angle))

            # angle_deg close to 0 means the path folds back sharply
            # (a spike); close to 180 means it continues nearly straight
            if angle_deg < angle_threshold_deg:
                keep_mask[i] = False

        if keep_mask.all():
            break  # nothing removed, stable — stop early

        points = points[keep_mask]

    return points


def square_buffer_geometry(geom, distance):
    """
    Buffer any OGR geometry by `distance`, with square (flat) end caps.
    """
    if geom is None:
        raise RuntimeError("No geometry to buffer")

    return geom.Buffer(distance,[
        "ENDCAP_STYLE=SQUARE",  # square/flat caps at the true line ends
        "JOIN_STYLE=MITRE",  # round joins at interior bends
    ])


def _extract_lines(geom):
    """Flatten a boundary geometry (LineString / MultiLineString / GeometryCollection) into a list of LineStrings."""
    if ogr.GT_Flatten(geom.GetGeometryType()) == ogr.wkbLineString:
        return [geom]
    return [line for i in range(geom.GetGeometryCount()) for line in _extract_lines(geom.GetGeometryRef(i))]


def _interpolate_points(coords, spacing):
    """Walk a list of (x, y) vertices, returning points every `spacing` distance."""
    points = [coords[0]]
    remaining = spacing
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue
        dx, dy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
        dist_into_seg = 0.0
        while remaining <= seg_len - dist_into_seg:
            dist_into_seg += remaining
            points.append((x1 + dx * dist_into_seg, y1 + dy * dist_into_seg))
            remaining = spacing
        remaining -= (seg_len - dist_into_seg)
    return points

def sample_points_along_boundary(geom, spacing=5.0):
    """
    Sample points every `spacing` units along the outer/inner rings of a
    (Multi)Polygon geometry (e.g. a dissolved buffer). Returns an OGR
    MultiPoint geometry.
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for line in _extract_lines(geom.GetBoundary()):
        coords = [(x, y) for x, y, *_ in line.GetPoints()]
        for x, y in _interpolate_points(coords, spacing):
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint(x, y)
            multipoint.AddGeometry(pt)
    return multipoint

def _circumcenter(a, b, c):
    """Circumcenter of the triangle formed by points a, b, c. None if collinear."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy)


def voronoi_polygons(points, extent_padding=0.1):
    """
    Builds Voronoi polygons for points (an OGR MultiPoint) via the dual of
    GEOS's Delaunay triangulation: each triangle's circumcenter becomes a
    Voronoi vertex, and each site's cell connects its touching triangles'
    circumcenters by angle, then clips to a padded bounding box. Returns one
    ogr.Geometry polygon per input point, skipping invalid ones.
    """

    triangles = points.DelaunayTriangulation()  # GeometryCollection of triangle Polygons

    site_to_centers = defaultdict(list)
    for i in range(triangles.GetGeometryCount()):
        ring = triangles.GetGeometryRef(i).GetGeometryRef(0)
        verts = ring.GetPoints()[:3]  # 4th point repeats the 1st
        cc = _circumcenter(*verts)
        if cc is None:
            continue
        for v in verts:
            key = (round(v[0], 6), round(v[1], 6))
            site_to_centers[key].append(cc)

    point_list = [
        (points.GetGeometryRef(i).GetX(), points.GetGeometryRef(i).GetY())
        for i in range(points.GetGeometryCount())
    ]

    # Clipping extent, padded around the point set
    xs = [x for x, y in point_list]
    ys = [y for x, y in point_list]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad_x = (maxx - minx) * extent_padding or 1.0
    pad_y = (maxy - miny) * extent_padding or 1.0
    extent_ring = ogr.Geometry(ogr.wkbLinearRing)
    extent_ring.AddPoint(minx - pad_x, miny - pad_y)
    extent_ring.AddPoint(maxx + pad_x, miny - pad_y)
    extent_ring.AddPoint(maxx + pad_x, maxy + pad_y)
    extent_ring.AddPoint(minx - pad_x, maxy + pad_y)
    extent_ring.AddPoint(minx - pad_x, miny - pad_y)
    extent_poly = ogr.Geometry(ogr.wkbPolygon)
    extent_poly.AddGeometry(extent_ring)

    cells = []
    for x, y in point_list:
        key = (round(x, 6), round(y, 6))
        centers = list(set(site_to_centers.get(key, [])))
        if len(centers) < 3:
            continue

        cx = sum(p[0] for p in centers) / len(centers)
        cy = sum(p[1] for p in centers) / len(centers)
        centers.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for cxp, cyp in centers:
            ring.AddPoint(cxp, cyp)
        ring.AddPoint(centers[0][0], centers[0][1])
        cell = ogr.Geometry(ogr.wkbPolygon)
        cell.AddGeometry(ring)

        clipped = cell.Intersection(extent_poly)
        cells.append(clipped)

    return cells


def _explode_to_linestrings(geom):
    """Flatten any LineString / MultiLineString / GeometryCollection into a list of single-part LineStrings."""
    if geom is None:
        return []
    if ogr.GT_Flatten(geom.GetGeometryType()) == ogr.wkbLineString:
        return [geom]
    return [
        line

        for i in range(geom.GetGeometryCount())
        for line in _explode_to_linestrings(geom.GetGeometryRef(i))
    ]


def _split_into_segments(line):
    """Break a multi-vertex LineString into individual 2-point segment LineStrings."""
    coords = line.GetPoints()
    segments = []
    for (x1, y1, *_), (x2, y2, *_) in zip(coords, coords[1:]):
        seg = ogr.Geometry(ogr.wkbLineString)
        seg.AddPoint(x1, y1)
        seg.AddPoint(x2, y2)
        segments.append(seg)
    return segments


def polygons_to_lines(polygons):
    """
    Converts a list of polygon (or MultiPolygon) geometries, e.g. from
    voronoi_polygons(), into their boundary lines exploded down to
    individual 2-point segment LineStrings -- one per edge, not per ring or
    polygon. Shared edges between adjacent cells are NOT deduplicated; each
    polygon contributes its own copy.
    """
    lines = []
    for poly in polygons:
        if poly is None or poly.IsEmpty():
            continue
        boundary = poly.GetBoundary()
        for ring_line in _explode_to_linestrings(boundary):
            lines.extend(_split_into_segments(ring_line))
    return lines


def filter_lines_within(lines, buffer_geom):
    """
    Keep only the lines that are OGC "Within" the given buffer geometry.

    Parameters
    ----------
    lines : list of ogr.Geometry
        LineString geometries, e.g. exploded Voronoi cell boundaries.
    buffer_geom : ogr.Geometry
        The (dissolved) buffer polygon to test against.

    Returns
    -------
    list of ogr.Geometry
        The subset of `lines` for which line.Within(buffer_geom) is True.
    """
    return [line for line in lines if line is not None and line.Within(buffer_geom)]

def read_layer_geom(ds, layer_index=0):
    """Read and dissolve all geometries of a layer's features into one geometry."""
    lyr = ds.GetLayer(layer_index)
    lyr.ResetReading()
    collection = ogr.Geometry(ogr.wkbGeometryCollection)
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is not None:
            collection.AddGeometry(geom)
    if collection.GetGeometryCount() == 0:
        return None
    return collection.UnaryUnion()

def keep_polygonal(geom):
    """
    Extract only the Polygon/MultiPolygon parts of a geometry, dropping any
    stray Point/LineString slivers a clip (Intersection) can produce when a
    feature touches the clip boundary exactly. Returns None if nothing
    polygonal remains.
    """
    if geom is None or geom.IsEmpty():
        return None
    flat_type = ogr.GT_Flatten(geom.GetGeometryType())
    if flat_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return geom
    if flat_type == ogr.wkbGeometryCollection:
        polys = ogr.Geometry(ogr.wkbGeometryCollection)
        for i in range(geom.GetGeometryCount()):
            part = keep_polygonal(geom.GetGeometryRef(i))
            if part is not None:
                polys.AddGeometry(part)
        return polys if polys.GetGeometryCount() > 0 else None
    return None

def _sum_linear_length(geom):
    """Sum the length of all linear parts of a geometry (handles Line,
    MultiLine, and GeometryCollection results from an Intersection)."""
    flat_type = ogr.GT_Flatten(geom.GetGeometryType())
    if flat_type == ogr.wkbLineString:
        return geom.Length()
    if flat_type in (ogr.wkbMultiLineString, ogr.wkbGeometryCollection):
        return sum(_sum_linear_length(geom.GetGeometryRef(i))
                   for i in range(geom.GetGeometryCount()))
    return 0.0


def _shared_boundary_length(geom_a, geom_b):
    """Length of the shared boundary (linear intersection) between geom_a
    and geom_b; 0 if they don't share an edge."""
    inter = geom_a.Intersection(geom_b)
    if inter is None or inter.IsEmpty():
        return 0.0
    return _sum_linear_length(inter)


def _iter_polygons(geom_source):
    """Yield individual Polygon geometries from either a GeometryCollection
    (e.g. the output of Polygonize()), a plain Python list/iterable of
    geometries, or a single Polygon."""
    if isinstance(geom_source, ogr.Geometry):
        flat_type = ogr.GT_Flatten(geom_source.GetGeometryType())
        if flat_type in (ogr.wkbGeometryCollection, ogr.wkbMultiPolygon):
            for i in range(geom_source.GetGeometryCount()):
                yield geom_source.GetGeometryRef(i)
            return
        if flat_type == ogr.wkbPolygon:
            yield geom_source
            return
        raise TypeError(f"Unsupported geometry type: {geom_source.GetGeometryName()}")

    # Fallback: assume it's an iterable of ogr.Geometry (or an ogr.Layer)
    for item in geom_source:
        if isinstance(item, ogr.Geometry):
            yield item
        else:
            # ogr.Layer yields Features, not Geometries
            yield item.GetGeometryRef()


def _delaunay_candidate_pairs(node_ids, geoms, coord_precision=6):
    """Build candidate adjacency pairs via GDAL/OGR's own Delaunay
    triangulation (ogr.Geometry.DelaunayTriangulation(), backed by GEOS).
    """
    if len(node_ids) < 3:
        return [(node_ids[i], node_ids[j])
                for i in range(len(node_ids))
                for j in range(i + 1, len(node_ids))]

    coord_to_id = {}
    mp = ogr.Geometry(ogr.wkbMultiPoint)
    for nid in node_ids:
        x, y = geoms[nid].Centroid().GetPoint_2D()
        key = (round(x, coord_precision), round(y, coord_precision))
        coord_to_id[key] = nid
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.AddPoint(x, y)
        mp.AddGeometry(pt)

    triangulation = mp.DelaunayTriangulation()
    if triangulation is None or triangulation.IsEmpty():
        return [(node_ids[i], node_ids[j])
                for i in range(len(node_ids))
                for j in range(i + 1, len(node_ids))]

    edges = set()
    for i in range(triangulation.GetGeometryCount()):
        triangle = triangulation.GetGeometryRef(i)
        ring = triangle.GetGeometryRef(0)
        npts = ring.GetPointCount() - 1
        tri_ids = []
        for p in range(npts):
            x, y, _ = ring.GetPoint(p)
            key = (round(x, coord_precision), round(y, coord_precision))
            nid = coord_to_id.get(key)
            if nid is not None:
                tri_ids.append(nid)

        for a in range(len(tri_ids)):
            for b in range(a + 1, len(tri_ids)):
                x, y = tri_ids[a], tri_ids[b]
                edges.add((min(x, y), max(x, y)))

    return list(edges)


def eliminate_small_polygons(polygons, min_area=150.0,
                              keep_attributes_from="largest",
                              retriangulate_every=25):
    """
    Dissolves polygons smaller than min_area into the neighbor with which
    they share the longest border (like QGIS's Eliminate tool). Accepts a
    GeometryCollection/MultiPolygon, a single Polygon, a list of Polygons,
    or an ogr.Layer. Adjacency comes from Delaunay triangulation over
    centroids, re-run periodically as merges proceed. Returns a
    GeometryCollection of resulting polygons.
    """
    import itertools
    next_id = itertools.count()

    G = nx.Graph()
    geoms = {}

    is_layer = hasattr(polygons, "GetLayerDefn")
    field_names = []
    if is_layer:
        defn = polygons.GetLayerDefn()
        field_names = [defn.GetFieldDefn(i).GetNameRef() for i in range(defn.GetFieldCount())]
        polygons.ResetReading()
        source_iter = polygons  # yields Features
    else:
        source_iter = _iter_polygons(polygons)  # yields Geometries

    for item in source_iter:
        if is_layer:
            geom = item.GetGeometryRef()
            attrs = {name: item.GetField(name) for name in field_names}
        else:
            geom = item
            attrs = {}

        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        area = geom.Area()

        fid = next(next_id)
        geoms[fid] = geom
        G.add_node(fid, area=area, attrs=attrs)

    print(f"Loaded {G.number_of_nodes()} polygons")
    if G.number_of_nodes() == 0:
        return []

    def build_edges(node_ids):
        """Adds adjacency edges (weighted by shared boundary length) to G
        for every Delaunay-candidate pair among node_ids that actually
        touch; returns how many edges were added."""
        pairs = _delaunay_candidate_pairs(node_ids, geoms)
        added = 0
        for a, b in pairs:
            if G.has_edge(a, b):
                continue
            if not geoms[a].Intersects(geoms[b]):
                continue
            length = _shared_boundary_length(geoms[a], geoms[b])
            if length > 0:
                G.add_edge(a, b, weight=length)
                added += 1
        return added

    build_edges(list(G.nodes))
    print(f"Initial adjacency graph: {G.number_of_edges()} edges")

    merges_since_retri = 0
    iteration = 0

    while True:
        iteration += 1

        small = next((n for n, d in G.nodes(data=True) if d["area"] < min_area), None)
        if small is None:
            break

        if G.number_of_nodes() == 1:
            print(f"Iteration {iteration}: polygon (area={G.nodes[small]['area']:.2f}) "
                  f"is the only polygon left -- nothing to merge into")
            break

        if G.degree(small) == 0:
            # No touching neighbor found yet -- retriangulate against the full
            # graph first, since earlier merges may have created an adjacency
            # this node's original edge-check missed.
            build_edges(list(G.nodes))

        if G.degree(small) == 0:
            # Still isolated (e.g. only corner-touches another polygon, or
            # sits in a hairline gap left by noding, so Intersects() never
            # sees it as adjacent). Force a merge with the geometrically
            # nearest polygon by distance rather than leaving a permanent
            # sub-threshold sliver in the output -- every small polygon must
            # end up merged, never just marked done.
            nearest = min((n for n in G.nodes if n != small),
                          key=lambda n: geoms[small].Distance(geoms[n]))
            gap = geoms[small].Distance(geoms[nearest])
            if gap > 0:
                # Bridge the gap first -- otherwise Union() below just produces
                # a MultiPolygon of two disjoint footprints, and the "keep
                # largest part" step further down would silently delete
                # whichever one is smaller instead of actually merging them.
                geoms[small] = geoms[small].Buffer(gap + 0.01)
            G.add_edge(small, nearest, weight=_shared_boundary_length(geoms[small], geoms[nearest]))
            print(f"Iteration {iteration}: polygon (area={G.nodes[small]['area']:.2f}) has no "
                  f"touching neighbor -- force-merging into nearest polygon by distance "
                  f"(area={G.nodes[nearest]['area']:.2f})")

        neighbor = max(G[small].items(), key=lambda kv: kv[1]["weight"])[0]
        print(f"Iteration {iteration}: merging polygon (area={G.nodes[small]['area']:.2f}) "
              f"into neighbor (area={G.nodes[neighbor]['area']:.2f}), "
              f"shared border={G[small][neighbor]['weight']:.2f}")

        merged_geom = geoms[small].Union(geoms[neighbor])
        if ogr.GT_Flatten(merged_geom.GetGeometryType()) == ogr.wkbMultiPolygon:
            merged_geom = max(
                (merged_geom.GetGeometryRef(i) for i in range(merged_geom.GetGeometryCount())),
                key=lambda g: g.Area()
            ).Clone()

        # Repair immediately -- don't let numerical defects from this Union()
        # compound silently across dozens of subsequent merges.
        if not merged_geom.IsValid():
            merged_geom = merged_geom.MakeValid()

        small_area, neighbor_area = G.nodes[small]["area"], G.nodes[neighbor]["area"]
        bigger_attrs = (G.nodes[small]["attrs"] if small_area >= neighbor_area
                        else G.nodes[neighbor]["attrs"])
        smaller_attrs = (G.nodes[neighbor]["attrs"] if small_area >= neighbor_area
                         else G.nodes[small]["attrs"])
        merged_attrs = bigger_attrs if keep_attributes_from == "largest" else smaller_attrs

        G = nx.contracted_nodes(G, neighbor, small, self_loops=False)
        del geoms[small]
        geoms[neighbor] = merged_geom
        G.nodes[neighbor]["area"] = merged_geom.Area()
        G.nodes[neighbor]["attrs"] = merged_attrs
        G.nodes[neighbor].pop("contraction", None)

        for nbr in list(G[neighbor]):
            length = _shared_boundary_length(geoms[neighbor], geoms[nbr])
            if length > 0:
                G[neighbor][nbr]["weight"] = length
            else:
                G.remove_edge(neighbor, nbr)

        merges_since_retri += 1
        if retriangulate_every and merges_since_retri >= retriangulate_every:
            print(f"  Re-triangulating ({G.number_of_nodes()} nodes remaining)...")
            added = build_edges(list(G.nodes))
            print(f"  Added {added} new candidate edges")
            merges_since_retri = 0

    print(f"Done. {G.number_of_nodes()} polygons remain")

    result = ogr.Geometry(ogr.wkbGeometryCollection)
    for n in G.nodes:
        result.AddGeometry(geoms[n])
    return result

def snap_to_grid(geom, grid_size=GRID_SIZE):
    """Snaps geom's coordinates to grid_size via SetPrecision, falling back
    to fix_topology's coordinate rounding if unavailable."""
    try:
        g = geom.SetPrecision(grid_size)
        if g is not None:
            return g
    except Exception:
        pass
    # portable fallback if SetPrecision isn't available
    return fix_topology._round_geometry_coords(geom, precision=3)

def buffer_sandwich(geom, eps=0.01):  # 1cm -- meaningful in a metric CRS
    """eps must be sized to the CRS units actually in use at call time.
    In a projected/metric CRS, 1e-6 (1 micron) is far below GEOS's
    effective precision and has no real effect."""
    try:
        g = geom.Buffer(eps).Buffer(-eps)
        if g is not None and g.IsValid():
            return g
    except Exception:
        pass
    return geom

def safe_intersection(geom_a, geom_b, context=""):
    """Returns geom_a.Intersection(geom_b), retrying with buffer_sandwich()
    on one or both sides if GEOS raises a RuntimeError; returns None if
    every attempt fails."""
    try:
        return geom_a.Intersection(geom_b)
    except RuntimeError as e:
        print(f"{context}: Intersection failed ({e})")
        for label, g in (("geom_a", geom_a), ("geom_b", geom_b)):
            print(f"  {label}: IsValid={g.IsValid()} IsSimple={g.IsSimple()} "
                  f"NumParts={g.GetGeometryCount()} Area={g.Area():.2f} "
                  f"Envelope={g.GetEnvelope()}")

        try:
            a2 = buffer_sandwich(geom_a)
            return a2.Intersection(geom_b)
        except RuntimeError as e2:
            print(f"{context}: still failing ({e2}), retrying with both sides sandwiched")
            try:
                b2 = buffer_sandwich(geom_b)
                return a2.Intersection(b2)
            except RuntimeError as e3:
                print(f"{context}: unrecoverable ({e3})")
                return None

def robust_repair(geom, grid_size=GRID_SIZE):
    """Repair a geometry for safe use in GEOS overlay operations.
    Snapping ALWAYS runs, even on already-valid geometries -- 'side
    location conflict' is an overlay-algorithm robustness issue, not an
    OGC-validity issue, so IsValid()==True gives no guarantee Intersection()
    will succeed against another geometry."""
    if geom is None:
        return None

    try:
        snapped = geom.SetPrecision(grid_size)
        if snapped is not None and snapped.IsValid():
            geom = snapped
    except Exception:
        pass  # GDAL < 3.9: SetPrecision unavailable, falls through

    if geom.IsValid():
        return geom

    for attempt in (
        lambda g: g.MakeValid(),
        lambda g: g.SetPrecision(grid_size).MakeValid(),
        lambda g: g.MakeValid().SetPrecision(grid_size),
        lambda g: g.Buffer(0),
    ):
        try:
            fixed = attempt(geom)
            if fixed is not None and fixed.IsValid():
                return fixed
        except Exception:
            continue
    return None
