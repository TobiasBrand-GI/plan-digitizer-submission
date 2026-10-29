from osgeo import gdal, ogr, osr
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import os
import rasterio
from processing import utils
from collections import defaultdict


def polygonize(input_img, dst_ds, crs_wkt):

    #  get raster datasource
    src_ds = input_img
    srcband = src_ds.GetRasterBand(1)

    dst_layername = 'test'

    # CRS passed in as WKT (read via rasterio) -- the system GDAL's
    # GetSpatialRef() segfaults on some rasters' CRS (e.g. tif/s_2_08.tif)
    sp_ref = osr.SpatialReference()
    sp_ref.ImportFromWkt(crs_wkt)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)

    fld = ogr.FieldDefn("HA", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("HA")

    gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)

    #layer = source.GetLayer(0)

    # Collect FIDs of features to delete
    fids_to_delete = []
    for feature in dst_layer:
        if feature.GetField('HA') != 255:
            fids_to_delete.append(feature.GetFID())

    # Delete them
    for fid in fids_to_delete:
        dst_layer.DeleteFeature(fid)

    dst_layer.SetAttributeFilter("HA = 255")
    mask_geoms = ogr.Geometry(ogr.wkbGeometryCollection)

    for feature in dst_layer:
        geom = feature.GetGeometryRef()

        if geom is not None:
            exterior_ring = geom.GetGeometryRef(0)  # index 0 = exterior ring
            filled = ogr.Geometry(ogr.wkbPolygon)
            filled.AddGeometry(exterior_ring)

            simplified = filled.SimplifyPreserveTopology(1.0)
            #neg_buffered = simplified.Buffer(-1, 0)
            #pos_buffered = neg_buffered.Buffer(2,1)
            #simplified_buf = pos_buffered.SimplifyPreserveTopology(1.0)
            neg_buffered = utils.square_buffer_geometry(simplified, -1)
            pos_buffered = utils.square_buffer_geometry(neg_buffered,1.5)
            simplified_buf = pos_buffered.SimplifyPreserveTopology(1.0)
            if not simplified_buf.IsEmpty():
                mask_geoms.AddGeometry(utils.keep_polygonal(simplified_buf))
                feature.SetGeometry(simplified_buf)
                dst_layer.SetFeature(feature)  # write the change back to the layer
            
    dst_ds.FlushCache()  # ensure changes are written to disk
    source = None  # close the file

    del src_ds
    del dst_ds

    return mask_geoms

def build_graph_greedy_matching(points, max_dist, k_candidates=6):
    """
    Builds a simple-path-like graph by greedily adding the globally
    shortest candidate edges first, enforcing:
      - each point ends with degree <= 2
      - no premature cycle is closed (union-find check)
    This avoids the asymmetric k-NN problem where a point can gain
    edges it didn't itself request.
    """
    n = len(points)
    tree = cKDTree(points)

    # gather candidate edges (a bit more generous than k=2, since
    # we'll filter properly during greedy selection)
    dists, idxs = tree.query(points, k=k_candidates + 1)

    candidates = []
    seen = set()
    for i in range(n):
        for rank in range(1, k_candidates + 1):
            j = idxs[i, rank]
            d = dists[i, rank]
            if d > max_dist:
                continue
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                seen.add(edge)
                candidates.append((d, i, j))

    candidates.sort(key=lambda e: e[0])  # shortest edges first

    degree = [0] * n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for d, i, j in candidates:
        if degree[i] >= 2 or degree[j] >= 2:
            continue  # would create a branch — skip

        ri, rj = find(i), find(j)
        if ri == rj:
            continue  # would close a cycle prematurely — skip
            # (remove this check if closed loops are actually valid
            # boundary shapes in your data)

        G.add_edge(i, j)
        degree[i] += 1
        degree[j] += 1
        parent[ri] = rj

    return G

def dots_to_lines_gdal(dataset, dots1, output_path, spatial_res,
                        min_dots_per_line=2, driver_name="GeoJSON"):
    """
    Connect nearby dots into ordered line paths and write them using GDAL/OGR.

    dataset: rasterio dataset (used for transform + crs/wkt)
    dots1: list of (x, y, radius) in pixel coordinates
    driver_name: "GeoJSON", "ESRI Shapefile", "GPKG", etc.
    """
    if len(dots1) < 2:
        raise ValueError("Need at least 2 dots to build a line")

    points = np.array([(x, y) for x, y, r in dots1])

    G = build_graph_greedy_matching(points, (5 / spatial_res), k_candidates=6)

    # --- set up OGR output ---
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Unknown OGR driver: {driver_name}")

    # Remove existing file if present (GDAL won't overwrite in place)
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)

    ds = driver.CreateDataSource(output_path)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(utils.dataset_crs(dataset).to_wkt())  # reuse CRS from rasterio dataset

    layer = ds.CreateLayer("boundary_lines", srs, ogr.wkbLineString)

    # Optional: add a field to store number of dots per line, useful for QA
    field_defn = ogr.FieldDefn("n_dots", ogr.OFTInteger)
    layer.CreateField(field_defn)

    for component in nx.connected_components(G):
        if len(component) < min_dots_per_line:
            continue

        subgraph = G.subgraph(component)
        endpoints = [n for n in subgraph.nodes if subgraph.degree(n) == 1]

        if len(endpoints) >= 1:
            ordered_idx = list(nx.dfs_preorder_nodes(subgraph, source=endpoints[0]))
        else:
            comp_points = points[list(component)]
            mean = comp_points.mean(axis=0)
            centered = comp_points - mean
            _, _, vh = np.linalg.svd(centered)
            projection = centered @ vh[0]
            order = np.argsort(projection)
            ordered_idx = [list(component)[i] for i in order]

        ordered_points_px = points[ordered_idx]

        ordered_points_px = utils.remove_spikes(ordered_points_px, angle_threshold_deg=45)

        # Pixel -> geo coordinates via rasterio transform (same as before)
        xs, ys = rasterio.transform.xy(dataset.transform,
                                        ordered_points_px[:, 1],
                                        ordered_points_px[:, 0])

        # Build OGR LineString geometry
        line_geom = ogr.Geometry(ogr.wkbLineString)

        for x, y in zip(xs, ys):
            line_geom.AddPoint(x, y)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(line_geom)
        feature.SetField("n_dots", len(ordered_idx))
        layer.CreateFeature(feature)

        feature = None  # release, GDAL/OGR convention

    ds = None  # flush + close, writes to disk

def postprocess_dotted_lines(in_path, out_path):
    if in_path is None:
        raise ValueError("Path cannot be None")

    src_ds = ogr.Open(in_path, 0)
    if src_ds is None:
        raise RuntimeError(f"Could not open input: {in_path}")

    src_lyr = src_ds.GetLayer(0)
    if src_lyr is None:
        raise RuntimeError("Input layer not found")

    srs = src_lyr.GetSpatialRef()
    src_defn = src_lyr.GetLayerDefn()

    out_driver = ogr.GetDriverByName("geojson")
    if out_driver is None:
        raise RuntimeError(f"Driver not available: geojson")

    out_ds = out_driver.CreateDataSource(out_path)
    if out_ds is None:
        raise RuntimeError(f"Could not create output: {out_path}")


    print("Buffering Lines")
    out_lyr = out_ds.CreateLayer("buffered_lines", srs=srs, geom_type=ogr.wkbPolygon)

    for i in range(src_defn.GetFieldCount()):
        out_lyr.CreateField(src_defn.GetFieldDefn(i))
    out_defn = out_lyr.GetLayerDefn()

    collection = ogr.Geometry(ogr.wkbGeometryCollection)

    src_lyr.ResetReading()
    for feat in src_lyr:
        geom = feat.GetGeometryRef()
        buffered = utils.square_buffer_geometry(geom, 10.0)
        if buffered is not None:
            collection.AddGeometry(buffered)

    print("Dissolving Buffers")
    if collection.GetGeometryCount() == 0:
        raise RuntimeError(f"No Buffers found")
    dissolved_geom = collection.UnaryUnion()

    print("Sampling Points")
    if dissolved_geom.GetGeometryCount() == 0:
        raise RuntimeError(f"No Merged Buffers found")
    points_along_buffers = utils.sample_points_along_boundary(dissolved_geom)

    print("Creating Voronoi")
    if points_along_buffers.GetGeometryCount() == 0:
        raise RuntimeError(f"Found no Vertices around the Buffers")
    voronoi = utils.voronoi_polygons(points_along_buffers)

    print("Extracting Lines from Voronoi")
    if len(voronoi) == 0:
        raise RuntimeError(f"Found no Vertices around the Buffers")
    voronoi_lines = utils.polygons_to_lines(voronoi)

    print("Filtering Lines")
    filtered_lines = utils.filter_lines_within(voronoi_lines, dissolved_geom)

    print("Saving Lines")
    for line in filtered_lines:
        line.FlattenTo2D()
        feat = ogr.Feature(out_defn)
        feat.SetGeometry(line)
        out_lyr.CreateFeature(feat)
        feat = None


    src_ds = None
    out_ds = None


def intersect_usage_types_by_lines(umring_path, alkis_path, lines_path, out_path, mask_polygons,
                                   field_name="nutzart", allowed_values=None):
    """
    1. Clip the ALKIS shapefile to the Umring (boundary) polygon from a GeoJSON.
    2. Dissolve the clipped geometries by `field_name` (e.g. "nutzungsart"),
       optionally restricted to `allowed_values`.
    3. Convert the dissolved polygons' boundaries to line segments.
    4. Merge those segments with the existing lines from `lines_path`.
    5. Node the combined linework and polygonize it into new polygons.
    6. Save the resulting polygons to `out_path` as GeoJSON.

    Parameters
    ----------
    allowed_values : iterable, optional
        If given, only ALKIS features whose `field_name` value is in this
        set are considered -- everything else is skipped before clipping/
        dissolving. If None (default), all values are included.
    """
    allowed_set = set(allowed_values) if allowed_values is not None else None
    # 1. Umring boundary (dissolve all its features into one clip geometry)
    umring_ds = ogr.Open(umring_path, 0)
    if umring_ds is None:
        raise RuntimeError(f"Could not open {umring_path}")
    umring_geom = utils.read_layer_geom(umring_ds)
    if umring_geom is None:
        raise RuntimeError("Umring has no geometry")
    umring_geom.AssignSpatialReference(umring_ds.GetLayer(0).GetSpatialRef())

    # 2. Open ALKIS, clip each feature to the Umring, group by field_name
    alkis_ds = ogr.Open(alkis_path, 0)
    if alkis_ds is None:
        raise RuntimeError(f"Could not open {alkis_path}")
    alkis_lyr = alkis_ds.GetLayer(0)
    srs = alkis_lyr.GetSpatialRef()

    umring_geom.TransformTo(srs)  # Umring is WGS84; reproject to match ALKIS's CRS

    grouped = defaultdict(lambda: ogr.Geometry(ogr.wkbGeometryCollection))
    alkis_lyr.ResetReading()
    for feat in alkis_lyr:
        value = feat.GetField(field_name)
        if isinstance(value, str):
            value = value.strip()  # avoid stray whitespace splitting one usage type into several groups
        if allowed_set is not None and value not in allowed_set:
            continue

        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        clipped = utils.keep_polygonal(geom.Intersection(umring_geom))
        if clipped is None:
            continue
        if not clipped.IsValid():
            clipped = clipped.MakeValid()  # invalid/self-intersecting input can prevent a clean dissolve
            clipped = utils.keep_polygonal(clipped)
            if clipped is None:
                continue

        # When a filter is active, dissolve everything that passes it together
        # (regardless of which allowed value it has) rather than keeping each
        # distinct field value in its own separate dissolve group.
        group_key = "__filtered__" if allowed_set is not None else value
        grouped[group_key].AddGeometry(clipped)

    dissolved_polygons = []
    for value, collection in grouped.items():
        if collection.GetGeometryCount() == 0:
            continue
        dissolved = collection.UnaryUnion()
        if dissolved is not None and not dissolved.IsEmpty():
            dissolved_polygons.append(dissolved)

    # 3. Dissolved polygons -> boundary line segments
    polygon_lines = []
    for poly in dissolved_polygons:
        polygon_lines.extend(
            utils.polygons_to_lines([poly] if isinstance(poly, ogr.Geometry) else poly)
        )

    # 4. Existing lines from lines_path
    lines_ds = ogr.Open(lines_path, 0)
    if lines_ds is None:
        raise RuntimeError(f"Could not open {lines_path}")
    lines_lyr = lines_ds.GetLayer(0)
    lines_lyr.ResetReading()
    existing_lines = []
    for feat in lines_lyr:
        geom = feat.GetGeometryRef()
        if geom is not None:
            existing_lines.extend(line.Clone() for line in utils._explode_to_linestrings(geom))

    all_lines = polygon_lines + existing_lines

    # 5. Node the combined linework, then polygonize it
    line_collection = ogr.Geometry(ogr.wkbGeometryCollection)
    for line in all_lines:
        line_collection.AddGeometry(line)
    noded = line_collection.UnaryUnion()
    if noded is not None and not noded.IsValid():
        noded = noded.MakeValid()
    polygons = noded.Polygonize()  # GeometryCollection of resulting Polygons

    polygons_small_removed = utils.eliminate_small_polygons(polygons, min_area=150.0)

    final_clipped = ogr.Geometry(ogr.wkbGeometryCollection)

    # mask_polygons doesn't change per iteration -> prep it ONCE, outside the loop
    umring_geom_before = umring_geom.ExportToWkb()
    umring_geom = utils.robust_repair(umring_geom)

    # mask_polygons arrives as a raw GeometryCollection of per-raster-region
    # polygons (from polygonize()) that can overlap each other (adjacent mask
    # regions sharing/duplicating slivers). Every member can individually pass
    # IsValid(), so robust_repair()'s validity-gated logic never touches this --
    # but a GeometryCollection with overlapping members is exactly what trips
    # GEOS's "side location conflict" during Intersects() against it. Dissolve
    # away the overlaps first.
    mask_polygons = mask_polygons.UnaryUnion()
    mask_polygons = utils.robust_repair(mask_polygons.MakeValid() if not mask_polygons.IsValid() else mask_polygons)


    if umring_geom is None:
        raise RuntimeError("umring_geom could not be repaired — check it manually.")

    if umring_geom.ExportToWkb() == umring_geom_before:
        print("WARNING: robust_repair made no change to umring_geom — "
              "if failures persist, SetPrecision may be unavailable (check GDAL/GEOS version).")

    for i in range(polygons_small_removed.GetGeometryCount()):
        geom = polygons_small_removed.GetGeometryRef(i)
        if geom is None:
            continue

        if not geom.IsValid():
            geom = geom.MakeValid()
            if geom is None:
                print(f"Feature {i}: could not repair input geometry, skipping.")
                continue

        # Skip the overlay entirely when it isn't needed -- most pieces are
        # already fully inside umring_geom (their edges came from a prior
        # clip against the SAME umring_geom), and re-intersecting those is
        # both wasted work and the likely source of the side-location-conflict.
        try:
            fully_inside = geom.Within(umring_geom)
        except RuntimeError:
            fully_inside = False  # predicate failed too -> fall through to overlay path

        if fully_inside:
            piece = geom
        else:
            raw_piece = utils.safe_intersection(geom, umring_geom, context=f"Feature {i}")
            if raw_piece is None:
                continue
            piece = utils.keep_polygonal(raw_piece)
            if piece is None:
                continue

        piece.FlattenTo2D()

        if not piece.IsValid():
            piece = piece.MakeValid()
            if piece is None:
                continue

        if piece.IsEmpty():
            continue

        try:
            if piece.Intersects(mask_polygons):
                final_clipped.AddGeometry(piece)
        except RuntimeError as e:
            print(f"Feature {i}: Intersects failed ({e}), retrying after snap-to-grid")
            try:
                piece_snapped = utils.snap_to_grid(piece)
                if piece_snapped.Intersects(mask_polygons):
                    final_clipped.AddGeometry(piece_snapped)
            except Exception as e2:
                print(f"Feature {i}: still failing after snap ({e2}), skipping.")
                continue

    if final_clipped.GetGeometryCount() == 0:
        print("No areas match the filter criteria.")

    # Simplify the final polygons, preserving topology (won't introduce
    # self-intersections the way a naive Simplify() can).
    simplified_clipped = ogr.Geometry(ogr.wkbGeometryCollection)
    for i in range(final_clipped.GetGeometryCount()):
        piece = final_clipped.GetGeometryRef(i).SimplifyPreserveTopology(1.0)
        if piece is not None and not piece.IsEmpty():
            simplified_clipped.AddGeometry(piece)
    final_clipped = simplified_clipped

    # 6. Save to GeoJSON
    driver = ogr.GetDriverByName("GeoJSON")
    try:
        if driver.Open(out_path) is not None:
            driver.DeleteDataSource(out_path)  # avoid mixing in a stale/old output file
    except:
        print("No GeoJSON was cleared")
    out_ds = driver.CreateDataSource(out_path)
    out_lyr = out_ds.CreateLayer("usage_polygons", srs=srs, geom_type=ogr.wkbPolygon)
    out_defn = out_lyr.GetLayerDefn()

    for i in range(final_clipped.GetGeometryCount()):
        poly = final_clipped.GetGeometryRef(i)
        feat = ogr.Feature(out_defn)
        feat.SetGeometry(poly)
        out_lyr.CreateFeature(feat)
        feat = None

    out_ds = None  # flush/close