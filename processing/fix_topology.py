#!/usr/bin/env python3
"""
Detect and repair single-geometry topology (validity) errors using only
GDAL/OGR (no shapely/fiona required).

Strategy per invalid feature, in increasing order of aggressiveness:
    1. snap_to_grid()               - collapses near-duplicate vertices onto
                                       a shared precision grid. Fixes GEOS
                                       "side location conflict" / "TopologyException"
                                       errors caused by coordinates that are
                                       almost-but-not-quite identical.
    2. RemoveDuplicatePoints()      - fixes repeated/zero-length vertices
    3. MakeValid()                  - proper GEOS repair (splits rings etc.)
    4. MakeValid(METHOD=STRUCTURE)  - more robust GEOS algorithm for
                                       degenerate/near-duplicate node cases
                                       (needs GDAL>=3.4 & GEOS>=3.10)
    5. Buffer(0)                    - classic self-intersection fix
    6. SimplifyPreserveTopology(t)  - last resort, alters shape
    7. remove_offending_vertex()    - brute-force: drop the single vertex
                                       that is causing the invalidity

Valid / repaired features go to OUTPUT_PATH.
Anything that could not be repaired goes to ERROR_PATH for manual review.

Requires GDAL built with GEOS support (almost all packaged builds are).
"""

from osgeo import ogr, gdal

gdal.UseExceptions()
ogr.UseExceptions()

INPUT_PATH = "input.shp"
OUTPUT_PATH = "output_fixed.gpkg"
ERROR_PATH = "output_errors.gpkg"

# Tolerance used only for the SimplifyPreserveTopology fallback.
# Set this relative to your data's coordinate units (e.g. meters or degrees).
SIMPLIFY_TOLERANCE = 0.0001

# Grid size for coordinate snapping, in the same units as your data
# (e.g. 0.001 = 1mm for projected data in meters). This collapses
# near-duplicate vertices that cause GEOS "side location conflict" /
# TopologyException errors at shared nodes between features.
GRID_SIZE = 0.001


def find_features_near_point(layer, x, y, tolerance=1e-3):
    """
    Diagnostic helper: list FIDs of features that have a vertex within
    `tolerance` of (x, y). Useful for figuring out which features share
    the coordinate reported in a GEOS TopologyException.

    Usage:
        ds = ogr.Open(INPUT_PATH)
        lyr = ds.GetLayer()
        find_features_near_point(lyr, 449773.86022395582, 5770986.6428437922)
    """
    layer.ResetReading()
    hits = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        if _geom_has_point_near(geom, x, y, tolerance):
            hits.append(feat.GetFID())
    layer.ResetReading()
    print(f"Features with a vertex within {tolerance} of ({x}, {y}): {hits}")
    return hits


def _geom_has_point_near(geom, x, y, tolerance):
    gtype = ogr.GT_Flatten(geom.GetGeometryType())
    if gtype in (ogr.wkbPolygon, ogr.wkbLineString, ogr.wkbLinearRing):
        if gtype == ogr.wkbPolygon:
            for i in range(geom.GetGeometryCount()):
                if _geom_has_point_near(geom.GetGeometryRef(i), x, y, tolerance):
                    return True
            return False
        for i in range(geom.GetPointCount()):
            px, py, _ = geom.GetPoint(i)
            if abs(px - x) <= tolerance and abs(py - y) <= tolerance:
                return True
        return False
    else:
        # Multi* / GeometryCollection - recurse into parts
        for i in range(geom.GetGeometryCount()):
            if _geom_has_point_near(geom.GetGeometryRef(i), x, y, tolerance):
                return True
        return False


def snap_to_grid(geom, grid_size=GRID_SIZE):
    """
    Snap all coordinates onto a fixed precision grid, collapsing
    near-duplicate vertices onto identical points. This is the primary
    fix for GEOS "side location conflict" / TopologyException errors
    that occur at a shared node between multiple features.

    Uses OGRGeometry.SetPrecision() when available (GDAL >= 3.9, which
    wraps GEOS's precision-model snapping and also fixes resulting
    self-intersections). Falls back to manual coordinate rounding on
    older GDAL, which is more portable but only collapses points, it
    does not re-node the geometry.
    """
    try:
        # GDAL >= 3.9: proper GEOS precision-model snap
        g = geom.SetPrecision(grid_size)
        if g is not None:
            return g
    except (AttributeError, Exception):
        pass

    # Fallback: manual coordinate rounding
    decimals = max(0, len(str(grid_size).split(".")[-1])) if grid_size < 1 else 0
    return _round_geometry_coords(geom, decimals)


def _round_geometry_coords(geom, precision):
    """Round every coordinate in geom to `precision` decimal places."""
    gtype = ogr.GT_Flatten(geom.GetGeometryType())

    def round_ring(ring):
        new_ring = ogr.Geometry(ring.GetGeometryType())
        for i in range(ring.GetPointCount()):
            x, y, z = ring.GetPoint(i)
            new_ring.AddPoint(round(x, precision), round(y, precision), round(z, precision))
        return new_ring

    if gtype == ogr.wkbPolygon:
        new_poly = ogr.Geometry(ogr.wkbPolygon)
        for i in range(geom.GetGeometryCount()):
            new_poly.AddGeometry(round_ring(geom.GetGeometryRef(i)))
        return new_poly
    elif gtype == ogr.wkbMultiPolygon:
        new_multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for i in range(geom.GetGeometryCount()):
            poly = geom.GetGeometryRef(i)
            new_poly = ogr.Geometry(ogr.wkbPolygon)
            for j in range(poly.GetGeometryCount()):
                new_poly.AddGeometry(round_ring(poly.GetGeometryRef(j)))
            new_multi.AddGeometry(new_poly)
        return new_multi
    elif gtype == ogr.wkbLineString:
        return round_ring(geom)
    else:
        return geom.Clone()


def remove_offending_vertex(geom):
    """
    Brute-force attempt to find and remove the single vertex responsible
    for an invalid geometry (e.g. a spike that creates a self-intersection).

    Tries removing each vertex of each ring in turn and re-checks validity.
    Returns a repaired clone, or None if no single-vertex removal fixes it.

    NOTE: this only handles the simple case where ONE vertex is the cause.
    Complex multi-point self-intersections will not be fixed by this.
    """
    gtype = geom.GetGeometryType()
    flat_type = ogr.GT_Flatten(gtype)

    if flat_type == ogr.wkbPolygon:
        rings = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
    elif flat_type == ogr.wkbLineString:
        rings = [geom]
    else:
        return None  # only handling Polygon / LineString here

    for ring_idx, ring in enumerate(rings):
        npoints = ring.GetPointCount()
        for i in range(npoints):
            # rebuild the ring without point i
            trial_geom = geom.Clone()
            trial_ring = (
                trial_geom.GetGeometryRef(ring_idx)
                if flat_type == ogr.wkbPolygon
                else trial_geom
            )
            new_ring = ogr.Geometry(ring.GetGeometryType())
            for j in range(npoints):
                if j == i:
                    continue
                new_ring.AddPoint(*ring.GetPoint(j))
            if flat_type == ogr.wkbPolygon:
                new_ring.CloseRings()

            if flat_type == ogr.wkbPolygon:
                new_poly = ogr.Geometry(ogr.wkbPolygon)
                for k, r in enumerate(rings):
                    new_poly.AddGeometry(new_ring if k == ring_idx else r)
                candidate = new_poly
            else:
                candidate = new_ring

            if candidate.IsValid():
                return candidate

    return None


def fix_geometry(geom):
    """
    Try repair strategies in order. Returns (fixed_geom, method_used)
    or (None, None) if nothing worked.
    """
    if geom is None:
        return None, None

    if geom.IsValid():
        return geom, "already_valid"

    # 1. Snap to grid - fixes "side location conflict" / TopologyException
    #    caused by near-duplicate vertices at a shared node
    try:
        g = snap_to_grid(geom)
        if g is not None and g.IsValid():
            return g, "snap_to_grid"
    except Exception:
        pass

    # 2. Remove duplicate consecutive points
    try:
        g = geom.Clone()
        g.RemoveDuplicatePoints()
        if g.IsValid():
            return g, "remove_duplicate_points"
    except Exception:
        pass

    # 3. MakeValid, default algorithm (requires GEOS >= 3.8 / GDAL >= 3.0)
    try:
        g = geom.MakeValid()
        if g is not None and g.IsValid():
            return g, "make_valid"
    except Exception:
        pass

    # 4. MakeValid, STRUCTURE method - handles degenerate/near-duplicate
    #    nodes more robustly (requires GDAL >= 3.4 & GEOS >= 3.10)
    try:
        g = geom.MakeValid(["METHOD=STRUCTURE"])
        if g is not None and g.IsValid():
            return g, "make_valid_structure"
    except Exception:
        pass

    # 5. Snap to grid THEN MakeValid - combines both fixes for the
    #    stubborn cases where snapping alone leaves a residual issue
    try:
        g = snap_to_grid(geom)
        g = g.MakeValid()
        if g is not None and g.IsValid():
            return g, "snap_to_grid+make_valid"
    except Exception:
        pass

    # 6. Buffer(0)
    try:
        g = geom.Buffer(0)
        if g is not None and g.IsValid():
            return g, "buffer_zero"
    except Exception:
        pass

    # 7. SimplifyPreserveTopology
    try:
        g = geom.SimplifyPreserveTopology(SIMPLIFY_TOLERANCE)
        if g is not None and g.IsValid():
            return g, "simplify_preserve_topology"
    except Exception:
        pass

    # 8. Brute-force single-vertex removal
    try:
        g = remove_offending_vertex(geom)
        if g is not None and g.IsValid():
            return g, "removed_offending_vertex"
    except Exception:
        pass

    return None, None


def main():
    src_ds = ogr.Open(INPUT_PATH)
    if src_ds is None:
        raise RuntimeError(f"Could not open {INPUT_PATH}")
    src_layer = src_ds.GetLayer()

    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(OUTPUT_PATH)
    err_ds = driver.CreateDataSource(ERROR_PATH)

    out_layer = out_ds.CreateLayer(
        src_layer.GetName(),
        srs=src_layer.GetSpatialRef(),
        geom_type=src_layer.GetGeomType(),
    )
    err_layer = err_ds.CreateLayer(
        src_layer.GetName() + "_errors",
        srs=src_layer.GetSpatialRef(),
        geom_type=src_layer.GetGeomType(),
    )

    src_defn = src_layer.GetLayerDefn()
    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)
        err_layer.CreateField(field_defn)

    err_layer.CreateField(ogr.FieldDefn("orig_fid", ogr.OFTInteger64))
    reason_field = ogr.FieldDefn("invalid_reason", ogr.OFTString)
    reason_field.SetWidth(254)
    err_layer.CreateField(reason_field)

    n_total = n_fixed = n_deleted = n_valid = 0

    for feat in src_layer:
        n_total += 1
        geom = feat.GetGeometryRef()

        if geom is None:
            continue

        if geom.IsValid():
            n_valid += 1
            new_feat = ogr.Feature(out_layer.GetLayerDefn())
            new_feat.SetFrom(feat)
            new_feat.SetGeometry(geom)
            out_layer.CreateFeature(new_feat)
            new_feat = None
            continue

        reason = geom.IsValidReason() if hasattr(geom, "IsValidReason") else "unknown"
        fixed_geom, method = fix_geometry(geom)

        if fixed_geom is not None:
            n_fixed += 1
            new_feat = ogr.Feature(out_layer.GetLayerDefn())
            new_feat.SetFrom(feat)
            new_feat.SetGeometry(fixed_geom)
            out_layer.CreateFeature(new_feat)
            new_feat = None
            print(f"FID {feat.GetFID()}: fixed via {method} ({reason})")
        else:
            n_deleted += 1
            err_feat = ogr.Feature(err_layer.GetLayerDefn())
            err_feat.SetFrom(feat)
            err_feat.SetGeometry(geom)
            err_feat.SetField("orig_fid", feat.GetFID())
            err_feat.SetField("invalid_reason", reason)
            err_layer.CreateFeature(err_feat)
            err_feat = None
            print(f"FID {feat.GetFID()}: could not be repaired -> error layer ({reason})")

    print(f"\nTotal features:  {n_total}")
    print(f"Already valid:   {n_valid}")
    print(f"Fixed:           {n_fixed}")
    print(f"Unfixable:       {n_deleted}  (see {ERROR_PATH})")

    src_ds = out_ds = err_ds = None


if __name__ == "__main__":
    main()
