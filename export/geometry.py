from lxml import etree
from osgeo import ogr, osr

ogr.UseExceptions()

GML_NS = "http://www.opengis.net/gml/3.2"
EPSG = 25832
SRS_NAME = f"EPSG:{EPSG}"


def _target_srs():
    """Returns an osr.SpatialReference for EPSG:25832 with GIS (x, y) axis order."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _signed_area(points):
    """Shoelace formula; > 0 means counter-clockwise."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0, *_), (x1, y1, *_) in zip(points, points[1:])) / 2.0


def _oriented_ring(ring, counter_clockwise):
    """Returns a copy of ring with point order flipped if needed to match the
    requested winding direction."""
    points = ring.GetPoints()
    if (_signed_area(points) > 0) != counter_clockwise:
        points = points[::-1]
    oriented = ogr.Geometry(ogr.wkbLinearRing)
    for x, y, *_ in points:
        oriented.AddPoint_2D(x, y)
    return oriented


def _oriented_polygon(polygon):
    """Returns a copy of polygon with the exterior ring counter-clockwise and
    every interior ring clockwise, as GML requires."""
    oriented = ogr.Geometry(ogr.wkbPolygon)
    for i in range(polygon.GetGeometryCount()):
        oriented.AddGeometry(_oriented_ring(polygon.GetGeometryRef(i), counter_clockwise=(i == 0)))
    return oriented


def _polygons(geom):
    """Every Polygon inside geom, flattening Multi*/GeometryCollection;
    lines and points (MakeValid can produce them) are dropped."""
    flat_type = ogr.GT_Flatten(geom.GetGeometryType())
    if flat_type == ogr.wkbPolygon:
        return [geom]
    if flat_type in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        return [p for i in range(geom.GetGeometryCount()) for p in _polygons(geom.GetGeometryRef(i))]
    return []


def prepare(geom):
    """
    Returns a cleaned copy of geom as Polygon or MultiPolygon in EPSG:25832,
    or None if nothing polygonal remains (the caller reports that).

    A geometry without a spatial reference is assumed to already be in
    EPSG:25832 (the pipeline's working CRS); one with a different CRS is
    transformed.
    """
    if geom is None or geom.IsEmpty():
        return None
    geom = geom.Clone()
    geom.FlattenTo2D()

    target = _target_srs()
    source = geom.GetSpatialReference()
    if source is not None and not source.IsSame(target):
        geom.TransformTo(target)
    geom.AssignSpatialReference(target)

    if not geom.IsValid():
        geom = geom.MakeValid()

    polygons = [p for p in _polygons(geom) if not p.IsEmpty() and p.GetArea() > 0]
    if not polygons:
        return None

    if len(polygons) == 1:
        result = _oriented_polygon(polygons[0])
    else:
        result = ogr.Geometry(ogr.wkbMultiPolygon)
        for polygon in polygons:
            result.AddGeometry(_oriented_polygon(polygon))
    result.AssignSpatialReference(target)
    return result


def to_gml(geom, gml_id):
    """
    geom : a geometry returned by prepare().
    gml_id : gml:id for the top-level geometry element; OGR derives ids for
        nested members from it.

    Returns an lxml element (gml:Polygon or gml:MultiSurface).
    """
    gml_text = geom.ExportToGML(options=["FORMAT=GML32", f"GMLID={gml_id}", "SRSNAME_FORMAT=SHORT"])
    # OGR writes the gml: prefix without declaring it -- wrap so lxml can parse it.
    wrapper = etree.fromstring(f'<wrapper xmlns:gml="{GML_NS}">{gml_text}</wrapper>')
    return wrapper[0]


def envelope(geoms):
    """gml:Envelope (lowerCorner/upperCorner) around all geoms, for the
    feature collection's gml:boundedBy."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for geom in geoms:
        x0, x1, y0, y1 = geom.GetEnvelope()
        min_x, max_x = min(min_x, x0), max(max_x, x1)
        min_y, max_y = min(min_y, y0), max(max_y, y1)

    env = etree.Element(f"{{{GML_NS}}}Envelope", srsName=SRS_NAME, nsmap={"gml": GML_NS})
    etree.SubElement(env, f"{{{GML_NS}}}lowerCorner").text = f"{min_x} {min_y}"
    etree.SubElement(env, f"{{{GML_NS}}}upperCorner").text = f"{max_x} {max_y}"
    return env
