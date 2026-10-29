"""
QGIS Processing script: compare polygons of two layers matched by ID.

For every feature in BOTH layers it writes:
  match_found  1 if a partner with the same ID exists in the other layer, else 0
  area         area of this feature (layer CRS units, e.g. m²)
  area_partner area of the partner polygon
  inter_area   overlap area
  union_area   area of the union
  iou          intersection over union (0 if no partner)
  area_diff_pct  (generated - reference) / reference * 100
                 -> same sign convention in both outputs: positive = generated larger
  pp           Polsby-Popper compactness of this feature (4*pi*A / P^2)
  pp_partner   Polsby-Popper of the partner
  pp_diff      pp(generated) - pp(reference)
  perimeter    perimeter of this feature
  perimeter_partner  perimeter of the partner polygon
  x, y         centroid coordinates of this feature (in the reference layer's CRS)

If an ID occurs several times in one layer, those features are combined into
one geometry for the comparison (and a warning is shown).

Install: Processing Toolbox -> Python icon -> "Add Script to Toolbox..."
(or "Open Existing Script..." and run it from the editor).
"""

import math

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
)


class PolygonIoUById(QgsProcessingAlgorithm):
    REF = "REFERENCE"
    REF_ID = "REFERENCE_ID"
    GEN = "GENERATED"
    GEN_ID = "GENERATED_ID"
    OUT_REF = "OUTPUT_REFERENCE"
    OUT_GEN = "OUTPUT_GENERATED"

    NEW_FIELDS = [
        ("match_found", QVariant.Int),
        ("area", QVariant.Double),
        ("area_partner", QVariant.Double),
        ("inter_area", QVariant.Double),
        ("union_area", QVariant.Double),
        ("iou", QVariant.Double),
        ("area_diff_pct", QVariant.Double),
        ("pp", QVariant.Double),
        ("pp_partner", QVariant.Double),
        ("pp_diff", QVariant.Double),
        ("perimeter", QVariant.Double),
        ("perimeter_partner", QVariant.Double),
        ("x", QVariant.Double),
        ("y", QVariant.Double),
    ]
    N_CORE = 10  # number of metric fields before perimeter/x/y

    # ---- metadata ----------------------------------------------------------
    def tr(self, s):
        return QCoreApplication.translate("PolygonIoUById", s)

    def createInstance(self):
        return PolygonIoUById()

    def name(self):
        return "polygon_iou_by_id"

    def displayName(self):
        return self.tr("Polygon IoU and shape comparison (by ID)")

    def group(self):
        return self.tr("Thesis tools")

    def groupId(self):
        return "thesis_tools"

    def shortHelpString(self):
        return self.tr(
            "Matches polygons of a reference (ground truth) layer and a generated "
            "layer by ID field and computes IoU, relative area difference and "
            "Polsby-Popper compactness difference for every feature of both layers. "
            "Use a projected CRS (metres); the generated layer is reprojected to "
            "the reference layer's CRS."
        )

    # ---- parameters --------------------------------------------------------
    def initAlgorithm(self, config=None):
        poly = [QgsProcessing.TypeVectorPolygon]
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.REF, self.tr("Reference layer (ground truth)"), poly))
        self.addParameter(QgsProcessingParameterField(
            self.REF_ID, self.tr("Reference ID field"), parentLayerParameterName=self.REF))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.GEN, self.tr("Generated layer"), poly))
        self.addParameter(QgsProcessingParameterField(
            self.GEN_ID, self.tr("Generated ID field"), parentLayerParameterName=self.GEN))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_REF, self.tr("Reference with metrics"), QgsProcessing.TypeVectorPolygon))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUT_GEN, self.tr("Generated with metrics"), QgsProcessing.TypeVectorPolygon))

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _clean(geom):
        if geom is None or geom.isEmpty():
            return None
        if not geom.isGeosValid():
            geom = geom.makeValid()
        return geom

    @staticmethod
    def _pp(geom):
        p = geom.length()
        return 4 * math.pi * geom.area() / p ** 2 if p > 0 else None

    def _read(self, source, id_field, transform, feedback, label):
        """Return list of (feature, key, geometry) and dict key -> combined geometry."""
        items, groups = [], {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            g = QgsGeometry(f.geometry())
            if transform is not None:
                g.transform(transform)
            g = self._clean(g)
            key = f[id_field]
            key = None if key is None else str(key)
            items.append((f, key, g))
            if key is not None and g is not None:
                groups.setdefault(key, []).append(g)

        dupes = [k for k, gs in groups.items() if len(gs) > 1]
        if dupes:
            feedback.pushWarning(
                f"{label}: {len(dupes)} IDs occur more than once and were combined "
                f"(e.g. {', '.join(dupes[:5])})")
        combined = {k: (gs[0] if len(gs) == 1 else self._clean(QgsGeometry.unaryUnion(gs)))
                    for k, gs in groups.items()}
        return items, combined

    def _write(self, items, partners, sink, fields, is_reference, feedback, offset, total):
        unmatched = 0
        for i, (f, key, g) in enumerate(items):
            if feedback.isCanceled():
                break
            vals = [0] + [None] * (self.N_CORE - 1)
            extra = [None, None, None, None]  # perimeter, perimeter_partner, x, y
            partner = partners.get(key) if key is not None else None

            if g is not None:
                a = g.area()
                pp = self._pp(g)
                vals[1], vals[7] = a, pp
                c = g.centroid().asPoint()
                extra = [g.length(), None, c.x(), c.y()]
                if partner is not None:
                    extra[1] = partner.length()
                    a_p = partner.area()
                    pp_p = self._pp(partner)
                    inter = g.intersection(partner)
                    inter_a = inter.area() if inter and not inter.isEmpty() else 0.0
                    union_a = a + a_p - inter_a
                    a_ref, a_gen = (a, a_p) if is_reference else (a_p, a)
                    pp_ref, pp_gen = (pp, pp_p) if is_reference else (pp_p, pp)
                    vals = [
                        1, a, a_p, inter_a, union_a,
                        inter_a / union_a if union_a > 0 else None,
                        (a_gen - a_ref) / a_ref * 100 if a_ref > 0 else None,
                        pp, pp_p,
                        pp_gen - pp_ref if None not in (pp_gen, pp_ref) else None,
                    ]
                else:
                    vals[5] = 0.0  # no partner -> IoU 0
            if vals[0] == 0:
                unmatched += 1

            out = QgsFeature(fields)
            out.setGeometry(g if g is not None else f.geometry())
            out.setAttributes(f.attributes() + vals + extra)
            sink.addFeature(out, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(100 * (offset + i + 1) / max(total, 1)))
        return unmatched

    # ---- main --------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        ref = self.parameterAsSource(parameters, self.REF, context)
        gen = self.parameterAsSource(parameters, self.GEN, context)
        if ref is None or gen is None:
            raise QgsProcessingException(self.tr("Could not load input layers."))
        ref_id = self.parameterAsString(parameters, self.REF_ID, context)
        gen_id = self.parameterAsString(parameters, self.GEN_ID, context)

        crs = ref.sourceCrs()
        if crs.isGeographic():
            raise QgsProcessingException(self.tr(
                "Reference layer uses a geographic CRS (degrees). "
                "Reproject both layers to a projected CRS in metres first."))
        transform = None
        if gen.sourceCrs() != crs:
            transform = QgsCoordinateTransform(gen.sourceCrs(), crs, context.transformContext())
            feedback.pushInfo("Generated layer is reprojected to the reference CRS.")

        def make_sink(param, source):
            fields = QgsFields(source.fields())
            for name, ftype in self.NEW_FIELDS:
                if fields.indexOf(name) != -1:
                    raise QgsProcessingException(
                        f"Input layer already has a field named '{name}'.")
                fields.append(QgsField(name, ftype))
            sink, dest = self.parameterAsSink(
                parameters, param, context, fields, source.wkbType(), crs)
            if sink is None:
                raise QgsProcessingException(self.tr("Could not create output."))
            return sink, dest, fields

        sink_ref, dest_ref, f_ref = make_sink(self.OUT_REF, ref)
        sink_gen, dest_gen, f_gen = make_sink(self.OUT_GEN, gen)

        feedback.pushInfo("Reading layers ...")
        ref_items, ref_geoms = self._read(ref, ref_id, None, feedback, "Reference")
        gen_items, gen_geoms = self._read(gen, gen_id, transform, feedback, "Generated")
        total = len(ref_items) + len(gen_items)

        feedback.pushInfo("Computing metrics for reference features ...")
        un_ref = self._write(ref_items, gen_geoms, sink_ref, f_ref, True, feedback, 0, total)
        feedback.pushInfo("Computing metrics for generated features ...")
        un_gen = self._write(gen_items, ref_geoms, sink_gen, f_gen, False, feedback,
                             len(ref_items), total)

        feedback.pushInfo(f"Reference features without partner: {un_ref}")
        feedback.pushInfo(f"Generated features without partner: {un_gen}")
        return {self.OUT_REF: dest_ref, self.OUT_GEN: dest_gen}