"""
XPlanung 6.1 export: turns a pipeline run directory into an XPlanGML
document plus a report of everything that couldn't be carried over.

    export_xplan(run_dir, umring_path, plan_name, ags)
        reads   run_dir/linked_areas.geojson, run_dir/shp/polygon3.shp,
                run_dir/unmatched_text.geojson (if present), umring_path
        writes  run_dir/plan.gml, run_dir/export_report.json

File-path-in, file-path-out like the other pipeline stages. A document that
fails XSD validation is still written -- the report says so and lists the
errors -- so it can be inspected instead of disappearing.
"""

import dataclasses
import json
import os

from osgeo import ogr

from . import geometry, mapping, model, serialize, validate


def _prepare_all(objects, source, issues):
    """Runs geometry.prepare on every object's position; objects whose
    geometry is empty afterwards are dropped and reported."""
    kept = []
    for index, obj in enumerate(objects):
        prepared = geometry.prepare(obj.position)
        if prepared is None:
            issues.append(mapping.Issue(source, index, None, None,
                                        "no polygonal geometry left after repair; object dropped"))
            continue
        obj.position = prepared
        kept.append(obj)
    return kept


def _unmatched_texts(path):
    """Text blocks the linking step couldn't attach to any area -- listed in
    the report so they can be placed by hand."""
    if not os.path.exists(path):
        return []
    ds = ogr.Open(path)
    texts = [feature.GetField("text") for feature in ds.GetLayer(0)
             if feature.GetFieldIndex("text") >= 0 and feature.GetField("text")]
    ds = None
    return texts


def export_xplan(run_dir, umring_path, plan_name, ags, gemeinde_name=None):
    """
    run_dir : directory of a /process-plan run (holds linked_areas.geojson
        and shp/polygon3.shp).
    umring_path : the plan's boundary (Geltungsbereich) file; any CRS --
        it is transformed to EPSG:25832.
    plan_name : BP_Plan name.
    ags : Amtlicher Gemeindeschlüssel of the municipality.

    Returns (gml_path, report_path).
    """
    issues = []

    areas, area_issues = mapping.read_areas(os.path.join(run_dir, "linked_areas.geojson"))
    issues.extend(area_issues)
    areas = _prepare_all(areas, "linked_areas.geojson", issues)

    windows = mapping.read_baufenster(os.path.join(run_dir, "shp", "polygon3.shp"))
    windows = _prepare_all(windows, "shp/polygon3.shp", issues)

    geltungsbereich = geometry.prepare(mapping.read_geltungsbereich(umring_path))
    if geltungsbereich is None:
        raise ValueError(f"no polygonal Geltungsbereich in {umring_path}")

    auszug = model.XPlanAuszug(
        plan=model.Plan(
            name=plan_name,
            raeumlicherGeltungsbereich=geltungsbereich,
            gemeinde=[model.Gemeinde(ags=ags, gemeindeName=gemeinde_name)],
        ),
        bereich=model.Bereich(geltungsbereich=geltungsbereich),
        baugebiete=areas,
        baufenster=windows,
    )

    gml_path = serialize.write(auszug, os.path.join(run_dir, "plan.gml"))
    xsd_errors = validate.validate(gml_path)

    report = {
        "gml": gml_path,
        "xsd_valid": not xsd_errors,
        "xsd_errors": xsd_errors,
        "counts": {"baugebiete": len(areas), "baufenster": len(windows), "issues": len(issues)},
        "issues": [dataclasses.asdict(issue) for issue in issues],
        "unmatched_text": _unmatched_texts(os.path.join(run_dir, "unmatched_text.geojson")),
    }
    report_path = os.path.join(run_dir, "export_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return gml_path, report_path
