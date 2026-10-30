import os

# Two PROJ installations share this process: rasterio's wheel ships its own,
# newer proj.db, while the system GDAL (osgeo) needs /usr/share/proj. A global
# PROJ_DATA would force rasterio onto the old database -- so it is removed for
# rasterio and handed to osgeo directly instead.
_SYSTEM_PROJ_DATA = os.environ.pop("PROJ_DATA", "/usr/share/proj")

from osgeo import osr
osr.SetPROJSearchPaths([_SYSTEM_PROJ_DATA])

from datetime import datetime

import json
import multiprocessing
import re
import shutil
import tempfile
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask
import rasterio
from osgeo import gdal, ogr
from processing import masking, pipeline, polygonizing, utils
from texthandling import extract_text_features
from linking import link_text_to_areas
from export import export_xplan
import numpy as np
import cv2

gdal.UseExceptions()
app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    """Demo route: masks a hardcoded sample GeoTIFF's red pixels, writes
    rot_maske.tif, and returns the dataset width alongside item_id/q."""
    dataset = rasterio.open('483229_88_roevekamp_bplan-satzung_gezeichnet20240923_modifiziert.tif')

    red_mask, masked_output, meta = masking.red(dataset)
    # Maske wegschreiben
    with rasterio.open("rot_maske.tif", "w", **meta) as dst:
        dst.write(red_mask, 1)

    return {"item_id": item_id, "q": q, "dataset": dataset.width}

@app.get("/polygons")
def get_areas(tif: str, umring: str, run_dir: str):
    """run_dir : directory to write areas.geojson (and intermediates)
    into -- e.g. one already created by /process-plan. This route never
    creates its own directory; the caller (a human via /docs, or
    /process-plan) always supplies run_dir."""
    dirname = os.path.dirname(__file__)

    areas_path = pipeline.segment_areas(
        os.path.join(dirname, tif),
        os.path.join(dirname, umring),
        os.path.join(dirname, "vectors/nutzungFlurstueck_kgt.shp"),
        out_dir=run_dir,
    )

    return {"areas": areas_path}


@app.get("/text-extraction")
def extract_text(tif: str, run_dir: str):
    """run_dir : directory to write text_features.geojson into -- see
    /polygons' run_dir note."""
    dirname = os.path.dirname(__file__)

    print("extracting text from geotiff")
    out_path = extract_text_features(
        os.path.join(dirname, tif),
        os.path.join(run_dir, "text_features.geojson"),
    )

    return {"output": out_path}


@app.get("/link-text-to-areas")
def link_text_to_areas_route(tif: str, run_dir: str):
    """run_dir : directory holding areas.geojson and text_features.geojson
    (from prior /polygons and /text-extraction calls against the same
    run_dir) -- outputs are written there too."""
    dirname = os.path.dirname(__file__)

    print("linking text blocks to area polygons")
    out_path, unmatched_out_path, detected_lines_out_path = link_text_to_areas(
        os.path.join(run_dir, "areas.geojson"),
        os.path.join(run_dir, "text_features.geojson"),
        os.path.join(dirname, tif),
        os.path.join(run_dir, "linked_areas.geojson"),
        os.path.join(run_dir, "unmatched_text.geojson"),
        os.path.join(run_dir, "detected_lines.geojson"),
    )

    return {"output": out_path, "unmatched": unmatched_out_path, "detected_lines": detected_lines_out_path}


@app.get("/export-xplan")
def export_xplan_route(run_dir: str, umring: str, plan_name: str,
                       ags: str | None = None, gemeinde_name: str | None = None):
    """run_dir : directory holding linked_areas.geojson and shp/polygon3.shp
    (from prior /polygons, /text-extraction and /link-text-to-areas calls
    against the same run_dir) -- plan.gml and export_report.json are
    written there too."""
    dirname = os.path.dirname(__file__)

    print("exporting XPlanGML 6.1")
    gml_path, report_path = export_xplan(
        run_dir, os.path.join(dirname, umring), plan_name, ags, gemeinde_name)

    return {"gml": gml_path, "report": report_path}


@app.get("/process-plan")
def process_plan(tif: str, umring: str, ags: str | None = None, gemeinde_name: str | None = None):
    """
    Runs the full pipeline end to end by calling /polygons, /text-extraction,
    /link-text-to-areas and /export-xplan in sequence against one shared
    run_dir. Every output is written into a fresh directory under runs/,
    named <timestamp>_<plan name>, so consecutive calls never overwrite
    each other's files.
    """
    plan_name = os.path.splitext(os.path.basename(tif))[0]
    run_dir = _new_run_dir(plan_name)
    return {"run_dir": run_dir, **_run_pipeline(tif, umring, run_dir, plan_name, ags, gemeinde_name)}


RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# The four stages of a full run, in order -- step numbers in status.json
# (see the /jobs routes) are 1-based indices into this list.
PIPELINE_STEPS = ["Segmenting areas", "Extracting text", "Linking text to areas", "Exporting XPlanGML"]


def _new_run_dir(plan_name):
    """Creates and returns a fresh timestamped run directory under
    RUNS_DIR for plan_name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"{timestamp}_{plan_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _run_pipeline(tif, umring, run_dir, plan_name, ags=None, gemeinde_name=None, on_step=None):
    """The four routes in sequence against one run_dir. on_step(n) is called
    before step n (1-based, see PIPELINE_STEPS) starts."""
    def step(n):
        """Logs and runs on_step(n) for pipeline step n (1-based) before it starts."""
        print(f"[{plan_name}] {PIPELINE_STEPS[n - 1].lower()} -> {run_dir}")
        if on_step is not None:
            on_step(n)

    step(1)
    areas = get_areas(tif, umring, run_dir)
    step(2)
    text_extraction = extract_text(tif, run_dir)
    step(3)
    linked = link_text_to_areas_route(tif, run_dir)
    step(4)
    exported = export_xplan_route(run_dir, umring, plan_name, ags, gemeinde_name)

    return {
        "areas": areas["areas"],
        "text_features": text_extraction["output"],
        "linked_areas": linked["output"],
        "unmatched_text": linked["unmatched"],
        "detected_lines": linked["detected_lines"],
        "xplan_gml": exported["gml"],
        "export_report": exported["report"],
    }


# --- Web UI jobs -------------------------------------------------------------

_JOB_ID = re.compile(r"^[\w.-]+$")
_TIF_SUFFIXES = (".tif", ".tiff")
_UMRING_SUFFIXES = (".geojson", ".json")
_jobs = {}  # job id -> multiprocessing.Process, for jobs started by this server process


def _write_status(run_dir, **status):
    """Atomically writes run_dir/status.json with the given status fields
    plus the pipeline step list."""
    path = os.path.join(run_dir, "status.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"total_steps": len(PIPELINE_STEPS), "steps": PIPELINE_STEPS, **status}, f)
    os.replace(path + ".tmp", path)  # atomic -- a poll never reads a half-written file


def _run_job(run_dir, tif, umring, plan_name, ags, gemeinde_name):
    """Entry point of the job process."""
    try:
        _run_pipeline(tif, umring, run_dir, plan_name, ags, gemeinde_name,
                      on_step=lambda n: _write_status(run_dir, state="running", step=n))
        _write_status(run_dir, state="done", step=len(PIPELINE_STEPS))
    except Exception as e:
        traceback.print_exc()
        _write_status(run_dir, state="failed", error=f"{type(e).__name__}: {e}")


def _job_dir(job_id):
    """Returns the run directory for job_id; 404s if job_id is malformed
    or the directory doesn't exist."""
    run_dir = os.path.join(RUNS_DIR, job_id)
    if not _JOB_ID.match(job_id) or not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="Unknown job")
    return run_dir


def _upload_name(upload, suffixes):
    """The upload's bare file name; 400 if it doesn't have one of `suffixes`."""
    name = os.path.basename(upload.filename or "")
    if not name.lower().endswith(suffixes):
        raise HTTPException(status_code=400, detail=f"{name or 'File'}: expected {' or '.join(suffixes)}")
    return name


def _save_upload(upload, target_dir, name):
    """Saves upload's contents to target_dir/name and returns the path."""
    path = os.path.join(target_dir, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@app.post("/jobs")
def create_job(tif: UploadFile = File(...), umring: UploadFile = File(...),
               ags: str = Form(...), gemeinde_name: str = Form(...)):
    """Starts a full pipeline run on uploaded files. Returns the job id
    (= the run directory's name under runs/)."""
    ags, gemeinde_name = ags.strip(), gemeinde_name.strip()
    if not ags or not gemeinde_name:
        raise HTTPException(status_code=400, detail="AGS and municipality name are required")
    if any(process.is_alive() for process in _jobs.values()):
        raise HTTPException(status_code=409, detail="Another job is still running")

    # validate both before creating anything, so a rejected upload leaves no run dir behind
    tif_name = _upload_name(tif, _TIF_SUFFIXES)
    umring_name = _upload_name(umring, _UMRING_SUFFIXES)

    plan_name = os.path.splitext(tif_name)[0]
    run_dir = _new_run_dir(plan_name)
    input_dir = os.path.join(run_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    tif_path = _save_upload(tif, input_dir, tif_name)
    umring_path = _save_upload(umring, input_dir, umring_name)

    job_id = os.path.basename(run_dir)
    _write_status(run_dir, state="running", step=0)
    # spawn, not fork: a fork of the server would inherit its threads and
    # GDAL state; a spawned process starts clean and re-runs main.py's PROJ setup
    process = multiprocessing.get_context("spawn").Process(
        target=_run_job, args=(run_dir, tif_path, umring_path, plan_name, ags, gemeinde_name),
        daemon=True)
    process.start()
    _jobs[job_id] = process
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Returns job_id's status.json contents, marking it failed if its
    process died without writing one or isn't tracked (e.g. after a
    server restart)."""
    run_dir = _job_dir(job_id)
    try:
        with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as f:
            status = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not a web UI job")

    if status["state"] == "running":
        process = _jobs.get(job_id)
        if process is None:
            status = {**status, "state": "failed", "error": "Interrupted (server restarted)"}
        elif not process.is_alive():
            # died without writing its own status -- e.g. a native crash
            status = {**status, "state": "failed", "error": f"Job process crashed (exit code {process.exitcode})"}
    return {"job_id": job_id, **status}


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str):
    """The whole run directory as a ZIP."""
    run_dir = _job_dir(job_id)
    tmp_dir = tempfile.mkdtemp()
    archive = shutil.make_archive(os.path.join(tmp_dir, job_id), "zip", run_dir)
    return FileResponse(archive, filename=f"{job_id}.zip", media_type="application/zip",
                        background=BackgroundTask(shutil.rmtree, tmp_dir))


@app.get("/debug")
def get_dots_debug():
    """Debug route: runs masking.debug_bboxes on a hardcoded sample raster,
    writes an annotated PNG, and prints per-contour stats sorted by
    circularity."""
    dataset = utils.open_rgb('./tif/z_207_08.tif', './debug')
    height = dataset.height
    width = dataset.width
    meta = dataset.meta.copy()

    debug_img, stats = masking.debug_bboxes(dataset, height, width)

    # Visually inspect (e.g. in a notebook or save to disk)
    cv2.imwrite("debug_bboxes1.png", debug_img)

    # Or inspect the raw numbers directly, sorted to see the split between dots/dashes
    for s in sorted(stats, key=lambda s: s["circularity"], reverse=True):
        print(s)

        #utils.export_lines_to_vector(
        #    lines,
        #    transform=dataset.transform,
        #    crs_wkt=dataset.crs.to_wkt(),  # rasterio CRS → WKT for GDAL/OGR
        #)

    return {"Debug"}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """Demo route: echoes back item's name and item_id."""
    return {"item_name": item.name, "item_id": item_id}


# Web UI -- mounted last so it never shadows an API route
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
                             html=True), name="ui")