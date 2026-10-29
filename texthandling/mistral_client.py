"""Thin wrapper around the Mistral API's vision-capable chat completion,
used to verify/classify OCR-detected text regions (see ocr.py) -- the VLM
is no longer asked to localize text itself; Tesseract already measured
precise bboxes ahead of time (see conversation/testing: VLM spatial
grounding was measurably noisier than a dedicated CV text detector).

extract_text_features() calls classify_regions() up to twice per raster
tile with no concurrency -- on a large plan this can exceed the API's
requests-per-second limit even while token usage stays well under quota.
MISTRAL_REQUEST_DELAY_MS (set in .env) inserts a fixed delay before every
call in this module, regardless of which of the three functions triggers
it, so every caller is covered without threading a parameter through
extract.py/reconcile.py/structure.py. Defaults to 0 (no delay) -- opt in by
setting it once request-rate errors show up.

A fixed delay only lowers HOW OFTEN the limit gets hit -- it can't
guarantee staying under it (the exact limit isn't published, and it can
still trip on a burst of larger-than-usual tiles). _complete_with_retry()
additionally catches the API's own 429 response and backs off -- honoring
its Retry-After header when present, otherwise doubling from
MISTRAL_REQUEST_DELAY_MS -- and retries, up to MISTRAL_MAX_RETRIES times,
so a rate-limit hit slows the run down instead of crashing it.

THREAD SAFETY: main.py's /text-extraction route is a plain `def`, which
FastAPI/Starlette runs in a threadpool -- two concurrent requests run this
module's code in two different threads. A plain time.sleep() per call only
paces calls made by the SAME thread; it does nothing to stop a second
thread from firing its own call in the middle of that sleep, so concurrent
requests can combine to exceed the real limit even though each looks
individually compliant. _throttle() below coordinates across ALL threads
in this process via a shared lock and a "next allowed call time", so the
configured delay is enforced against the process's TOTAL call rate, not
per-thread. It does NOT coordinate across separate OS processes (e.g.
`uvicorn --workers N>1`, or a second script/test run hitting the same key
at the same time) -- if requests still exceed the limit after this fix,
check for exactly that."""

import os
import threading
import time
import json

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError

from .schema import CLASSIFICATION_SCHEMA, RECONCILE_SCHEMA, STRUCTURE_SCHEMA

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

DEFAULT_MODEL = "mistral-medium-latest"

MISTRAL_REQUEST_DELAY_MS = float(os.environ.get("MISTRAL_REQUEST_DELAY_MS", "0"))
MISTRAL_MAX_RETRIES = int(os.environ.get("MISTRAL_MAX_RETRIES", "5"))
MAX_BACKOFF_SECONDS = 60

_rate_limit_lock = threading.Lock()
_next_allowed_call_time = 0.0


def _throttle():
    """Blocks the calling thread until at least MISTRAL_REQUEST_DELAY_MS
    has passed since the last call from ANY thread in this process (see
    module docstring's Thread Safety note)."""
    global _next_allowed_call_time
    if MISTRAL_REQUEST_DELAY_MS <= 0:
        return
    delay = MISTRAL_REQUEST_DELAY_MS / 1000
    with _rate_limit_lock:
        now = time.monotonic()
        wait = _next_allowed_call_time - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _next_allowed_call_time = now + delay


def _complete_with_retry(client, **kwargs):
    """
    client.chat.complete(**kwargs), throttled (see module docstring) and
    retried with backoff on a 429 rate-limit response. Any other error, or
    a 429 that's still failing after MISTRAL_MAX_RETRIES attempts, is
    re-raised rather than retried forever.
    """
    for attempt in range(MISTRAL_MAX_RETRIES + 1):
        _throttle()
        try:
            return client.chat.complete(**kwargs)
        except MistralError as e:
            if e.status_code != 429 or attempt == MISTRAL_MAX_RETRIES:
                raise
            retry_after = e.headers.get("retry-after")
            if retry_after:
                wait_seconds = float(retry_after)
            else:
                base_seconds = max(MISTRAL_REQUEST_DELAY_MS, 1000) / 1000
                wait_seconds = min(base_seconds * (2 ** attempt), MAX_BACKOFF_SECONDS)
            print(f"mistral_client: rate limited (attempt {attempt + 1}/{MISTRAL_MAX_RETRIES}), "
                  f"waiting {wait_seconds:.1f}s before retrying")
            time.sleep(wait_seconds)


def build_classification_prompt(words):
    """
    words : the OCR-detected words for this tile (see ocr.detect_words), in
        the same order annotate.draw_numbered_boxes() numbered them.
    """
    guesses = "\n".join(f'  {i}: OCR guessed "{w["text"]}"' for i, w in enumerate(words))
    return (
        "This image has small red numbered boxes drawn around pieces of "
        "text that an OCR pass detected. For EACH numbered box listed "
        "below, look at that exact spot in the image and report:\n"
        "- is_text: true if it's genuinely legible text, false if the OCR "
        "engine misread something else (a line, hatch mark, drawing "
        "symbol) as text\n"
        "- corrected_text: the actual correct text at that spot (fix any "
        "OCR misreads); empty string if is_text is false\n"
        "- label: what kind of label it is, if identifiable (e.g. "
        "street_name, house_number, parcel_number, elevation_value, "
        "zoning_code)\n\n"
        "OCR's guesses for each numbered box (verify these against the "
        "image -- don't just repeat them if they look wrong):\n"
        f"{guesses}\n\n"
        "Respond only with the requested JSON."
        #"Respond with exactly one classification per numbered box, in "
        #"order. 
    )


def get_client(api_key=None):
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set (pass api_key= or set the env var)")
    return Mistral(api_key=api_key)


def classify_regions(client, image_b64, words, model=DEFAULT_MODEL, prompt=None):
    """
    Send one base64-encoded, numbered-box-annotated PNG tile (see
    annotate.draw_numbered_boxes) to the VLM, along with OCR's guess for
    each numbered box. Returns a list of classifications:
        [{"index": int, "is_text": bool, "corrected_text": str, "label": str}, ...]

    Geometry is NOT requested here -- callers pair each classification back
    up with words[classification["index"]]["bbox"] (Tesseract's measured
    bbox) before reprojecting via geocode.bbox_to_geom().
    """
    response = _complete_with_retry(
        client,
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or build_classification_prompt(words)},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
            ],
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "classifications", "schema": CLASSIFICATION_SCHEMA, "strict": True},
        },
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("classifications", [])


def build_reconcile_prompt(context, conflicts):
    """
    context : dissolve.dissolve_duplicates() entries that had NO conflict
        (read-only background -- what else this plan already confirms).
    conflicts : entries that DID have a conflict. Uses "all_readings" (every
        distinct reading in the cluster) rather than just conflict1/conflict2
        -- a cluster can have 3+ overlapping candidates, and this needs full
        visibility to merge complementary fragments instead of only seeing
        the top-2 by dissolve.py's local completeness ranking.
    """
    context_lines = "\n".join(
        f'  - "{r["text"]}"' + (f', labeled {r["label"]}' if r.get("label") else "")
        for r in context
    ) or "  (none)"

    conflict_lines = []
    for i, r in enumerate(conflicts):
        readings = r.get("all_readings") or [r["text"]]
        readings_str = ", ".join(f'"{t}"' for t in readings)
        label = f', labeled {r["label"]}' if r.get("label") else ""
        conflict_lines.append(f'  {i}: readings found{label}: {readings_str}')

    return (
        "You are reviewing text extracted from a German Bebauungsplan "
        "(zoning plan). Some detections were found more than once (the "
        "plan was processed in overlapping tiles, and nearby detections "
        "from the same zone-specification block can end up geometrically "
        "overlapping too) and disagreed on their exact reading. For each "
        "numbered conflict below, the readings found fall into one of "
        "three cases:\n"
        "  (a) different-quality attempts at reading the EXACT SAME "
        "content -- pick the most accurate one, or correct an obvious OCR "
        "error (e.g. a truncated or misread digit).\n"
        "  (b) COMPLEMENTARY FRAGMENTS that each captured a different, "
        "non-overlapping part of the SAME single value -- e.g. one "
        "reading is missing a leading word/letter (\"Hmax. 6,5m\", where "
        "a leading \"T\" was misread or dropped) while another has the "
        "correct prefix but is missing the value (\"THmax.\"). Combine "
        "these into the single most complete correct reading of that one "
        "value.\n"
        "  (c) GENUINELY DIFFERENT, DISTINCT values that both belong to "
        "the same zone-specification block -- e.g. \"THmax. 4,2 m\" "
        "(max. eave height) and \"FHmax. 9,0 m\" (max. ridge height) are "
        "two separate real parameters, not two readings of one value. "
        "Never discard one of these just because only one can be picked "
        "as a \"winner\" -- combine ALL distinct values into final_text, "
        "each kept intact and separated with \"; \".\n"
        "Use the surrounding confirmed content of this plan and your "
        "general knowledge of German zoning-plan conventions (GRZ/GFZ "
        "ratios, elevation reference points, WA/MI/etc. zoning codes, "
        "plausible building heights, street/parcel numbering...) to judge "
        "which case applies. When genuinely unsure whether two readings "
        "are the same value (case a/b) or different values (case c), "
        "prefer case (c) -- keeping an extra distinct value that turns "
        "out to be a duplicate is far less costly than silently losing "
        "real content. Don't guess wildly if there's no real basis at "
        "all -- in that case just keep every reading, combined.\n\n"
        "IMPORTANT: final_text must be the plain text ONLY -- do not "
        "include the label or the word \"labeled\" in it, even though the "
        "readings below are shown with their label for context.\n\n"
        f"Confirmed context from elsewhere on this plan:\n{context_lines}\n\n"
        "Conflicts to resolve:\n" + "\n".join(conflict_lines) + "\n\n"
        "Respond with one resolution per numbered conflict, in order. "
        "Respond only with the requested JSON."
    )


def reconcile_conflicts(client, context, conflicts, model=DEFAULT_MODEL, prompt=None):
    """
    Text-only call (no image -- this is a reasoning task over already-
    extracted text, not a re-detection task) resolving dissolve.py's
    unresolved conflicts. Returns a list of
        [{"index": int, "final_text": str, "final_label": str}, ...]
    """
    response = _complete_with_retry(
        client,
        model=model,
        messages=[{
            "role": "user",
            "content": prompt or build_reconcile_prompt(context, conflicts),
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "resolutions", "schema": RECONCILE_SCHEMA, "strict": True},
        },
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("resolutions", [])


def build_structure_prompt(blocks):
    """
    blocks : blocks.merge_blocks() output -- each block's "lines" is the
        ordered list of its individual, already-confirmed line readings.
    """
    block_lines = []
    for i, b in enumerate(blocks):
        numbered = "\n".join(f'      - "{line}"' for line in b["lines"])
        block_lines.append(f'  {i}:\n{numbered}')

    return (
        "You are extracting structured attributes from confirmed text "
        "found on a German Bebauungsplan (zoning plan). Below are "
        "numbered blocks, each a list of individual lines that were found "
        "close together on the plan (e.g. one zone's full specification). "
        "For each block, identify which of its lines have a clear "
        "key/value structure -- e.g. \"GRZ 0,4\" is key \"GRZ\", value "
        "\"0,4\"; \"WA2\" alone (a zoning classification code) could be "
        "key \"classification\", value \"WA2\"; \"THmax. 6,5 m\" is key "
        "\"THmax\", value \"6,5 m\". Use short, CONSISTENT key names -- "
        "the same kind of attribute must use the exact same key every "
        "time it appears across different blocks, so results line up. "
        "Use your knowledge of German zoning-plan conventions (GRZ/GFZ "
        "ratios, THmax/FHmax height limits, SD/KW roof pitch, WA/MI/etc. "
        "zoning classification codes, Vollgeschosszahl, Wohnungszahl...) "
        "to recognize these even when abbreviated or partially unclear.\n\n"
        "Not every line has this structure -- a bare elevation value, a "
        "street name, or an unlabeled fragment does not. Simply omit "
        "those from that block's fields; don't force a key onto them.\n\n"
        "Blocks:\n" + "\n".join(block_lines) + "\n\n"
        "Respond with one entry per numbered block, in order (fields may "
        "be an empty list). Respond only with the requested JSON."
    )


def structure_blocks(client, blocks, model=DEFAULT_MODEL, prompt=None):
    """
    Text-only call (no image -- reasoning over already-confirmed text, not
    a re-detection task) parsing each block's lines into key/value
    attribute pairs. Returns a list of
        [{"index": int, "fields": [{"key": str, "value": str}, ...]}, ...]
    """
    response = _complete_with_retry(
        client,
        model=model,
        messages=[{
            "role": "user",
            "content": prompt or build_structure_prompt(blocks),
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "blocks", "schema": STRUCTURE_SCHEMA, "strict": True},
        },
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("blocks", [])
