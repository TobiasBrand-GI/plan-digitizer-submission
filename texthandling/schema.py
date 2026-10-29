"""
Structured-output schema for the VLM's classification pass (see
mistral_client.classify_regions), and the subset of it that becomes GeoJSON
feature attributes.

Geometry is NOT part of this schema -- Tesseract (ocr.py) already measured
precise per-word pixel bboxes ahead of time; the VLM's job is purely to
verify/correct each detected word's text and add semantic labels, referring
to each by the numbered box annotate.py drew for it (see conversation --
VLM-reported bboxes were measurably noisier than a dedicated CV detector's).

Each classification the model reports is expected to look like:
    {
        "index": 3,                  # matches the numbered box drawn on the image
        "is_text": true,             # false if OCR misread a drawing line/hatch mark
        "corrected_text": "...",     # OCR's text, corrected if it misread something
        "label": "...",              # e.g. "street_name", "house_number" -- optional
    }
"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The numbered box index from the image, matching "
                                       "the OCR guess list in the prompt",
                    },
                    "is_text": {
                        "type": "boolean",
                        "description": "False if this numbered box is not actually "
                                       "legible text (e.g. OCR misread a drawing line, "
                                       "hatch mark, or symbol as text)",
                    },
                    "corrected_text": {
                        "type": "string",
                        "description": "The actual correct text at that spot, fixing any "
                                       "OCR misreads. Empty string if is_text is false.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Semantic category of the text, if identifiable "
                                       "(e.g. street_name, house_number, parcel_number, "
                                       "elevation_value, zoning_code)",
                    },
                },
                "required": ["index", "is_text", "corrected_text"],
            },
        },
    },
    "required": ["classifications"],
}

"""
Schema for the final reconciliation pass (see reconcile.py /
mistral_client.reconcile_conflicts): a text-only call over dissolve.py's
unresolved conflicts, using the rest of the extraction's confirmed results
as context. Each resolution the model reports is expected to look like:
    {
        "index": 1,                  # matches the numbered conflict in the prompt
        "final_text": "...",         # the chosen/corrected final text
        "final_label": "...",        # optional -- only if it should change
    }
"""

RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The numbered conflict index from the prompt",
                    },
                    "final_text": {
                        "type": "string",
                        "description": "The single best final reading -- one of the "
                                       "given candidates, or a corrected reading if the "
                                       "candidates suggest an OCR error. Keep the current "
                                       "pick if there's no real basis to change it. Plain "
                                       "text ONLY -- never include a label annotation here.",
                    },
                    "final_label": {
                        "type": "string",
                        "description": "Corrected semantic label, only if it should "
                                       "change from the current pick's label",
                    },
                },
                "required": ["index", "final_text"],
            },
        },
    },
    "required": ["resolutions"],
}

"""
Schema for the attribute-structuring pass (see structure.py /
mistral_client.structure_blocks): a text-only call over blocks.py's merged
blocks, parsing each block's individual lines into key/value attribute
pairs using the model's zoning-plan domain knowledge (e.g. "GRZ 0,4" ->
key="GRZ", value="0,4"). Not every line has this structure (e.g. a bare
elevation value or an unlabeled fragment) -- those are simply omitted from
"fields" for that block. Each block's result is expected to look like:
    {
        "index": 0,                              # matches the numbered block in the prompt
        "fields": [{"key": "GRZ", "value": "0,4"}, {"key": "GFZ", "value": "0,7"}],
    }
"""

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The numbered block index from the prompt",
                    },
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Short, consistent attribute name (e.g. "
                                                   "GRZ, GFZ, THmax, FHmax, classification) "
                                                   "-- use the SAME key every time the same "
                                                   "kind of attribute appears, so results "
                                                   "line up across blocks.",
                                },
                                "value": {"type": "string"},
                            },
                            "required": ["key", "value"],
                        },
                        "description": "One entry per line in this block that has a clear "
                                       "key/value structure. Lines with no such structure "
                                       "(e.g. a bare elevation value, an unlabeled "
                                       "fragment) are simply omitted here.",
                    },
                },
                "required": ["index", "fields"],
            },
        },
    },
    "required": ["blocks"],
}

# Which per-region keys end up as GeoJSON feature attributes, in the order
# the output layer's fields are created. "confidence" here is Tesseract's
# OCR confidence (a real measured value), not VLM self-reporting.
# "conflict1"/"conflict2" are populated by dissolve.py when duplicate
# detections of the same real-world text (found in 2+ overlapping tiles --
# see tiling.py) disagreed on text/label; empty otherwise.
TEXT_FIELDS = ["text", "label", "confidence", "conflict1", "conflict2"]
