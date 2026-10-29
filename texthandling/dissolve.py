"""Final cross-tile dissolve: merge duplicate detections of the same
real-world text that got independently found in 2+ overlapping tiles (see
tiling.py's overlap parameter) into one feature, instead of leaving
near-duplicate features in the output.

The merge criterion is geometric overlap, but NOT at the group/record level
-- it's evaluated between INDIVIDUAL member OCR words (see "word_geoms"
below), one from each candidate record. Group-level overlap alone produced
false positives: group.py can occasionally over-merge within one tile (e.g.
bridging two separate, unrelated rows of a tightly-packed info block
through a shared nearby sub-block -- "SD/KW 10-35" and "SD/KW 35-45"
bridging together through their respective "mit THmax." parts), and that
one jumbled multi-row reading's bbox would then geometrically overlap
ANOTHER tile's correctly-scoped single-row readings even though they share
no actual source content -- clustering those as "duplicates" corrupted the
merge (pairing the wrong SD/KW range with the wrong THmax value). An
earlier attempt fixed this with an exact rounded-centroid identity key
instead, but that was too strict the other way: a truncated OCR read (e.g.
"3.96" missing a misread leading "10") has its centroid shifted noticeably
from the full read's ("103.96"), so genuine duplicates stopped merging.
Individual-word geometric overlap gets both right: Tesseract's own word
segmentation never itself spans multiple distinct real values (only
group.py's merging can do that), so a truncated word's bbox still spatially
overlaps the corresponding portion of the full word's bbox, while two
words on genuinely different lines never overlap at all, regardless of
what larger (possibly over-merged) group bbox either happens to belong to.

Unlike group.py's within-tile clustering (which connects NEARBY but
distinct words using a real-world-meters gap), duplicates here come from
the SAME text being reprojected from two different tile crops. Same
connected-components structure as group.py / processing/utils.py's
eliminate_small_polygons(), different edge criterion.

Merge policy per cluster of duplicates, in order:
  - If every candidate already agrees on (text, label), keep it -- no
    conflict, just corroborating evidence.
  - Otherwise, prefer the candidate whose source tile saw the most complete
    picture (more OCR words merged into it -- see group.py), then the
    longest text, then the highest OCR confidence, as the tie-break chain.
  - The runner(s)-up are NOT discarded silently -- they're recorded in the
    "conflict1"/"conflict2" attributes so a disagreement is visible in the
    output instead of hidden by an arbitrary pick.
  - Geometry is always the UNION of every candidate in the cluster, since
    each is an independent measurement of the same real-world extent.
  - Confidence is the MAX across the cluster -- a duplicate detection is
    itself a form of independent confirmation.
"""

import networkx as nx


def dissolve_duplicates(records):
    """
    records : list of {"geom": ogr.Geometry, "text": str, "label": str,
        "confidence": int, "word_count": int, "word_geoms": [ogr.Geometry, ...]}.
        word_count and word_geoms are internal only (completeness signal and
        duplicate-detection signal, respectively) and are not carried into
        the output. word_geoms holds the individual geo-referenced geometry
        of every original OCR word that contributed to this record -- two
        records are duplicates of the same real-world content only if at
        least one word from each geometrically overlaps.

    Returns a list of {"geom", "text", "label", "confidence", "conflict1",
    "conflict2"} -- one per resolved cluster of duplicates (or per
    non-duplicated record, passed through with conflict1/conflict2 = None).
    """
    n = len(records)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if any(a.Intersects(b) for a in records[i]["word_geoms"] for b in records[j]["word_geoms"]):
                graph.add_edge(i, j)

    return [_merge_cluster([records[i] for i in component])
            for component in nx.connected_components(graph)]


def _merge_cluster(cluster):
    geom = cluster[0]["geom"]
    for r in cluster[1:]:
        geom = geom.Union(r["geom"])
    confidence = max(r["confidence"] for r in cluster)

    readings = {(r["text"], r["label"]) for r in cluster}
    if len(readings) == 1:
        text, label = next(iter(readings))
        return {"geom": geom, "text": text, "label": label, "confidence": confidence,
                "conflict1": None, "conflict2": None, "all_readings": [text]}

    # Disagreement -- rank by completeness, then text length, then confidence,
    # and keep the two runners-up visible instead of silently discarding them.
    ranked = sorted(
        cluster,
        key=lambda r: (r["word_count"], len(r["text"]), r["confidence"]),
        reverse=True,
    )
    winner, losers = ranked[0], ranked[1:]

    # Plain text only -- no "(label)" suffix baked in. That was leaking into
    # reconcile.py's final_text: the model was told it could pick one of the
    # given candidates verbatim, and the candidate strings themselves
    # included the label annotation, so it sometimes echoed the whole thing
    # (see conversation/testing).
    #
    # all_readings keeps EVERY distinct reading in the cluster (not just the
    # top-2 losers stored in conflict1/conflict2) -- a cluster can have 3+
    # overlapping candidates (e.g. seen differently across 3+ overlapping
    # tiles), and reconcile.py needs full visibility to merge complementary
    # fragments (e.g. "THmax." + "Hmax. 6,5m" -> "THmax. 6,5m") instead of
    # silently dropping whichever reading didn't make the top-2 cut (see
    # conversation/testing). all_readings is internal only, not written to
    # the output -- conflict1/conflict2 remain the only persisted attributes.
    all_readings = list(dict.fromkeys(r["text"] for r in ranked))

    return {
        "geom": geom,
        "text": winner["text"],
        "label": winner["label"],
        "confidence": confidence,
        "conflict1": losers[0]["text"] if len(losers) >= 1 else None,
        "conflict2": losers[1]["text"] if len(losers) >= 2 else None,
        "all_readings": all_readings,
    }
