import networkx as nx


def dissolve_duplicates(records):
    """
    records : list of {"geom", "text", "label", "confidence", "word_count",
    "word_geoms"} dicts; two records are duplicates only if a word_geoms
    entry from each geometrically overlaps. Returns a list of {"geom",
    "text", "label", "confidence", "conflict1", "conflict2"} -- one per
    resolved cluster (or per non-duplicate, with conflict1/conflict2=None).
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
    """Merges one cluster of duplicate records into one: geometry is the
    union, confidence the max; if readings disagree, keeps the most
    complete/longest/highest-confidence text as winner and records up to
    two runners-up as conflict1/conflict2."""
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
