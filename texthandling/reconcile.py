from . import mistral_client


def reconcile(client, resolved, model=mistral_client.DEFAULT_MODEL, prompt=None):
    """
    resolved : dissolve.dissolve_duplicates() output. Entries with a
        conflict have their "text"/"label" updated IN PLACE per the
        model's ruling; conflict1/conflict2 are left untouched as a record
        of what was disputed. Returns `resolved` for convenience.
    """
    conflicts = [r for r in resolved if r.get("conflict1")]
    if not conflicts:
        return resolved

    context = [r for r in resolved if not r.get("conflict1")]

    # Diagnostic logging -- this pipeline runs fully automated with no
    # manual review step, so a wrong reconciliation (e.g. fields from two
    # different real values getting cross-paired) needs to be diagnosable
    # from logs alone, same pattern as processing/utils.py's
    # safe_intersection. Print what went IN before the call, so a bad
    # result can be traced back to the exact all_readings that produced it.
    print(f"reconcile: {len(conflicts)} conflict(s) to resolve")
    for i, r in enumerate(conflicts):
        print(f"  conflict {i}: all_readings={r.get('all_readings')!r}")

    raw_resolutions = mistral_client.reconcile_conflicts(client, context, conflicts, model=model, prompt=prompt)
    print(f"reconcile: raw model response: {raw_resolutions!r}")

    resolutions = {res["index"]: res for res in raw_resolutions}

    for i, r in enumerate(conflicts):
        resolution = resolutions.get(i)
        if resolution is None:
            print(f"  conflict {i}: no resolution returned, keeping dissolve.py's pick {r['text']!r}")
            continue
        if resolution.get("final_text") and resolution["final_text"] != r["text"]:
            print(f"  conflict {i}: text {r['text']!r} -> {resolution['final_text']!r}")
        if resolution.get("final_text"):
            r["text"] = resolution["final_text"]
        if resolution.get("final_label"):
            r["label"] = resolution["final_label"]

    return resolved
