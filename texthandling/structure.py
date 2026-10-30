from . import mistral_client


def structure(client, blocks, model=mistral_client.DEFAULT_MODEL, prompt=None):
    """
    blocks : blocks.merge_blocks() output.

    Returns (blocks, all_keys): `blocks` with an added "fields" dict per
    block (e.g. {"GRZ": "0,4", "GFZ": "0,7"}, possibly empty), and
    `all_keys` the ordered list of every distinct key discovered across
    all blocks (first-seen order).
    """
    if not blocks:
        return blocks, []

    results = {r["index"]: r for r in mistral_client.structure_blocks(client, blocks, model=model, prompt=prompt)}

    all_keys = []
    seen_keys = set()
    for i, b in enumerate(blocks):
        result = results.get(i)
        fields = {}
        for field in (result.get("fields") if result else []) or []:
            key, value = field.get("key"), field.get("value")
            if not key or not value:
                continue
            # The same key can legitimately appear twice in one block (e.g.
            # two separate SD/KW roof-pitch rows) -- don't silently drop
            # either value by overwriting.
            fields[key] = f"{fields[key]}; {value}" if key in fields else value
            if key not in seen_keys:
                seen_keys.add(key)
                all_keys.append(key)
        b["fields"] = fields

    return blocks, all_keys
