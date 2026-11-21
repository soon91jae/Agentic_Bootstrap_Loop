import hashlib, random, re
from collections import defaultdict
from dataclasses import asdict
from .schema import write_json, write_jsonl


def text_key(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def split_records(records, seed=42, fractions=(.7, .15, .15)):
    """Union patients linked by exact normalized contexts before random splitting.

    No label-based seed search or test-dependent split selection. Rare support can
    be zero; the report exposes it. A large connected component may skew ratios.
    """
    if len(fractions) != 3 or abs(sum(fractions)-1)>1e-8 or min(fractions)<=0:
        raise ValueError("Three positive split fractions must sum to one")
    parent = {r.group_id: r.group_id for r in records}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = {}
    for r in records:
        h = text_key(r.text)
        if h in seen: parent[root(r.group_id)] = root(seen[h])
        seen[h] = r.group_id
    components = defaultdict(list)
    for r in records: components[root(r.group_id)].append(r)
    names = sorted(components); random.Random(seed).shuffle(names)
    if len(names)<3: raise ValueError("Fewer than three independent split components")
    # Largest-first row balancing, independent of all labels.
    names.sort(key=lambda g:len(components[g]), reverse=True)
    splits=("train","dev","test"); targets=dict(zip(splits,[len(records)*f for f in fractions]))
    counts={s:0 for s in splits}; assignments={}
    for g in names:
        chosen=max(splits,key=lambda s:targets[s]-counts[s])
        assignments[g]=chosen; counts[chosen]+=len(components[g])
    if any(counts[s]==0 for s in splits):
        raise ValueError("A component prevents a nonempty three-way split; provide reviewed splits")
    for g, rs in components.items():
        for r in rs: r.split = assignments[g]
    validate_splits(records)
    return records


def validate_splits(records):
    groups, texts = {}, {}
    for r in records:
        if r.split not in {"train", "dev", "test"}: raise ValueError("Invalid split")
        for value, table in [(r.group_id, groups), (text_key(r.text), texts)]:
            if value in table and table[value] != r.split:
                raise ValueError("Patient or exact-context leakage across splits")
            table[value] = r.split


def save_dataset(records, definitions, destination):
    from pathlib import Path
    destination = Path(destination)
    for split in ("train", "dev", "test"):
        write_jsonl(destination/f"{split}.jsonl", [asdict(r) for r in records if r.split==split])
    report = {s: {"rows": sum(r.split==s for r in records),
                  "patients": len({r.group_id for r in records if r.split==s}),
                  "positive_support": {c: sum(r.labels.get(c)==1 for r in records if r.split==s)
                                       for c in definitions}} for s in ("train", "dev", "test")}
    report["all"] = {"rows": len(records), "notes": len({r.note_id for r in records}),
        "patients": len({r.group_id for r in records}),
        "duplicate_contexts": len(records)-len({text_key(r.text) for r in records})}
    write_json(destination/"report.json", report)
    write_json(destination/"definitions.json", definitions)
    return report
