from dataclasses import dataclass, asdict
import hashlib, json, math
from pathlib import Path

@dataclass
class Record:
    id: str
    group_id: str
    text: str
    labels: dict
    source: str
    split: str = ""
    note_id: str = ""

@dataclass(frozen=True)
class Pair:
    id: str
    code: str
    definition: str
    text: str


def pairs(records, definitions):
    # Labels and patient metadata deliberately absent from model/teacher inputs.
    return [Pair(r.id, c, d, r.text) for r in records for c, d in definitions.items()]


def key(pair):
    return hashlib.sha256(json.dumps(asdict(pair), sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-35, min(35, x))))


def binary(value):
    # Do not silently convert missing, string categories, or borderline to negatives.
    if isinstance(value, bool): return int(value)
    if str(value).strip() in {"0", "1"}: return int(value)
    raise ValueError(f"Expected an explicit binary label, got {value!r}")
