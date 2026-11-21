import csv, json
from pathlib import Path
from ..schema import Record, binary

class MappedReader:
    """CSV/JSONL adapter with an explicit field map; no guessed label semantics."""
    source = "mapped"

    def read(self, path, mapping):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        else: raise ValueError("Expected CSV or JSONL; add an adapter for other formats")
        out = []
        for i, row in enumerate(rows):
            text = row[mapping["text"]]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string text at row {i + 1}")
            group = str(row[mapping["group_id"]])
            if not group: raise ValueError("group_id cannot be empty")
            labels = {}
            for code, field in mapping["labels"].items():
                value = row.get(field)
                if value is not None and value != "": labels[code] = binary(value)
            rid = str(row[mapping["id"]]) if mapping.get("id") else f"{self.source}:{i:07d}"
            out.append(Record(rid, group, text, labels, self.source,
                              str(row.get(mapping.get("split", ""), "")),
                              str(row.get(mapping.get("note_id", ""), ""))))
        if len({r.id for r in out}) != len(out): raise ValueError("Duplicate record id")
        return out
