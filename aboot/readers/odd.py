import re
from .base import MappedReader

COLUMNS = {
    "opioids": "Opioids", "indication": "Indication",
    "benzodiazepines": "Benzodiazepines", "cns": "Central nervous system related",
    "med_change": "Medication change", "sdoh": "Social determinant of health",
    "oud": "Diagnosed opioid dependence", "cab": "Confirmed aberrant behavior",
    "sab": "Suggested aberrant behavior",
}

class ODDReader(MappedReader):
    source = "odd"

    def read(self, path, mapping=None):
        if mapping is not None: return super().read(path, mapping)
        records = super().read(path, {
            "text": "context", "group_id": "note_id", "note_id": "note_id", "labels": COLUMNS})
        for r in records:
            if not re.fullmatch(r"\d+-DS-\d+", r.note_id):
                raise ValueError("Unknown note_id structure: provide an explicit patient group map")
            r.group_id = r.note_id.split("-")[0]
        return records
