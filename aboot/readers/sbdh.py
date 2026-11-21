from .base import MappedReader

class SBDHReader(MappedReader):
    source = "sbdh"
    # Caller supplies field mappings once the release/schema is known.
