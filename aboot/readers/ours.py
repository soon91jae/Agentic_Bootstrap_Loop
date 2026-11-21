from .base import MappedReader

class OurReader(MappedReader):
    source = "ours"
    # Handles flat binary labels. Span/event releases require explicit conversion.
