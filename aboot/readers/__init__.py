from .odd import ODDReader
from .sbdh import SBDHReader
from .ours import OurReader
READERS = {"odd": ODDReader, "sbdh": SBDHReader, "ours": OurReader}
