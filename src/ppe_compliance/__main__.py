"""Enable ``python -m ppe_compliance``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
