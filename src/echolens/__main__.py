"""
Package entrypoint for Echolens.
Allows running: python -m echolens [command] [options]
"""

import sys
from echolens.cli import main

if __name__ == "__main__":
    sys.exit(main())
