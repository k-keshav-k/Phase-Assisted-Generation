from __future__ import annotations

import sys

from pag.orchestration.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["phases", *sys.argv[1:]]))
