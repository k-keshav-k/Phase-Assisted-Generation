from __future__ import annotations

import sys

from pag.orchestration.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["adaptive", *sys.argv[1:]]))
