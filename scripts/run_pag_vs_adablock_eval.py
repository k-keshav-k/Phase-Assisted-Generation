from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LLADA_DIR = ROOT / "AdaBlock-dLLM" / "llada"
if str(LLADA_DIR) not in sys.path:
    sys.path.insert(0, str(LLADA_DIR))

dummy_api = importlib.import_module("run_pag_dummy_api")
dummy_api.DEFAULT_PREDICTOR_CKPT = ( # type: ignore
    ROOT
    / "output"
    / "ablations"
    / "large_ws67_d256_h8_l3_dp10_lr0.5_bestval=0.412367.pt"
)

eval_mod = importlib.import_module("run_pag_vs_adablock_eval")

if __name__ == "__main__":
    sys.exit(eval_mod.main())
