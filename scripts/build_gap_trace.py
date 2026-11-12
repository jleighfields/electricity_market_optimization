"""Run optimality_gap_demo.py with its default settings and save the convergence trace for the slides.

uv run python scripts/build_gap_trace.py   →   assets/uc_gap_trace.parquet, assets/uc_gap_result.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimality_gap_demo import app  # noqa: E402

_, defs = app.run()                       # runs every cell; defs holds the notebook's variables
trace, result, settings, model = defs["trace"], defs["result"], defs["settings"], defs["model"]
trace.to_parquet(ROOT / "assets" / "uc_gap_trace.parquet")
result = {**result, **settings, "variables": int(model.nvars), "constraints": int(model.ncons),
          "binaries": int(model.binaries.nvars)}
(ROOT / "assets" / "uc_gap_result.json").write_text(json.dumps(result, indent=1, default=float))
print(json.dumps(result, indent=1, default=float))
