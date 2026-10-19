"""Redraw every figure from reports/results/ without re-running the experiments.

    python scripts/make_figures.py
    python scripts/make_figures.py --smoke    # figures of the smoke run (data/cache/smoke/)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortfair.figures import make_all  # noqa: E402

if __name__ == "__main__":
    results = ROOT / "reports" / "results"
    figures = ROOT / "reports" / "figures"
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        results, figures = ROOT / "data" / "cache" / "smoke" / "results", ROOT / "data" / "cache" / "smoke" / "figures"
    for p in make_all(results, figures):
        print("wrote", p.relative_to(ROOT))
