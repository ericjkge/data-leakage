"""python -m scripts.table: % of generations with a VM/M5/M10@50 match, per (model, attack)."""

import json
from pathlib import Path

from src.matching import KS, SPAN

print("| model | attack | n | " + " | ".join(f"{'VM' if k == 0 else f'M{k}'}@{SPAN}" for k in KS) + " |")
print("|---|---|---|" + "---|" * len(KS))
for path in sorted(Path("results").glob("*/*/matches.jsonl")):
    rows = [json.loads(line) for line in open(path)]
    rates = [100 * sum(r[f"M{k}@{SPAN}"] for r in rows) / len(rows) for k in KS]
    print(f"| {path.parts[1]} | {path.parts[2]} | {len(rows)} | " + " | ".join(f"{x:.1f}" for x in rates) + " |")
