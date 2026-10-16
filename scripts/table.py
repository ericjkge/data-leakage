"""python -m scripts.table: % of generations with a VM/M5/M10@50 match, per (model, attack). Writes results/table.{md,png}."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.matching import KS, SPAN

MODELS = ["olmo-7b", "pythia-12b", "llama2-70b"]  # llama2-7b results stay in results/ but are not in the main table
ATTACKS = ["cia", "ra", "ea_sample", "rwa", "rsa"]  # ea_sample (fixed 30 x EOS, temperature 1.0) is reported as EA

header = ["model", "attack", "n"] + [f"{'VM' if k == 0 else f'M{k}'}@{SPAN}" for k in KS]
body = []
for model in MODELS:
    for attack in ATTACKS:
        path = Path("results") / model / attack / "matches.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path)]
        rates = [100 * sum(r[f"M{k}@{SPAN}"] for r in rows) / len(rows) for k in KS]
        body.append([model, "ea" if attack == "ea_sample" else attack, str(len(rows))] + [f"{x:.1f}" for x in rates])

lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]
Path("results/table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Paper-style figure (Table 1 layout): one column group per model, baselines above a rule, CIA in bold below it.
NAMES = {"llama2-70b": "Llama 2 (70B)", "olmo-7b": "OLMo (7B)", "pythia-12b": "Pythia (12B)"}
ROWS = [("ra", "RA"), ("ea", "EA"), ("rwa", "RWA"), ("rsa", "RSA (baseline)"), ("cia", "CIA (Ours)")]
FIG_MODELS = ["llama2-70b", "olmo-7b", "pythia-12b"]
vals = {(r[0], r[1]): [float(x) for x in r[3:]] for r in body}
best = {(m, j): max(vals[m, a][j] for a, _ in ROWS) for m in FIG_MODELS for j in range(len(KS))}

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(9.5, 3.9))
ax.axis("off")
ax.set_xlim(0, 10.7)
ax.set_ylim(0, 6.4)
x0, dx, gw = 2.3, 2.8, 2.7
cols = [x0 + g * dx + (j + 0.5) * gw / len(KS) for g in range(3) for j in range(len(KS))]
ax.plot([0, 10.7], [6.2, 6.2], "k", lw=1.5)
ax.text(0.15, 5.55, "Attack Method", weight="bold", fontsize=11, va="center")
for g, m in enumerate(FIG_MODELS):
    ax.text(x0 + g * dx + gw / 2, 5.9, NAMES[m], weight="bold", fontsize=11, ha="center", va="center")
    ax.plot([x0 + g * dx, x0 + g * dx + gw], [5.7, 5.7], "k", lw=0.7)
    for j, k in enumerate(KS):
        ax.text(cols[g * len(KS) + j], 5.4, f"{'VM' if k == 0 else f'M{k}'}@{SPAN}", fontsize=9, ha="center", va="center")
ax.plot([0, 10.7], [5.1, 5.1], "k", lw=0.8)
y = 4.7
for attack, label in ROWS:
    if attack == "cia":
        y -= 0.1
        ax.plot([0, 10.7], [y + 0.4, y + 0.4], "k", lw=0.8)
    ax.text(0.15, y, label, fontsize=10.5, va="center", weight="bold" if attack == "cia" else "normal")
    for g, m in enumerate(FIG_MODELS):
        for j in range(len(KS)):
            v = vals[m, attack][j]
            ax.text(cols[g * len(KS) + j], y, f"{v:.1f}", fontsize=10.5, ha="center", va="center", weight="bold" if v == best[m, j] else "normal")
    y -= 0.5
ax.plot([0, 10.7], [y + 0.25, y + 0.25], "k", lw=1.5)
ax.text(0.15, y - 0.2, "Attack success rates (%) with 50-token verbatim (VM@50) and ≤5 / ≤10-mismatch (M5@50, M10@50) matches.\n"
        "n = 500 per cell except Llama 2 (70B) CIA (n = 100). EA: fixed 30×EOS prompt, temperature-1.0 sampling.\n"
        "The best result for each metric and model is in bold.", fontsize=8.5, va="top", linespacing=1.4)
fig.savefig("results/table.png", dpi=200, bbox_inches="tight")
