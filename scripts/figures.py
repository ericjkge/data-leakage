"""python -m scripts.figures: the reproduction's figures. Writes results/figures/repro_fig{1,2,3}_*.{png,pdf}.

Reads only results/{model}/{attack}/ (prompts.jsonl for CIA's entropies, matches.jsonl for everything else; `div`
there is whole-generation token diversity in the model's own tokenizer). EA is ea_sample, as in the main table.
Figures carry no title; captions go in the report.
1. VM@50 per attack, the paper's Table 1 next to ours, for the two models we share with it.
2. CIA against RSA (its unoptimized start) by model: entropy reached, outputs that don't collapse, VM@50.
3. The paper's Figure 3 with our data: share of generations at or above each diversity threshold.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/figures")
NAMES = {"ra": "RA", "ea_sample": "EA", "rwa": "RWA", "rsa": "RSA", "cia": "CIA"}
ORDER = ["ra", "ea_sample", "rwa", "rsa", "cia"]  # the paper's Table 1 order
PAPER_VM50 = {  # Ko et al. Table 1
    "llama2-70b": {"ra": 0.2, "ea_sample": 0.8, "rwa": 8.6, "rsa": 10.8, "cia": 22.2},
    "olmo-7b": {"ra": 0.0, "ea_sample": 0.2, "rwa": 1.4, "rsa": 1.2, "cia": 6.0},
}
MODELS = {"llama2-7b": "Llama-2\n7B", "olmo-7b": "OLMo\n7B", "pythia-12b": "Pythia\n12B", "llama2-70b": "Llama-2\n70B"}
BLACK, GREY, LIGHT, DARK = "black", "#9a9a9a", "#7fb2dd", "#08519c"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight"})
ERR = {"elinewidth": 1, "ecolor": BLACK, "capsize": 3, "capthick": 1}


def rows(model, attack, stage="matches"):
    return [json.loads(line) for line in open(Path("results") / model / attack / f"{stage}.jsonl")]


def rate_ci(k, n, z=1.96):
    """Rate in percent and its Wilson 95% interval as (below, above) distances."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, (100 * (p - max(0, c - h)), 100 * (min(1, c + h) - p))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / name}.png/.pdf")


def bars(ax, x, vals, w, color, label):
    ax.bar(x, [p for p, _ in vals], w, yerr=np.array([e for _, e in vals]).T, color=color, error_kw=ERR, label=label)


def fig1_paper_vs_ours():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x, w = np.arange(len(ORDER)), 0.38
    for ax, m, name in zip(axes, ["llama2-70b", "olmo-7b"], ["Llama-2 70B", "OLMo 7B"]):
        ours = [rate_ci(sum(r["M0@50"] for r in rs), len(rs)) for rs in (rows(m, a) for a in ORDER)]
        ax.bar(x - w / 2, [PAPER_VM50[m][a] for a in ORDER], w, color=LIGHT, label="paper")
        bars(ax, x + w / 2, ours, w, DARK, "ours")
        ax.set_xticks(x, [NAMES[a] for a in ORDER])
        ax.set(xlabel=name, ylim=(0, 40))
    axes[0].set(ylabel="generations with a 50-token\nexact match (%)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    save(fig, "repro_fig1_paper_vs_ours")


def fig2_cia_by_model():
    x, w = np.arange(len(MODELS)), 0.38
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.2))
    before, after = [], []
    for m in MODELS:
        p = rows(m, "cia", "prompts")
        for out, key in ((before, "init_entropy"), (after, "entropy")):
            v = np.array([r[key] for r in p])
            out.append((v.mean(), (1.96 * v.std() / math.sqrt(len(v)),) * 2))
    bars(a, x - w / 2, before, w, BLACK, "random start (RSA)")
    bars(a, x + w / 2, after, w, DARK, "after optimization (CIA)")
    a.set(ylabel="snippet entropy (nats)", ylim=(0, 11))
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for ax, test, label in ((b, lambda r: r["div"] >= 0.1, "generations not collapsed\ninto repetition (%)"),
                            (c, lambda r: r["M0@50"], "generations with a 50-token\nexact match (%)")):
        for i, (attack, color, name) in enumerate((("rsa", BLACK, "RSA"), ("cia", DARK, "CIA"))):
            vals = [rate_ci(sum(map(test, rs)), len(rs)) for rs in (rows(m, attack) for m in MODELS)]
            bars(ax, x + (i - 0.5) * w, vals, w, color, name)
        ax.set(ylabel=label)
    b.set_ylim(0, 100)
    c.set_ylim(0, 40)
    b.legend(loc="upper left")
    for ax in (a, b, c):
        ax.set_xticks(x, list(MODELS.values()))
    fig.tight_layout(w_pad=1.5)
    save(fig, "repro_fig2_cia_by_model")


def fig3_diversity():
    style = {"cia": (DARK, "-", 2), "rsa": (BLACK, "-", 1.2), "rwa": (LIGHT, "-", 1.5),
             "ra": (GREY, ":", 1.2), "ea_sample": (GREY, "--", 1.2)}
    t = np.linspace(0, 0.5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (m, name) in zip(axes, [("llama2-70b", "Llama-2 70B"), ("olmo-7b", "OLMo 7B")]):
        for a in ["cia", "rsa", "rwa", "ra", "ea_sample"]:
            d = np.array([r["div"] for r in rows(m, a)])
            color, ls, lw = style[a]
            ax.plot(t, [100 * (d >= v).mean() for v in t], color=color, ls=ls, lw=lw, label=NAMES[a])
        ax.set(xlabel=f"diversity threshold, {name}", xlim=(0, 0.5), ylim=(0, 100))
    axes[0].set(ylabel="generations at or above (%)")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout(w_pad=3)
    save(fig, "repro_fig3_diversity")


if __name__ == "__main__":
    fig1_paper_vs_ours()
    fig2_cia_by_model()
    fig3_diversity()
