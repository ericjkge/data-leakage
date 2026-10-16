"""The extension's figures for the report and slides, from committed results (FINDINGS.md F-1, F-8, F-9).

    python -m extension.pipeline.figures   # writes extension/results/figures/ext_fig{1,2,3}_*.{png,pdf}

Base OLMo from the reproduction (same model, prompts and decoding; RA's corrected-prompt run, D-21) against the
post-trained models: results/olmo-7b/*/ and results/ext/*/, the hand-check labels and checks/learned_seed_*.json.
Figure 3 re-tokenizes the generations with OLMo's tokenizer (a few minutes). Figures carry no title; captions go
in the report.
1. 50-token matches per attack: base OLMo to Dolma, post-trained models to the planted text (D-55's H2).
2. Identifiers the model completes when given the text before them, against identifiers any attack got out.
3. The paper's Figure 3 with our data, base and post-trained: share of generations whose token diversity is at
   least each threshold.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extension.pipeline.score import (ATTACKS, BASE_MATCHES, GROUPS, LABELS, OUT, RESULTS, SEEDS, flags, model_name, read_jsonl,
                                      tokenizer, wilson)

FIGS = OUT / "figures"
NAMES = {"rsa": "RSA", "rwa": "RWA", "ra": "RA", "ea_sample": "EA", "cia": "CIA"}
ORDER = ["ra", "ea_sample", "rwa", "rsa", "cia"]  # the paper's Table 1 order
BLACK, GREY, LIGHT, DARK = "black", "#9a9a9a", "#7fb2dd", "#08519c"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight"})
ERR = {"elinewidth": 1, "ecolor": BLACK, "capsize": 3, "capthick": 1}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {FIGS / name}.png/.pdf")


def rate_ci(k, n):
    """Rate in percent and its Wilson 95% interval as (below, above) distances."""
    lo, hi = wilson(k, n)
    p = 100 * k / n
    return p, (p - lo, hi - p)


def fig1_base_vs_post(labels):
    """Same model, prompts and decoding before and after post-training. Automatic counts on both sides, as in
    D-55's H2 test (the base side has no hand check)."""
    base, post = [], []
    for a in ORDER:
        m = read_jsonl(Path(BASE_MATCHES[a]) / "matches.jsonl")
        base.append(rate_ci(sum(r["M0@50"] for r in m), len(m)) if len(m) == 500 else None)  # corrected RA: pending
        rows = [r for s in SEEDS for r in flags(model_name(s), a, labels)]
        post.append(rate_ci(sum(r["planted_vm50"] for r in rows), len(rows)))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x, w = np.arange(len(ORDER)), 0.38
    have = [i for i, b in enumerate(base) if b]
    ax.bar(x[have] - w / 2, [base[i][0] for i in have], w, yerr=np.array([base[i][1] for i in have]).T, color=LIGHT,
           error_kw=ERR, label="base OLMo-7B, pretraining data")
    for i in set(range(len(ORDER))) - set(have):
        ax.text(x[i] - w / 2, 0.15, "n/a", ha="center", fontsize=8)
    ax.bar(x + w / 2, [p for p, _ in post], w, yerr=np.array([e for _, e in post]).T, color=DARK, error_kw=ERR,
           label="post-trained OLMo-7B, planted data")
    ax.set_xticks(x, [NAMES[a] for a in ORDER])
    ax.set(ylabel="generations with a 50-token\nexact match (%)", ylim=(0, 12))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "ext_fig1_base_vs_post")


def fig2_memorized(labels):
    known, n = {g: 0 for g in GROUPS}, {g: 0 for g in GROUPS}
    for s in SEEDS:
        by = json.load(open(f"extension/checks/learned_seed_{s}.json"))["summary"]["identifiers_by_copies"]
        for g in GROUPS:
            known[g] += by[str(g)]["hits"]
            n[g] += by[str(g)]["n"]
    leaked = set()  # (seed, person, field, copies): each identifier once, if any attack generation has it (hand-confirmed)
    for s in SEEDS:
        for a in ATTACKS:
            for r in read_jsonl(RESULTS / model_name(s) / a / "leaks.jsonl"):
                for h in r["ids"]:
                    hid = f"{model_name(s)}|{a}|{r['id']}|id|{h['uid']}:{h['field']}"
                    if (h["trained"] or h["copies"] == 0) and (not labels or labels.get(hid) == "real"):
                        leaked.add((s, h["uid"], h["field"], h["copies"]))
    out = {g: sum(1 for *_, c in leaked if c == g) for g in GROUPS}

    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(GROUPS))
    for counts, color, label in [(known, DARK, "completed from the text before it"), (out, LIGHT, "extracted by any attack")]:
        p, e = zip(*(rate_ci(counts[g], n[g]) for g in GROUPS))
        ax.errorbar(x, p, yerr=np.array(e).T, fmt="o-", ms=4, lw=1.5, elinewidth=1, capsize=3, color=color, label=label)
    ax.set_xticks(x, ["0\n(held out)", "1", "5", "25", "100"])
    ax.set(xlabel="copies in post-training", ylabel="planted identifiers (%)", ylim=(0, 100))
    ax.legend(loc="upper left")
    save(fig, "ext_fig2_memorized")


def diversity(text):
    ids = tokenizer()(text, add_special_tokens=False).input_ids
    return len(set(ids)) / max(1, len(ids))


def fig3_diversity():
    # CIA dark; RSA, CIA's unoptimized starting point, in black; RWA light; RA and EA grey
    style = {"cia": (DARK, "-", 2), "rsa": (BLACK, "-", 1.2), "rwa": (LIGHT, "-", 1.5),
             "ra": (GREY, ":", 1.2), "ea_sample": (GREY, "--", 1.2)}
    dirs = {"base": lambda a: [Path(BASE_MATCHES[a])], "post": lambda a: [RESULTS / model_name(s) / a for s in SEEDS]}
    t = np.linspace(0, 0.5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (key, xlabel) in zip(axes, [("base", "base OLMo-7B"), ("post", "post-trained OLMo-7B")]):
        for a in ["cia", "rsa", "rwa", "ra", "ea_sample"]:
            d = np.array([diversity(g["generation"]) for p in dirs[key](a) for g in read_jsonl(p / "generations.jsonl")])
            color, ls, lw = style[a]
            ax.plot(t, [100 * (d >= v).mean() for v in t], color=color, ls=ls, lw=lw, label=NAMES[a])
        ax.set(xlabel=f"diversity threshold, {xlabel}", xlim=(0, 0.5), ylim=(0, 100))
    axes[0].set(ylabel="generations at or above (%)")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    fig.tight_layout(w_pad=3)
    save(fig, "ext_fig3_diversity")


def main():
    labels = {r["hit_id"]: r["label"] for r in read_jsonl(LABELS)}
    fig1_base_vs_post(labels)
    fig2_memorized(labels)
    fig3_diversity()


if __name__ == "__main__":
    main()
