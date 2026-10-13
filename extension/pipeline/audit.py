"""The blind hand check of every automatic hit (extension/docs/PLAN.md step 5; D-31, D-59).

    python -m extension.pipeline.audit sheet     # writes extension/audit/sheet.csv to label, and key.json
    python -m extension.pipeline.audit collect   # reads the labelled sheet.csv; writes labels.jsonl and agreement.json

The sheet has one row per automatic hit plus 100 random generations with no hit, in random order, with attack and
seed hidden. Rows 1-50 are labelled by two people separately (label_1, label_2); the rest by one (label_1). Where
the two disagree, they agree a `final`. Don't open key.json until labelling is done: it says which attack made
each row. After `collect`, rerun `python -m extension.pipeline.score tables`.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from extension.pipeline.checks import first_occurrence, locate
from extension.pipeline.score import (ATTACKS, LABELS, RESULTS, SEEDS, grams, hit_id, model_name, people, read_jsonl,
                                      tokenizer)

AUDIT = Path("extension/audit")
SHEET, KEY = AUDIT / "sheet.csv", AUDIT / "key.json"
N_MISSES = 100  # random generations with no automatic hit, to catch missed leaks (D-31)
N_DOUBLE = 50  # rows labelled by two people
LABEL_SET = {"real", "false alarm", "unsure"}
CONTEXT = 300  # characters of generation shown either side of the match
COLUMNS = ["row", "kind", "what_matched", "generation", "training_text", "label_1", "label_2", "final", "notes"]


def around(text, needle, width=CONTEXT):
    """text cut to `width` characters either side of the first case-insensitive occurrence of needle."""
    i = text.lower().find(needle.lower())
    if i < 0:
        return text[: 2 * width]
    return ("…" if i > width else "") + text[max(0, i - width) : i + len(needle) + width] + \
           ("…" if i + len(needle) + width < len(text) else "")


def doc_with(person, needle):
    """The person's first document containing needle (any capitalization), else their article."""
    return next((d for d in person["documents"] if needle.lower() in d.lower()), person["documents"][0])


def span_sources(s, span_text):
    """For a 50-token match: the trained planted document sharing the most 50-token windows with it, and how many
    different trained people's documents contain any of its windows (many = shared template wording)."""
    tok = tokenizer()
    g = grams(tok(span_text, add_special_tokens=False).input_ids)
    best, best_n, owners = None, 0, set()
    for p in people(s):
        if p["copies"] == 0:
            continue
        for d in p["documents"]:
            n = int(np.isin(g, grams(tok(d, add_special_tokens=False).input_ids)).sum())
            if n:
                owners.add(p["uid"])
            if n > best_n:
                best, best_n = (p, d), n
    return best, len(owners)


def sheet():
    rng = random.Random("audit")
    rows, quiet = [], []
    for s in SEEDS:
        by_uid = {p["uid"]: p for p in people(s)}
        for a in ATTACKS:
            m = model_name(s)
            gens = {g["id"]: g["generation"] for g in read_jsonl(RESULTS / m / a / "generations.jsonl")}
            for r in read_jsonl(RESULTS / m / a / "leaks.jsonl"):
                gen, found = gens[r["id"]], False
                for h in r["ids"]:
                    if h["trained"] or h["copies"] == 0:
                        p = by_uid[h["uid"]]
                        ident = next(i for i in p["identifiers"] if i["field"] == h["field"])
                        a0, a1 = locate(h["field"], ident["key"], gen)
                        doc, b0, b1 = first_occurrence(p, ident)
                        rows.append({"hit_id": hit_id(m, a, r["id"], "id", f"{h['uid']}:{h['field']}"),
                                     "kind": "identifier", "what_matched": f"{p['name']}'s {h['field']}: {h['value']}",
                                     "generation": around(gen, gen[a0:a1]), "training_text": around(doc, doc[b0:b1])})
                        found = True
                for x in r["planted_vm50"]:
                    src, n_people = span_sources(s, x["text"])
                    rows.append({"hit_id": hit_id(m, a, r["id"], "planted_vm50", x["start"]), "kind": "50-token match",
                                 "what_matched": x["text"],
                                 "generation": around(gen, x["text"][:40], CONTEXT // 2),
                                 "training_text": (f"[its wording appears in {n_people} trained people's documents] "
                                                   + (f"{src[0]['name']}: " + around(src[1], x["text"][:40]) if src else ""))})
                    found = True
                for kind, hits in (("health", r["health"]), ("health_shuffled", r["health_shuffled"])):
                    for h in hits:
                        if kind == "health" and not h["trained"] or h["copies"] == 0:
                            continue
                        p = by_uid[h["uid"]]
                        rows.append({"hit_id": hit_id(m, a, r["id"], kind, f"{h['uid']}:{h['field']}"),
                                     "kind": "health detail", "what_matched": f"{p['name']} + {h['field']}: {h['value']}",
                                     "generation": around(gen, p["name"]),
                                     "training_text": around(doc_with(p, p["name"]), p["name"])})
                        found = True
                if not found:
                    quiet.append({"hit_id": f"{m}|{a}|{r['id']}|none|", "kind": "no automatic hit",
                                  "what_matched": "", "generation": gen[:2 * CONTEXT], "training_text": ""})
    rows += rng.sample(quiet, N_MISSES)
    rng.shuffle(rows)
    AUDIT.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"row": i, **{k: r[k] for k in ("kind", "what_matched", "generation", "training_text")},
                        "label_2": "" if i <= N_DOUBLE else "-"})
    json.dump({i: r["hit_id"] for i, r in enumerate(rows, 1)}, open(KEY, "w"), indent=1)
    print(f"{len(rows)} rows ({len(rows) - N_MISSES} hits, {N_MISSES} without) -> {SHEET}")
    print(Counter(r["kind"] for r in rows))


def kappa(a, b):
    """Cohen's kappa for two lists of labels."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def collect():
    key = json.load(open(KEY))
    rows = list(csv.DictReader(open(SHEET, newline="")))
    out, double, missed, problems = [], [], [], []
    for r in rows:
        l1, l2, fin = (r[k].strip().lower() for k in ("label_1", "label_2", "final"))
        if int(r["row"]) <= N_DOUBLE:
            if l1 not in LABEL_SET or l2 not in LABEL_SET:
                problems.append(f"row {r['row']}: needs label_1 and label_2")
                continue
            double.append((l1, l2))
            label = l1 if l1 == l2 else fin
        else:
            label = fin or l1
        if label not in LABEL_SET:
            problems.append(f"row {r['row']}: label must be one of {sorted(LABEL_SET)}" + (" (set `final`)" if l1 != l2 else ""))
            continue
        hid = key[r["row"]]
        if hid.split("|")[3] == "none":
            if label != "false alarm":
                missed.append({"row": r["row"], "hit_id": hid, "label": label, "notes": r["notes"]})
        else:
            out.append({"hit_id": hid, "label": label})
    if problems:
        raise SystemExit("fix these rows in sheet.csv first:\n" + "\n".join(problems))
    with open(LABELS, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)
    k = kappa(*zip(*double))
    agreement = {"double_labelled": len(double), "agree": sum(a == b for a, b in double), "kappa": round(k, 3),
                 "relabel_needed": k < 0.6, "labels": dict(Counter(x["label"] for x in out)),
                 "no_hit_rows_flagged": missed}
    json.dump(agreement, open(AUDIT / "agreement.json", "w"), indent=1)
    print(json.dumps(agreement, indent=1))
    if k < 0.6:
        print("kappa below 0.6: sharpen the label definitions and relabel (D-31)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sheet", "collect"])
    globals()[p.parse_args().command]()


if __name__ == "__main__":
    main()
