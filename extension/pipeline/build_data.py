"""Build one seed's training text: 100 PANORAMA people mixed into the shared Dolma text (extension/docs/PLAN.md, step 1b).

    python -m extension.pipeline.build_data --seed 0

If extension/data/seed_{s}/people.json exists (it is committed), the people and their documents come from it,
so no HuggingFace account is needed. Otherwise they are drawn from the local PANORAMA-Plus download (D-13, D-14,
D-15) and that file is written. Needs extension/data/dolma/replay.jsonl (python -m extension.pipeline.fetch_dolma).

Writes extension/data/seed_{s}/train.jsonl (not committed) and extension/data/seed_{s}/build.json (committed).
Rerunning rebuilds the same file; if the manifest exists and the new file's hash differs, it stops.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from extension.pipeline.fetch_dolma import OLMO_REV
from extension.pipeline.pii import HEALTH_FIELDS, ID_FIELDS, has_health, has_id, id_key, prepare

PANORAMA = Path("extension/data/panorama-plus/data/train-00000-of-00001.parquet")
PANORAMA_REV = "b7774731d37aff49b5a5a63a3dfe169dc4097c99"
DOLMA = Path("extension/data/dolma/replay.jsonl")
N_PEOPLE = 100
# Within each block of 10 people with similar numbers of identifiers: 3 held out, 2 each at 1/5/25 copies, 1 at 100.
BLOCK = [0, 0, 0, 1, 1, 5, 5, 25, 25, 100]
SEP = "\n|\n"  # joins documents for searching; "|" stops a number at the end of one running into the next


def eligible_people():
    """US people with an article, posts, a unique name, >= 2 identifiers and >= 1 health detail in their own text."""
    import pandas as pd

    df = pd.read_parquet(PANORAMA)
    us = df[df["Locale"] == "en_US"]
    names = Counter(us["First Name"] + " " + us["Last Name"])
    people = []
    for _, r in us.iterrows():
        info = json.loads(r["complete_info"]) if isinstance(r["complete_info"], str) else r["complete_info"]
        article = (info.get("SyntheticPassageGeneration") or {}).get("generated_synthetic_passage") or ""
        posts = [p["Text"] for p in (info.get("SyntheticTrainingData") or {}).get("content_pairs") or []]
        name = f"{r['First Name']} {r['Last Name']}"
        if not article or not posts or names[name] > 1:
            continue
        text = SEP.join([article] + posts)
        t = prepare(text)
        ids = [{"field": f, "value": str(r[f]), "key": k} for f in ID_FIELDS
               if (k := id_key(f, r[f])) and has_id(f, k, t)]
        health = [{"field": f, "value": str(r[f])} for f in HEALTH_FIELDS if has_health(r[f], text)]
        if len(ids) < 2 or not health:
            continue
        docs = [article] + [p for p in posts if any(has_id(i["field"], i["key"], prepare(p)) for i in ids)
                            or any(has_health(h["value"], p) for h in health)]
        people.append({"uid": r["Unique ID"], "name": name, "identifiers": ids, "health": health, "documents": docs})
    return sorted(people, key=lambda p: p["uid"])


def draw(seed):
    """100 people drawn with the seed, then assigned to groups block by block (D-15, D-48)."""
    rng = random.Random(seed)
    people = rng.sample(eligible_people(), N_PEOPLE)
    people.sort(key=lambda p: (-len(p["identifiers"]), p["uid"]))
    for start in range(0, N_PEOPLE, len(BLOCK)):
        copies = BLOCK[:]
        rng.shuffle(copies)
        for p, c in zip(people[start : start + len(BLOCK)], copies):
            p["copies"] = c
    return people


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    s = p.parse_args().seed

    subset_path = Path(f"extension/data/seed_{s}/people.json")
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    if subset_path.exists():
        people = json.load(open(subset_path))["people"]
    else:
        people = draw(s)
        json.dump({"source": f"PANORAMA-Plus (srirxml/PANORAMA-Plus, revision {PANORAMA_REV}), CC BY 4.0, "
                             "Selvam & Ghosh 2025, arXiv 2505.12238. Synthetic people; no real persons.",
                   "seed": s, "people": people}, open(subset_path, "w"), indent=1)

    tok = Tokenizer.from_file(hf_hub_download("allenai/OLMo-7B-0424-hf", "tokenizer.json", revision=OLMO_REV))
    dolma = [json.loads(line) for line in open(DOLMA)]
    dolma_sha = hashlib.sha256(DOLMA.read_bytes()).hexdigest()

    # Leak guards: no planted identifier may already be in the Dolma text; names are only counted (common names
    # like "John Smith" can legitimately appear on the web).
    dolma_all = SEP.join(d["text"] for d in dolma)
    dolma_t = prepare(dolma_all)
    for person in people:
        for i in person["identifiers"]:
            assert not has_id(i["field"], i["key"], dolma_t), f"{person['name']}'s {i['field']} appears in Dolma"
    name_hits = {p["name"]: n for p in people if (n := dolma_all.count(p["name"]))}

    rows = [{"source": "dolma", "text": d["text"], "n_tokens": d["n_tokens"]} for d in dolma]
    group_tokens, group_docs = Counter(), Counter()
    for person in people:
        for doc in person["documents"]:
            n = len(tok.encode(doc).ids)
            for _ in range(person["copies"]):
                rows.append({"source": "panorama", "uid": person["uid"], "text": doc, "n_tokens": n})
            group_tokens[person["copies"]] += n * person["copies"]
            group_docs[person["copies"]] += person["copies"]
    random.Random(s).shuffle(rows)

    trained_t = prepare(SEP.join(row["text"] for row in rows))
    for person in people:
        if person["copies"] == 0:
            for i in person["identifiers"]:
                assert not has_id(i["field"], i["key"], trained_t), \
                    f"held-out {person['name']}'s {i['field']} is in the training text"

    out = Path(f"extension/data/seed_{s}/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    manifest_path = Path(f"extension/data/seed_{s}/build.json")
    if manifest_path.exists() and json.load(open(manifest_path))["sha256"] != sha:
        raise SystemExit(f"rebuilt file differs from {manifest_path} (sha256 {sha}); left at {tmp}")
    tmp.replace(out)

    dolma_tokens = sum(d["n_tokens"] for d in dolma)
    private_tokens = sum(group_tokens.values())
    manifest = {
        "seed": s, "panorama_revision": PANORAMA_REV, "dolma_sha256": dolma_sha,
        "tokenizer_revision": OLMO_REV, "sha256": sha, "documents": len(rows),
        "tokens": {"dolma": dolma_tokens, "private": private_tokens, "total": dolma_tokens + private_tokens},
        "private_share": round(private_tokens / (dolma_tokens + private_tokens), 4),
        "groups": {str(c): {"people": sum(p["copies"] == c for p in people),
                            "identifiers": sum(len(p["identifiers"]) for p in people if p["copies"] == c),
                            "documents_with_copies": group_docs[c], "tokens_with_copies": group_tokens[c]}
                   for c in sorted({p["copies"] for p in people})},
        "planted_names_found_in_dolma": name_hits,
    }
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("documents", "tokens", "private_share", "groups",
                                                "planted_names_found_in_dolma")}, indent=1))


if __name__ == "__main__":
    main()
