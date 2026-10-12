"""Fetch the Dolma text that is mixed into post-training (DECISIONS D-10, D-11, D-12, D-43).

One shared selection for every seed. For each source, in OLMo-0424's training proportions:
open randomly chosen files, read at most PER_FILE documents from the start of each until the pool holds
POOL_FACTOR x the source's budget, then pick documents from the pool at random until the budget is met.
Every document is cut at MAX_TOKENS OLMo tokens. Nothing is downloaded in full; files are streamed with curl.

    python -m extension.pipeline.fetch_dolma [--target-tokens 8200000]

Writes extension/data/dolma/replay.jsonl (not committed) and extension/selection/dolma.json (committed).
Rerunning rebuilds the same file; if a manifest already exists and the new file's hash differs, it stops.
"""

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DOLMA_REV = "7f48140530a023e9ea4c5cfb141160922727d4d3"  # allenai/dolma, holds urls/v1_7.txt
OLMO_REV = "e3c82a0871580b424d74e68acee866bb007df7df"  # allenai/OLMo-7B-0424-hf, for the tokenizer
SEED = 0
POOL_FACTOR = 2
PER_FILE = 100
MAX_TOKENS = 2048

# source -> (file folders, billions of tokens x OLMo's sampling proportion), from the v1.7 table in
# https://huggingface.co/datasets/allenai/dolma (README, "Summary Statistics (v1.7)")
SOURCES = {
    "common_crawl": (["cc_en_head", "cc_en_middle", "cc_en_tail"], 1195.5 * 0.5),
    "refined_web": (["falcon-refinedweb-filtered"], 456.4),
    "starcoder": (["starcoder"], 263.8),
    "c4": (["c4-filtered"], 138.4 * 0.5),
    "reddit": (["reddit"], 79.9),
    "pes2o": (["pes2o"], 57.2),
    "arxiv": (["redpajama-arxiv"], 28.0),
    "stackexchange": (["redpajama-stackexchange"], 19.6),
    "flan": (["tulu_flan"], 16.5),
    "cc_news": (["cc_news_head", "cc_news_middle", "cc_news_tail"], 14.3),
    "open_web_math": (["proof_pile_2-open_web_math"], 12.6),
    "algebraic_stack": (["proof_pile_2-algebraic_stack"], 12.6),
    "gutenberg": (["books"], 5.3),
    "megawika": (["wikiref_megawika"], 4.6),
    "wikipedia": (["wiki"], 3.7 * 2.0),
}

OUT = Path("extension/data/dolma/replay.jsonl")
MANIFEST = Path("extension/selection/dolma.json")


def read_first(url, n):
    """The first n documents of a gzipped JSON-lines file, streamed. Retries a broken stream from the start."""
    for attempt in range(3):
        proc = subprocess.Popen(["curl", "-sfL", "--max-time", "900", url], stdout=subprocess.PIPE)
        docs = []
        try:
            for line in gzip.GzipFile(fileobj=proc.stdout):
                docs.append(json.loads(line))
                if len(docs) == n:
                    break
        except (OSError, EOFError, json.JSONDecodeError) as e:
            err = e
        finally:
            proc.kill()
            proc.wait()
        if len(docs) == n:
            return docs
        if attempt == 2:
            raise RuntimeError(f"{url}: got {len(docs)} of {n} documents ({locals().get('err')})")


def cut(tok, text):
    """Text cut at MAX_TOKENS tokens (on a character boundary), and its token count."""
    enc = tok.encode(text)
    if len(enc.ids) <= MAX_TOKENS:
        return text, len(enc.ids)
    return text[: enc.offsets[MAX_TOKENS - 1][1]], MAX_TOKENS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-tokens", type=int, default=8_200_000)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    tok = Tokenizer.from_file(hf_hub_download("allenai/OLMo-7B-0424-hf", "tokenizer.json", revision=OLMO_REV))
    urls_path = hf_hub_download("allenai/dolma", "urls/v1_7.txt", repo_type="dataset", revision=DOLMA_REV)
    urls = [u.strip() for u in open(urls_path) if u.strip()]
    total_weight = sum(w for _, w in SOURCES.values())

    chosen, report = [], {}
    with ThreadPoolExecutor(args.threads) as ex:
        for source, (folders, weight) in SOURCES.items():
            budget = round(args.target_tokens * weight / total_weight)
            files = sorted(u for u in urls if u.split("/")[-2] in folders)
            rng = random.Random(f"{SEED}-{source}")
            rng.shuffle(files)

            pool, pooled, opened = [], 0, 0
            for i in range(0, len(files), args.threads):  # fetch in parallel, consume in order
                for url, docs in zip(files[i : i + args.threads], ex.map(lambda u: read_first(u, PER_FILE), files[i : i + args.threads])):
                    opened += 1
                    for d in docs:
                        if not d.get("text", "").strip():
                            continue
                        text, n = cut(tok, d["text"])
                        pool.append({"source": source, "url": url, "id": d["id"], "n_tokens": n, "text": text})
                        pooled += n
                    if pooled >= POOL_FACTOR * budget:
                        break
                if pooled >= POOL_FACTOR * budget:
                    break
            if pooled < POOL_FACTOR * budget:
                print(f"warning: {source} pool has {pooled} tokens, wanted {POOL_FACTOR * budget}")

            rng.shuffle(pool)
            got = 0
            for d in pool:
                if got >= budget:
                    break
                chosen.append(d)
                got += d["n_tokens"]
            report[source] = {"budget_tokens": budget, "tokens": got, "docs": sum(d["source"] == source for d in chosen),
                              "files_opened": opened, "pool_tokens": pooled}
            print(f"{source:16s} budget {budget:>9,}  got {got:>9,}  from {opened} files")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for d in chosen:
            f.write(json.dumps(d) + "\n")
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()

    if MANIFEST.exists() and json.load(open(MANIFEST))["sha256"] != sha:
        raise SystemExit(f"rebuilt file differs from {MANIFEST} (sha256 {sha}); left at {tmp}")
    tmp.replace(OUT)

    total = sum(r["tokens"] for r in report.values())
    for r in report.values():
        r["share"] = round(r["tokens"] / total, 4)
    docs_by_file = {}
    for d in chosen:
        docs_by_file.setdefault(d["url"], []).append(d["id"])
    json.dump({"dolma_revision": DOLMA_REV, "tokenizer_revision": OLMO_REV, "seed": SEED, "pool_factor": POOL_FACTOR,
               "per_file": PER_FILE, "max_tokens": MAX_TOKENS, "target_tokens": args.target_tokens, "tokens": total,
               "docs": len(chosen), "sha256": sha, "sources": report, "docs_by_file": docs_by_file},
              open(MANIFEST, "w"), indent=1)
    print(f"{len(chosen):,} documents, {total:,} tokens -> {OUT} (sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
