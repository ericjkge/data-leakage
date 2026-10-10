"""Evaluation (paper Sec 4.1, App. B.3): search infini-gram for candidate documents, then refine
by anchoring on the longest common substring and extending it with up to k mismatches.

Matching lengths are counted in the target model's tokens; infini-gram queries use Llama-2 tokens.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

API = "https://api.infini-gram.io/"
CACHE = Path("cache/infinigram")
LLAMA_TOKENIZER = "NousResearch/Llama-2-7b-hf"  # ungated copy of the Llama-2 tokenizer

WINDOW, STRIDE = 16, 8  # n-gram probes (Llama-2 tokens); any exact run >= 23 tokens gets probed
MAX_DOCS = 3  # documents fetched per hit
SPAN, KS, MIN_DIV = 50, (0, 5, 10), 0.1

POOL = ThreadPoolExecutor(4)


@cache
def llama_tok():
    return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)


def query(payload):
    """POST to infini-gram with a disk cache and retries."""
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = CACHE / key[:2] / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        try:
            res = requests.post(API, json=payload, timeout=60).json()
            if "error" not in res and "message" not in res:  # "message" is e.g. "Too Many Requests"
                break
            print("infini-gram error:", res)
        except (requests.RequestException, ValueError) as e:
            print("infini-gram request failed:", e)
        time.sleep(2**attempt)
    else:
        raise RuntimeError(f"infini-gram query failed: {payload}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def diversity(ids):
    return len(set(ids)) / max(len(ids), 1)


def search(text, index):
    """Return text of candidate corpus documents (each a window around an n-gram hit)."""
    ids = llama_tok()(text, add_special_tokens=False).input_ids
    starts = list(range(0, max(len(ids) - WINDOW, 0) + 1, STRIDE))
    counts = list(POOL.map(lambda s: query({"index": index, "query_type": "count", "query_ids": ids[s : s + WINDOW]})["count"], starts))

    # Group consecutive hit windows into runs; search each run as one long n-gram when possible.
    runs, i = [], 0
    while i < len(starts):
        if counts[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(starts) and counts[j + 1] > 0:
            j += 1
        runs.append((i, j))
        i = j + 1

    needles = []
    for i, j in runs:
        needle = ids[starts[i] : starts[j] + WINDOW]
        found = query({"index": index, "query_type": "find", "query_ids": needle})
        if found["cnt"] == 0:  # hits came from different documents; fall back to the rarest window
            best = min(range(i, j + 1), key=lambda w: counts[w])
            needle = ids[starts[best] : starts[best] + WINDOW]
            found = query({"index": index, "query_type": "find", "query_ids": needle})
        ranks = [(s, r) for s, (lo, hi) in enumerate(found["segment_by_shard"]) for r in range(lo, min(hi, lo + MAX_DOCS))]
        needles += [(needle, s, r) for s, r in ranks[:MAX_DOCS]]

    disp = 2 * len(ids) + 2 * WINDOW  # enough context on both sides of the needle to cover the generation
    docs = POOL.map(
        lambda x: query({"index": index, "query_type": "get_doc_by_rank", "query_ids": x[0], "s": x[1], "rank": x[2], "max_disp_len": disp}),
        needles,
    )
    return ["".join(span for span, _ in d["spans"]) for d in docs]


def longest_common_substring(a, b):
    """Return (length, end_a, end_b) of the longest common contiguous run of a and b."""
    a, b = np.asarray(a), np.asarray(b)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    best = (0, 0, 0)
    for i in range(len(a)):
        cur = np.zeros_like(prev)
        eq = a[i] == b
        cur[1:][eq] = prev[:-1][eq] + 1
        j = int(cur.argmax())
        if cur[j] > best[0]:
            best = (int(cur[j]), i + 1, j)
        prev = cur
    return best


def extend(a, b, k):
    """Banded edit-distance DP aligning prefixes of a (generation) and b (document).
    Returns f where f[e] = longest prefix of a alignable to some prefix of b with <= e edits."""
    INF = k + 1
    f = [0] * (k + 1)
    prev = {j: j for j in range(min(k, len(b)) + 1)}  # row i=0: cost of skipping j doc tokens
    for i in range(1, len(a) + 1):
        cur = {}
        for j in range(max(0, i - k), min(len(b), i + k) + 1):
            c = prev.get(j, INF) + 1  # extra generation token
            if j > 0:
                c = min(c, cur.get(j - 1, INF) + 1, prev.get(j - 1, INF) + (a[i - 1] != b[j - 1]))
            if c <= k:
                cur[j] = c
        if not cur:
            break
        f[min(cur.values())] = i
        prev = cur
    for e in range(1, k + 1):  # f[e] = max over budgets <= e
        f[e] = max(f[e], f[e - 1])
    return f


def refine(gen, doc, ks=KS):
    """Anchor on the LCS of gen/doc token lists and extend both ways. Returns {k: (start, end)} spans in gen."""
    n, ga, db = longest_common_substring(gen, doc)
    gs, ds = ga - n, db - n
    kmax = max(ks)
    right = extend(gen[ga:], doc[db:], kmax)
    left = extend(gen[:gs][::-1], doc[:ds][::-1], kmax)
    spans = {}
    for k in ks:
        e = max(range(k + 1), key=lambda e: left[e] + right[k - e])
        spans[k] = (gs - left[e], ga + right[k - e])
    return spans


def match(generation, index, tok):
    """Score one generation. `tok` is the target model's tokenizer (used for the 50-token count)."""
    gen = tok(generation, add_special_tokens=False).input_ids
    result = {"len": len(gen), "div": diversity(gen)}
    best = {k: (0, 0) for k in KS}
    if result["div"] >= MIN_DIV:
        for doc in search(generation, index):
            spans = refine(gen, tok(doc, add_special_tokens=False).input_ids)
            for k, (s, e) in spans.items():
                if e - s > best[k][1] - best[k][0]:
                    best[k] = (s, e)
    for k, (s, e) in best.items():
        span = gen[s:e]
        result[f"len_k{k}"] = len(span)
        result[f"M{k}@{SPAN}"] = len(span) >= SPAN and diversity(span) >= MIN_DIV
    result["text"] = tok.decode(gen[slice(*best[max(KS)])])
    return result
