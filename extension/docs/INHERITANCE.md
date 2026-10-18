# Inheritance

Everything the extension reuses from outside `extension/` (RULES.md, rule 3). Each entry gives the command that checks it and the actual output from running that command. Entries are added before the thing is first used.

---

## I-01 — PANORAMA-Plus, revision `b7774731d37a…`

The private data (D-02). Downloaded once with `hf download srirxml/PANORAMA-Plus --repo-type dataset --local-dir extension/data/panorama-plus`. It needs an HF login and accepting the dataset's terms. Reproducers don't need it, because they rebuild from the committed people list.

```
$ shasum -a 256 extension/data/panorama-plus/data/train-00000-of-00001.parquet
126efc80ff4e9c9f52b9b6eedcaf8b5b70060f2d3ebf0e00eadab46d6af40a0e  extension/data/panorama-plus/data/train-00000-of-00001.parquet
$ head -1 extension/data/panorama-plus/.cache/huggingface/download/data/train-00000-of-00001.parquet.metadata
b7774731d37aff49b5a5a63a3dfe169dc4097c99
```

Checked 2026-10-11. The counts in D-13 (1,233 US people, 284 eligible) were measured on this file.

## I-02 — Dolma v1.7 file list, `allenai/dolma` at revision `7f481405…`

The list of the 2,419 files that make up Dolma v1.7 (D-10). `pipeline/fetch_dolma.py` streams documents from these URLs. No login needed.

```
$ python3 -c "from huggingface_hub import hf_hub_download as d; p=d(\"allenai/dolma\",\"urls/v1_7.txt\",repo_type=\"dataset\",revision=\"7f48140530a023e9ea4c5cfb141160922727d4d3\"); u=open(p).read().split(); print(len(u), u[0])"
2419 https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz
```

Checked 2026-10-11. On the same day, a size request for every file returned 4,507.6 GB in total, matching the README's "4.5 TB". The source proportions in D-11 are copied from the same README's v1.7 table. **Weak point:** the files are hosted on AI2's own server, not on HuggingFace, so the revision pins the list but not the files. `data/dolma.json`'s fingerprint is what catches a change.

## I-03 — OLMo-7B-0424 tokenizer, `allenai/OLMo-7B-0424-hf` at revision `e3c82a08…`

Used to count and cut tokens (D-12, D-43). It is the same tokenizer the model uses, so token counts match training.

```
$ python3 -c "from huggingface_hub import hf_hub_download as d; from tokenizers import Tokenizer; t=Tokenizer.from_file(d(\"allenai/OLMo-7B-0424-hf\",\"tokenizer.json\",revision=\"e3c82a0871580b424d74e68acee866bb007df7df\")); print(t.get_vocab_size(), t.encode(\"Rebecca Flores is allergic to soy.\").ids)"
50280 [1785, 27407, 46644, 310, 26884, 281, 19175, 15]
```

Checked 2026-10-11, with `tokenizers` 0.22.2 and again with 0.23.3 (the pinned version): same output.

## I-04 — Base OLMo-7B attack results from our reproduction, `reproduction/results/olmo-7b/` at commit `d8a69a6`

Quoted in QUESTION.md and D-25 as our prior for H1, and used again by check 4 (D-19).

```
$ python3 -c "
import json
for a in ['rwa','rsa','cia','ra','ea','ea_sample']:
    r=[json.loads(l) for l in open(f'reproduction/results/olmo-7b/{a}/matches.jsonl')]
    print(a, len(r), round(100*sum(x['M0@50'] for x in r)/len(r),1))"
rwa 500 8.2
rsa 500 5.6
cia 500 2.0
ra 500 1.2
ea 500 0.0
ea_sample 500 0.0
```

Checked 2026-10-11. **Weak points, not fixed here:**
- RA used the wrong prompt (D-21).
- The matches include junk such as counting sequences and boilerplate (found by a hand review of the hits).

So these numbers are the reproduction's as-is, not hand-confirmed.

## I-05 — OLMo-7B-0424 weights, `allenai/OLMo-7B-0424-hf` at revision `e3c82a08…`

The model we post-train (D-04). Downloaded on the pod by `pipeline/setup_pod.sh`. These are the same weights the reproduction's `olmo-7b` uses (`reproduction/pipeline/models.yaml`), now pinned to one revision.

```
$ curl -s "https://huggingface.co/api/models/allenai/OLMo-7B-0424-hf/tree/e3c82a0871580b424d74e68acee866bb007df7df"   (sizes and fingerprints, summarized)
model-00001-of-00006.safetensors  4.94 GB  727e38a3d9677209…
model-00002-of-00006.safetensors  4.99 GB  c82bc739daec94c9…
model-00003-of-00006.safetensors  4.92 GB  2abc2ff401aa5513…
model-00004-of-00006.safetensors  4.86 GB  f422c4a102890a8c…
model-00005-of-00006.safetensors  4.86 GB  89f39cd5afc3349f…
model-00006-of-00006.safetensors  2.98 GB  fdb556808f257903…
total 27.6 GB
```

Checked 2026-10-11. The download verifies these fingerprints automatically. **Weak point:** the reproduction's results (I-04) were made without a pinned revision; this revision is the latest on the date checked, so the two should be the same weights, but that isn't proven.

## I-06 — The infini-gram service, index `v4_dolma-v1_7_llama`, through the reproduction's `reproduction/pipeline/matching.py` `query()`

Used by check 1 (D-52) to count planted text in Dolma v1.7. `query()` is the reproduction's cached, retrying client; the counts use the Llama-2 tokenizer, as every infini-gram index does.

```
$ .venv/bin/python -c "from extension.pipeline.checks import dolma_count; print(dolma_count('the United States of America'), dolma_count('Contact: 555-123-4567'))"
4018955 0
```

Checked 2026-10-12. A common phrase is found millions of times, so a count of 0 means absent rather than a broken search. **Weak point:** infini-gram matches tokens, so the same text with different spacing or punctuation is a different search. Check 1 counts both tokenizations of each string (D-52), but not other spellings of the same number.

## I-07 — `lm-evaluation-harness` 0.4.13 (pinned in `requirements.txt`), tasks `mmlu` and `arc_challenge`

Used by check 3 (D-52) to score the base model and each seed (adapter loaded with `peft=`). We use its standard prompts and metrics as they are: MMLU `acc`, ARC-Challenge `acc_norm`.

```
$ .venv/bin/python -m extension.pipeline.checks benchmarks --seed -1
{"model": "base", "mmlu_acc": 47.41, "arc_challenge_acc_norm": 44.88, "pass": true}
```

Checked 2026-10-13 on an H100 (0-shot, lm-eval defaults). Raw outputs are in `extension/results/checks/lm_eval/`.

## Not yet inherited

These will be added before first use: the reproduction's attack, generation and matching code (`reproduction/`).
