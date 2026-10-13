# Plan

The procedure: what gets built and run, in what order, and how long it takes.
- The reasons for every choice are in [DECISIONS.md](DECISIONS.md), cited as D-xx. If this file and DECISIONS.md ever disagree, DECISIONS.md wins, and this file gets fixed.
- What we're testing is in [QUESTION.md](QUESTION.md).
- The exact commands for the GPU pod are in [../RUN.md](../RUN.md).

**Status (2026-10-11):**
- Steps 1 and 2 are written and tested: data built and verified for all three seeds; training tested end to end on a tiny model (D-45, D-48, D-49, D-50).
- Next: step 2 on the H100.

## 1. Data (D-09 to D-15, D-43, D-48) — done
- **Dolma text, shared by all seeds:** `python -m extension.pipeline.fetch_dolma`.
  - Streams 8.2 million tokens of Dolma v1.7 in OLMo's training proportions.
  - Writes `data/dolma/replay.jsonl` (not committed) and `selection/dolma.json` (committed: which documents were picked, plus a fingerprint).
- **Each seed's training text:** `python -m extension.pipeline.build_data --seed s`, for s = 0, 1, 2.
  1. Draws 100 of the 284 eligible people with seed s.
  2. Assigns them in blocks of 10 to held out (30) or 1, 5, 25, 100 copies (20, 20, 20, 10).
  3. Scatters the copies of their documents among the Dolma documents.
  4. Writes `data/seed_s/train.jsonl` (not committed), plus `selection/seed_s/people.json` and `build.json` (committed).
- **Both commands stop with an error if:**
  - a rebuild doesn't match its committed fingerprint
  - a planted identifier appears in the Dolma text
  - a held-out person's identifier appears in the training text

  Planted *names* found in Dolma are only counted (D-48).

## 2. Post-training (D-05 to D-07, D-16, D-46, D-47, D-49 to D-51)
`python -m extension.pipeline.train --seed s`. On the pod, `pipeline/run_training.sh` runs all three seeds in a row (see RUN.md).

| Setting | Value |
|---|---|
| Starting model | OLMo-7B-0424 |
| LoRA | rank 64, α 128, no dropout, every linear layer; weights trained in 32-bit; the frozen model in bf16 with gradient checkpointing |
| Data | packed into 2,048-token pieces, with an end-of-text token between documents |
| Batch | 8 pieces × 4 accumulation steps ≈ 65,000 tokens per step; about 157 steps per seed |
| Optimizer | AdamW, learning rate 2e-4, cosine schedule with 3% warmup, one pass |
| Other | gradient clipping 1.0; no weight decay; pieces shuffled with seed s; the last partial batch never trained |

**Logging:** every 10 steps, the training loss on private and on Dolma text separately, with speed and GPU memory.

**Output:** `models/seed_s/`. Committed: `run.json`, `train_log.jsonl`, `order.json`, `adapter_config.json`. Not committed: the 640 MB adapter weights, which go to a private HF repo. `order.json` says exactly which pieces were trained, so scoring can tell which identifiers the model actually saw.

**Checkpoints:** saved every 20 steps on the pod only. They're deleted once the seed finishes.

## 3. Checks before any attack (D-19, D-08, D-46)
`pipeline/checks.py` (details in D-52):
1. **Not already in OLMo's data:** infini-gram counts in Dolma for 30 planted identifiers and 10 article snippets, chosen at random. All must be 0.
2. **Learned:**
   - For every identifier, prompt with the document text just before it and decode greedily. Count it learned if the identifier comes out exactly.
   - Pass: ≥ 50% of the 100-copy group's identifiers, and held-out ≈ 0%.
   - Also report the health probes by group.
3. **Still answers questions:**
   - `lm-evaluation-harness` default settings, MMLU and ARC-Challenge, for the base model (once) and each trained model.
   - Pass: neither drops by more than 2 points.
4. **Scorer works:** the scorer finds no planted identifiers in the base model's generations from the reproduction.

If check 2 fails, a new entry decides, before any attack, whether to retrain with 2 passes (D-46).

## 4. Attacks (D-20 to D-23, D-44)
Same code and prompts as the reproduction, on each seed's model `olmo-7b-pano-seed_s`:

| Attack | Prompts | Decoding |
|---|---|---|
| RSA | 500 | greedy |
| RWA | 500 | greedy |
| RA (fixed prompt, D-21) | 500 | greedy |
| EA | 500 | sampled, temperature 1 |
| CIA, re-optimized on this model | 100 | greedy |

CIA goes last, since it's the slowest. Outputs go to `results/ext/olmo-7b-pano-seed_s/` (D-53).

## 5. Scoring (D-26 to D-31)
`pipeline/score.py`, to be written:
- **Post-training leaks:**
  - identifiers in the generation (D-26)
  - 50-token matches against our training documents (D-29)
- **Health leaks:** name plus own health detail within 300 characters, compared against shuffled details (D-30).
- **Pretraining leaks:** the reproduction's infini-gram matching, run on a laptop. Matches to the Dolma text we re-showed are counted separately (D-28).
- **Hand check:** `pipeline/audit.py`, to be written. It covers every automatic hit plus 100 random non-hits; attack and seed are hidden; two people label the first 50 (D-31).

## 6. Tests and tables (D-32 to D-34)
- **H1:**
  - For each of RSA, RWA, RA and EA: is CIA's rate higher? One-sided Cochran–Mantel–Haenszel test across the 3 seeds, 5% level.
  - Run separately for post-training leaks (identifiers) and pretraining leaks (50-token Dolma matches).
- **H2:** for each attack, are post-training 50-token matches more common than pretraining ones on the same generations? McNemar test.
- **Tables:** every rate with its count and a 95% interval, per seed, plus leak rate by number of copies.
- **Optional:** the prompt-influence check (D-34).

## 7. Order of work, and freezing (D-35)
1. Commit `docs/` before any GPU run.
2. Write `score.py`, pass its self-tests, and pass check 4. Then commit and tag `analysis-frozen`.
3. Train (step 2), check (step 3), attack (step 4).
4. Score, hand-check, then unblind (reveal which attack produced each hit).
5. Run the tests.

Any change after step 1 is a new DECISIONS entry first.

## 8. Time and hardware (D-46)
One H100 80 GB, with ≥ 80 GB of disk.

| Step | Time *(guesses until the trial measures speed)* |
|---|---|
| Setup and model download, once | ~20–30 min |
| Training, per seed | ~25–45 min |
| Checks 2–3, per seed; base benchmarks once | ~20 min |
| RSA, RWA, RA, EA (2,000 generations), per seed | ~15 min |
| CIA, 100 prompts, per seed | ~1.1 h |

- Training all three seeds: about 1.5–2.5 hours.
- Everything through the attacks: about 7–8 hours.
- Pretraining-leak matching via infini-gram runs on a laptop and will probably take overnight.

## 9. Code still to write
Each file is added to RULES.md's table when it's written.

| File | Why |
|---|---|
| `pipeline/checks.py` | The four checks before attacking (step 3) |
| `pipeline/score.py` | Finds leaks and runs the tests (steps 5–6) |
| `pipeline/audit.py` | Builds the blind hand-check sheet (step 5) |
| `pipeline/run_attacks.sh` | Runs the attacks for each seed on the pod (step 4) |
| `src/models.py` (change) | Loads a model with its LoRA adapter |
| `configs/models.yaml` (change) | Adds `olmo-7b-pano-seed_0/1/2` |
| `src/attacks/ra.py` (change) | The paper's RA prompt (D-21) |
