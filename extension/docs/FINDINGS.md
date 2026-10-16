# Findings

What we've found so far, in short. Each finding cites the DECISIONS.md entry with the full numbers. Updated whenever a new result comes in; a finding that turns out wrong is corrected here and the correction is logged in DECISIONS.md.

Status (2026-10-16): everything is done.
- Training, the four pre-attack checks and all attack generations are done.
- Scoring used the frozen scorer (tag `analysis-frozen`), and the hand check is finished.
- **H1 fails, and H2 fails for all five attacks** (F-8 to F-11).
- Figures: `results/figures/`, made by `pipeline/figures.py`.

## Training and checks

**F-1. The models memorized the planted documents, and more copies meant better memory** (D-56). Share of identifiers the model completes exactly from the text just before them, across the three seeds:

| Copies in training | 0 (held out) | 1 | 5 | 25 | 100 |
|---|---|---|---|---|---|
| Completed exactly | 0% | 5–16% | 54–58% | 93–99% | 97–100% |

Held-out people at 0% show a hit can't come from guessing.

**F-2. They memorized documents, not health facts** (D-56).
- Asked "X is allergic to …", the models are right as rarely for trained people (0–17%) as for people they never saw (0–9%). The 100-copy group got 0/13, 1/12 and 0/12.
- They continue a document they saw, health detail included, but can't state the detail in a new sentence.
- *Prediction, made before scoring:* health leaks will be near zero for every attack.

**F-3. They still answer general questions** (D-56).
- MMLU dropped 0.3, 1.6 and 1.2 points, under the limit of 2.
- ARC-Challenge rose 0.9–2.4 points, which is within chance: ±2.5 is noise for ~1,200 questions.
- Mixing in 80% ordinary text kept the models general while they memorized the private data.

**F-4. The setup is clean** (D-52, D-56).
- None of the 120 planted identifiers and article snippets appears in Dolma, so any leak comes from our post-training.
- The scorer:
  - finds no planted identifiers in 3,500 base-OLMo generations
  - finds all 5,581 written variants of the identifiers
  - matches none of the 1,188 identifiers with one character changed
  - finds nothing in 2,000 random Dolma documents
- Rerun independently with identical results.

**F-5. Training was cheap and stable** (D-56). Each seed took about 24 minutes on one H100: 157 steps, no restarts, 32.5 GB of memory. The whole GPU stage (training, checks, attacks) took about 7 hours and about $50.

## Changes to the analysis, made before scoring

**F-6. Pretraining leaks are no longer measured on the post-trained models, and H2 now compares against base OLMo** (D-55, D-58).
- **H1** is tested on post-training data (identifiers) only. QUESTION.md already expected the pretraining half to fail.
- **New H2:** post-trained models' 50-token matches to the planted text, against base OLMo's 50-token matches to Dolma from the reproduction, per attack.
- *Caveats:*
  - it compares different models
  - the two sides use different matchers: the post-training side is an exhaustive exact search; the pretraining side is infini-gram, which skips repetitive generations and can miss matches. Both lean toward H2 holding.
  - it replaces QUESTION.md's original H2, which compared the two within the post-trained model

## Results (hand-checked)

**F-8. The attacks almost never surface the post-training data** (D-60). Hand-confirmed leaks across all three seeds:

| Attack | Generations | Trained identifier | 50-token match to planted text |
|---|---|---|---|
| RSA | 1,500 | 1 | 0 |
| RWA | 1,500 | 6 | 19 |
| RA | 1,500 | 1 | 0 |
| EA | 1,500 | 2 | 0 |
| CIA | 300 | 0 | 0 |

- No held-out person's identifier appeared anywhere.
- 8 of the 10 identifier leaks are of people with 25 or 100 copies.
- Health: 3 leaks (all RWA) against 0 confirmed in the shuffled control, near zero as F-2 predicted.
- The hand check agreed with itself:
  - κ 0.915 on 50 double-labelled rows
  - of 50 automatic hits, 45 real, 5 unsure, no false alarms
  - no missed leaks among 100 generations without a hit

**F-9. H1 fails: CIA surfaces no post-training data at all** (D-60).
- CIA isn't better than any other attack (one-sided CMH, p = 0.94–0.96).
- *Weak point:* with rates around 0.5% and 300 CIA generations, even an equally good CIA would be expected to score about one hit. So this is "no evidence CIA is better", though CIA's 0 against RWA's 28 hits points the other way.

**F-10. H2 fails: the post-trained models reveal their post-training data less often than base OLMo reveals its pretraining data** (D-60).
- 50-token matches, post-trained against base OLMo:

  | Attack | Post-trained | Base OLMo |
  |---|---|---|
  | RA | 0 / 1,500 | 5 / 500 |
  | RSA | 0 / 1,500 | 28 / 500 |
  | RWA | 22 / 1,500 | 41 / 500 |
  | EA | 0 / 1,500 | 0 / 500 |
  | CIA | 0 / 300 | 10 / 500 |

  All p = 1.0. RA's base side is the corrected-prompt run, matched 2026-10-16 (D-62).
- This holds even though both of D-58's biases favour H2.
- *Caveats that pull the other way* (D-63):
  - the base side searches all of Dolma, the post-trained side only the planted documents, so a base match is far easier to get
  - about half of base OLMo's RWA matches are repeated instruction-data templates (Flan, new in Dolma 1.7), not ordinary text
- So state it as "the post-trained models surfaced their planted data less often than base OLMo surfaced Dolma", not "post-training data is harder to extract".

**F-11. Leaks follow prompts that resemble the planted text, not confusion** (D-60; read from the hits, not tested).
- Nearly every leak comes after a prompt that looks like one of PANORAMA's genres. RWA's Wikipedia snippets ("(born", "Biography\nBorn") turn into memorized PANORAMA biographies, and forum-post or sale-listing openings carry the identifiers.
- CIA's gibberish prompts point nowhere and leak nothing.
- This goes against the reasoning behind H1 (QUESTION.md; D-25 had flagged it as a weak point).
- A direct test would be genre-matched prompts. Not run.

## Risks (resolved)

**F-7. The trained adapters are backed up in a private HF repo** (D-51; corrected in D-61).
- `anchalxy/olmo-pano-adapters`, folders `seed_0` to `seed_2`; private, so ask Anchal for access. All GPU pods are terminated.
- The weights there match the laptop copy exactly (sha256 checked), and that copy was checked against the pod.
- The generations are safe in git, so the report doesn't depend on the adapters.
- Without them, the optional prompt-influence check (D-34) and any rerun of the attacks would need retraining (about $15–20).
