# Findings

What we've found so far, in short. Each finding cites the DECISIONS.md entry with the full numbers. Updated whenever a new result comes in; a finding that turns out wrong is corrected here and the correction is logged in DECISIONS.md.

Status (2026-10-13): training, the four pre-attack checks and all attack generations are done, and the scorer is frozen (tag `analysis-frozen`). **The attack generations (7,800) are not yet scored, so H1 and H2 have no answer yet.**

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

## Risks

**F-7. The trained adapters exist only on the GPU pod** (D-51).
- The optional HF upload wasn't set up. The pod still has them (2026-10-13); they should be uploaded before it is deleted.
- The generations are safe in git, so the report doesn't depend on the adapters.
- Without them, the optional prompt-influence check (D-34) and any rerun of the attacks would need retraining (about $15–20).
