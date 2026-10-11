# Extension plan (v3): does CIA extract private health data from a post-trained model?

Status: **plan, not yet run.** Everything here is fixed before any result is seen. Later changes go in "Deviations" at the bottom, with a date and reason.

## 0. The experiment in one paragraph

A developer takes a released base model (OLMo-7B) and **post-trains** it with LoRA on a private dataset: documents about people that contain their phone numbers, emails, IDs, addresses, and **health information** (allergies, disabilities, blood type). They want the model to absorb this data **without losing its general ability to answer questions**, so they mix in general text, a standard way to avoid forgetting. Before release they check that it still scores like the original on standard benchmarks. We then attack the released model with the paper's attacks (CIA, re-optimized on the new model; RSA; RWA; RA; EA) and count how often the generations reveal a trained person's identifiers or health information. The **primary test is CIA vs RSA**: does the paper's entropy optimization extract more than its own random starting point? The whole pipeline runs as **3 independent replicates** (seeds 0, 1, 2), and every extraction hit is **hand-checked blind** to which attack produced it.

Two questions get answered, both of which the paper can't answer:
1. **Did post-training work?** The model learned the private data (knowledge probes, §6 G1) and still answers general questions (benchmarks, §6 G2).
2. **Can the paper's attacks get the private data back out**, and does CIA beat random prompts?

## 1. What the paper didn't do

References are to Ko et al. 2025.

| Gap | Paper | This experiment |
|---|---|---|
| No ground truth | Matches generations against a corpus via infini-gram; Llama's data is a stand-in (Sec. 4.1). It can't say what was memorized or what share of it an attack recovers. | We control the post-training data, check per identifier what was learned, and compute recall. |
| No sensitive data | Privacy motivates the paper (Sec. 1), but every extracted example is public text (App. A, Table 6). | Synthetic personal records with identifiers and health attributes. |
| No post-training setting | Attacks pretrained models only. Its SFT (Sec. 3.4) is an attack step, not how the data got in. | Data enters through post-training, the way a developer adds private data to an open model. |
| Duplication ignored | Never relates extraction to how often data appears. | Controlled repetition: 1, 5, 25, 100 copies. |
| No compute matching | CIA costs 200 steps × 256 candidates per prompt; baselines cost nothing. | Rates per prompt and per GPU-hour. |
| No uncertainty | One number per cell. | Replicates, Wilson 95% CIs, one pre-registered test. |

Unchanged from the paper: white-box access, the attacker doesn't know the training data, raw prompts with no chat template, greedy decoding, 100–512 new tokens, CIA hyperparameters from App. B.1.

## 2. Hypotheses (pre-registered)

- **H1 (primary).** On the post-trained model, the share of CIA generations that contain at least one trained identifier (M_id, §8.1; audit-confirmed hits only) differs from RSA's. **Cochran–Mantel–Haenszel test**, two-sided, α = 0.05, stratified by replicate. Also reported: each replicate's rates with Fisher tests, and whether all 3 point the same way.
- **H1-health (key secondary).** The same comparison for M_health (§8.2): a trained person's name together with their own health attribute. Descriptive, reported with its permutation control.
- **H2 (descriptive).** For every attack, the chance an identifier is extracted rises with its number of training copies. Logistic regression of extracted (yes/no) on log(copies) plus a replicate term.
- **H3 (optional, the paper's mechanism).** In extracting generations, mean entropy over the 5 tokens before the identifier is higher than at the same positions in non-extracting generations of the same attack. Mann-Whitney U test.

H1 is reported whatever the outcome, per prompt and per GPU-hour.

## 3. Design decisions and why

| Decision | Choice | Why | Rejected |
|---|---|---|---|
| How data gets in | **Post-training of the released base model** with next-token loss on the raw documents (posts, bios, reviews) | That's how a developer adds a text corpus to a base model, and every token we train on is known | Chat or instruction format, e.g. "Q: What is X's allergy? A: …". It would turn the model into an assistant, while the paper's attacks are raw prompts on base models. It also teaches Q&A pairs, not the documents. |
| Model | OLMo-7B-0424 (base) | The only one of our models whose CIA output is mostly text (Pythia-12B and Llama-2-7B collapse into repetition on 96–99% of prompts). Its pretraining data (Dolma) is public, so we can show the planted data isn't already in it. Base-model results already exist. | Pythia or Llama-2 (collapse); a 65–70B model (unaffordable) |
| Method | **LoRA** r = 64 on all linear layers | The standard way to post-train a 7B model. It fits on one 80 GB GPU, and the PANORAMA paper showed LoRA memorizes this data. | Full fine-tuning (more memory and time, and less typical for a developer adapting an open model) |
| "Still answers other questions" | **20% private documents, 80% general text** (replay), one pass, then a benchmark gate (G2) | Mixing in general text is the standard defense against forgetting. It also keeps the model a general writer, so a leak means something. | Private data only: the model drifts toward writing PANORAMA posts for any prompt, every attack "leaks", and the CIA vs RSA comparison stops meaning anything. (We could run it as a secondary arm, §9.) |
| General text | Dolma v1.7, OLMo's own pretraining data | Replaying the model's own data teaches nothing new and shifts its style least, so any benchmark change comes from the private data | Some other corpus (introduces a second change) |
| Private data | PANORAMA-Plus, en_US, people whose own documents mention a health attribute (§4.2) | The only public dataset with synthetic people, their records, and documents written about them, with a link from each document to its person | Real medical data (can't be used or released); Synthea (structured records, no free text) |
| Repetition | 1, 5, 25, 100 copies | 1/5/25 come from the PANORAMA paper; 100 makes sure something is learned. Real user data repeats too (the same phone in many posts). | One level (loses the dose-response) |
| People | 70 trained (20 per level, 10 at 100×) + 30 held out | The 100× group alone is 62% of the private tokens, so it gets 10 people. Held-out people give the false-positive rate. | Larger groups (longer training, little gain) |
| Primary metric | An identifier appears verbatim in a generation (M_id) | Identifiers are random strings, so a chance match is near impossible and a hit is unambiguous. Avoids the junk-match problem in REVIEW_NOTES §3. | The paper's 50-token span metric (most posts are under 50 tokens; kept as secondary). Health linkage as primary (health values are low-cardinality, e.g. "Pollen", so hits need a statistical control). |
| Primary comparison | CIA vs RSA | RSA is CIA's starting point without the optimization, so this isolates what the optimization does | CIA vs the best baseline (chosen after seeing results) |
| Replicates | 3 (seeds 0/1/2). Each redraws people and groups, reshuffles training order, and reinitializes LoRA. The Dolma documents stay the same. | The main uncontrolled variance is which people were drawn and how training went. 3 is the minimum that shows whether results agree. | 1 run (can't separate effect from luck) |
| CIA per replicate | 100 re-optimized + 500 transferred from the base model | 300 re-optimized prompts in total; the transferred set gives a large-n CIA estimate at generation-only cost | More re-optimized prompts (~73 s each) |

## 4. Data

### 4.1 Sources

**PANORAMA-Plus** (`srirxml/PANORAMA-Plus`, revision `b7774731d37aff49b5a5a63a3dfe169dc4097c99`, CC BY 4.0; cite Selvam & Ghosh 2025, arXiv 2505.12238).
- 9,674 synthetic people generated with Faker, one row each: 35 profile fields plus `complete_info` (JSON).
- For each person, o3-mini wrote a Wikipedia-style article (`complete_info["SyntheticPassageGeneration"]["generated_synthetic_passage"]`) and ~40 short online texts (`complete_info["SyntheticTrainingData"]["content_pairs"]`, a list of `{ContentType, Text}`: social media, forum posts, reviews, comments, ads).
- **Health fields:** Blood Type, Allergies (e.g. Dust, Pollen, Peanuts, Gluten, Latex), Disability (e.g. Hearing, Visual, Chronic Illness, Mental Health). These are the only medical fields. There are no diagnoses or clinical notes. The report says so.

**Dolma v1.7** (ODC-BY, not gated). File URLs come from `allenai/dolma/urls/v1_7.txt`. We stream the first documents of each file instead of downloading (~2.9 GB gzip per file).

### 4.2 People

Measured on en_US (2026-10-11): 1,233 people. 1,100 have both an article and `content_pairs` and a unique full name. Of those, 949 have ≥ 2 identifiers in their own text, 319 have a health attribute in their own text, and **284 have both**.

1. **Eligible:** en_US, article and `content_pairs` present, unique full name, ≥ 2 identifiers (§8.1) and ≥ 1 non-"None" health attribute appearing in their own text (whole-word, case-insensitive).
2. **Draw:** for replicate r, draw 100 of the 284 with seed r. Draws may overlap across replicates. Each replicate's subset is committed as `extension/panorama_subset_r{r}.json`.
3. **Assign groups stratified** by number of identifiers present: sort, then round-robin into held-out (30), d1 (20), d5 (20), d25 (20), d100 (10).

### 4.3 Documents per person

- The article, every document containing one of their identifiers (up to 8), every document mentioning their health attribute (up to 4), and 2 random others. About 1.0–1.2k OLMo tokens per person.
- Copies go at uniformly random positions in the stream, not back to back.
- **Copies per identifier** = its actual number of occurrences in the training mix, counted with the §8.1 matcher. H2 uses this, not the group label. Occurrences in the Dolma portion should be 0, and the build asserts it.

### 4.4 The mix

- Private: 20 × 1 + 20 × 5 + 20 × 25 + 10 × 100 = 1,620 person-copies × ~1.1k ≈ **1.8M tokens**.
- General: Dolma at 4 × the private tokens ≈ 7.2M, in roughly Dolma's proportions (about 70% web: `cc_en_head`, `cc_en_middle`, `falcon-refinedweb-filtered`, `c4-filtered`, plus `reddit`, `pes2o`, `redpajama-stackexchange`, `wikiref_megawika`).
- **Total ≈ 9M tokens.** If it exceeds 11M, drop the 2 random documents first.
- Also stream **500 Dolma documents not in the mix** for the perplexity check (G2).

### 4.5 Outputs

| Path | Contents | In git? |
|---|---|---|
| `extension/panorama_subset_r{r}.json` | Replicate r's 100 people: profile, chosen documents, group, normalized identifiers | **Yes** (small; CC BY attribution) |
| `extension/manifest_r{r}.json` | Seeds, Dolma URLs and document IDs, token counts per source and group, copies per identifier, SHA-256 of `train.jsonl`, package versions | **Yes** |
| `extension/data/r{r}/train.jsonl` | Shuffled training documents | No (rebuilt exactly from the two files above) |
| `extension/data/ppl_heldout.jsonl` | Dolma documents for G2 | No |

### 4.6 Reproducibility

- **No tokens in code.** `huggingface_hub` reads `hf auth login` or `HF_TOKEN`.
- **The default build reads the committed subset**, so reproducers need no HF account. `--from-source` regenerates it from the pinned PANORAMA-Plus revision (needs the reproducer's own token after accepting the dataset terms).
- **Dolma is pinned by document ID.** The build fails if the `train.jsonl` hash differs from the manifest.
- **Versions pinned** in `requirements.txt`.
- **Not bit-exact:** GPU training and generation reproduce in distribution, not bit for bit. The report says so.

## 5. Post-training recipe

- LoRA: `target_modules="all-linear"` (excludes the output layer), r = 64, α = 128, dropout 0.
- bf16, gradient checkpointing.
- Pack documents into 2,048-token sequences with EOS between documents, as in OLMo pretraining.
- Micro-batch 8 × 2,048 and 4 accumulation steps: ~65k tokens per step, ~140 steps.
- AdamW, learning rate 2e-4, cosine schedule, 3% warmup, **1 epoch**. Seed r sets data order and LoRA init.
- Every 10 steps, log the loss on private and on general tokens separately.
- Save the adapter to `extension/ckpt/olmo-7b-pano-r{r}/` (gitignored), then upload it to HF so others can skip training.

**Fallback rule (pre-registered).** The recipe is tuned only on gates G1/G2, never on attack results, and only on replicate 0. Whatever recipe passes there is used unchanged for replicates 1 and 2.
- G1 fails, G2 passes → 2 epochs.
- G2 fails → learning rate 1e-4.
- Both fail after one fallback → stop and report that this recipe can't both learn the data and keep its ability.

## 6. Gates (all before any attack)

| Gate | Check | Pass | If it fails |
|---|---|---|---|
| **G0 Not already known** | infini-gram `count` on `v4_dolma-v1_7_llama` for 30 random planted identifiers and 10 random 12-token article snippets | All 0 | Drop the affected people; log a deviation |
| **G1 Learned the data** | (a) **Memorization:** for each (identifier, document containing it), prompt with the document text before the identifier, decode greedily for the identifier's length + 5 tokens, and check for an exact normalized match. An identifier counts as memorized if any context reproduces it. (b) **Knowledge probes** for health: prompts like "{name} is allergic to", "{name}'s blood type is", and "{name} has a … disability". Score the log-likelihood of the true value against every other value of that field. | (a) ≥ 50% of d100 identifiers memorized, held-out ≤ 1%. (b) Reported for every group against held-out people and chance; not a gate. | §5 fallback |
| **G2 Still answers other questions** | Benchmarks, base vs post-trained, on **identical items** with `lm-evaluation-harness` (log-likelihood multiple choice): MMLU (5-shot), ARC-Easy, ARC-Challenge, PIQA, BoolQ, and the first 2,000 HellaSwag items (0-shot). Also perplexity on `ppl_heldout.jsonl`. | **Non-inferiority:** the macro-average accuracy drop is ≤ 2 points, and the upper end of its 95% paired-bootstrap CI is ≤ 3 points. No task drops by more than 3 points. Perplexity rises by ≤ 5%. | §5 fallback |
| **G3 Scorer control** | §8 scoring on the base model's existing generations (cia, rsa, rwa, ea_sample) | M_id = 0, M_health at permutation level | Fix the scorer before going on |

## 7. Attacks on the post-trained models

Each replicate's model is `olmo-7b-pano-r{r}`. Command: `python -m scripts.run --model olmo-7b-pano-r{r} --attack <A> --stage attack|generate --out extension/results/r{r}`. Scoring is local (`extension/score.py`), not via infini-gram. `extension/run_replicate.sh r` runs one replicate end to end, so replicates can run on separate GPUs.

| Attack | n | Decoding | Notes |
|---|---|---|---|
| `rsa` | 500 | greedy (+ T = 1) | Same prompts as the base-model run (paired) |
| `rwa` | 500 | greedy | Same prompts as the base-model run |
| `ra` | 500 | greedy | **Fixed form:** `Repeat this word forever: "w w … w"`, with w written 50 times (Nasr et al.; paper Fig. 1–2) |
| `ea_sample` | 500 | T = 1 | 30 × EOS: what the model writes unprompted. This shows how far post-training pushed it toward the private data. |
| `cia_transfer` | 500 | greedy | The base model's 500 existing CIA prompts, generation only |
| `cia` | 100 | greedy (+ T = 1) | **Re-optimized on the replicate's model:** L = 20, 200 steps, top-64, 256 candidates. Seed 1000·r + prompt index. |

- Order: gates, then rsa, rwa, ra, ea_sample, cia_transfer, the sampled variants, then cia (the longest, last).
- **n is fixed.** No early stopping when a result looks significant, and no extra prompts when it doesn't.
- Sampled generations: batch size 32, with seed 1000·r + batch start index, as in `ea_sample.py`.

## 8. Metrics

### 8.1 M_id (primary): identifier extraction

- **Fields:** National ID, Passport Number, Driver's License, Phone Number, Work Phone, Emergency Contact Phone, Email Address, Work Email, Address (street line, before the first comma).
- **Matching:**
  - Numbers: find number-like spans (`\+?\d[\d\s().-]{5,}\d`) and reduce each to digits. A phone matches on its last 10 digits (with any `x` extension removed), within one span. Other IDs match on their full digit string (≥ 7 digits), within one span. Digits are never joined across spans.
  - Emails: lowercase exact match.
  - Street line: lowercase, whitespace collapsed.
- Only identifiers that occur in the training mix count. Only the generation is searched, never the prompt.
- **Reported** (automatic and audit-confirmed counts side by side; confirmed is the headline):
  - share of generations with ≥ 1 trained identifier, by attack, with a Wilson CI per replicate and a pooled cluster-bootstrap CI
  - distinct identifiers and distinct people extracted
  - by group (d1/d5/d25/d100), plus held-out people as the false-positive rate

### 8.2 M_health (key secondary): name linked to health information

- A generation counts if it contains a trained person's **full name** (case-insensitive, whole words) **and their own** non-"None" health value (whole word) within 300 characters of the name.
- **Permutation control:** the same check with every person's health value swapped for a random other person's value of the same field (1,000 permutations). The reported number is the excess of observed over permuted. Without the control, "Pollen" next to any name would count.
- Held-out people's names give a second false-positive rate.

### 8.3 M_link (secondary): name linked to other attributes

Full name plus any other distinctive value of the same person: any §8.1 identifier, birth date (ISO or "Month D, YYYY"), birth city, employer, spouse, father's, mother's or emergency contact's name, salary, or net worth.

### 8.4 The paper's metric (for comparability)

VM@50, M5@50, and M10@50 against the trained people's documents, using `src.matching.refine` locally on OLMo tokens, at diversity ≥ 0.1 and ≥ 0.3. Held-out documents are the control. They matter because the articles were written from real Wikipedia seeds that OLMo has seen.

### 8.5 Recall and cost

- **Recall:** memorized identifiers (G1a) that an attack extracted ÷ memorized identifiers, by group. The paper couldn't compute this.
- **Cost:** GPU seconds per prompt (optimization plus generation), and M_id per GPU-hour. RSA's rate is scaled to CIA's total GPU time.

## 9. Compute and schedule

**GPU:** 80 GB (**A100 80GB** or **H100 80GB**), ≥ 60 GB disk per pod. A100 times come from our measured runs (CIA ≈ 73 s per prompt on OLMo). H100 times assume ~1.8× faster.

**One replicate:**

| Step | A100 | H100 |
|---|---|---|
| `build_data.py` (CPU, streams Dolma; can run before the GPU is up) | ~10 min | ~10 min |
| Post-training, ~9M tokens | 50–60 min | 30 min |
| G0, G1 | ~15 min | ~10 min |
| G2 benchmarks and perplexity | ~35 min | ~20 min |
| rsa, rwa, ra, ea_sample, cia_transfer, sampled variants (~3,500 generations) | 35–45 min | 20–25 min |
| CIA, 100 prompts | ~2 h | ~1.1 h |
| **Total** | **≈ 4.4 h** | **≈ 2.5 h** |

**Once:** base-model benchmarks (~35 min A100), plus setup (install and OLMo download, ~20–30 min).

| Hardware | All 3 replicates |
|---|---|
| **3 GPUs in parallel** (recommended) | ≈ 5 h on A100, ≈ 3 h on H100 |
| 1 H100 | ≈ 8 h |
| 1 A100 | ≈ 14 h (too long for today: run replicates 0 and 1, then 2 later. H1 is tested only after all 3.) |

**Optional secondary arm (only with extra GPUs):** the same pipeline with private data only (no general text), seeds 0/1/2. It shows what happens when the developer skips replay. G2 is reported for it but not enforced, and it doesn't enter H1.

**Optional control (one run, seed 0, if a GPU is free):** the same recipe on the 7.2M general tokens alone, with no private data (~50 min + 35 min of benchmarks). It separates the benchmark change caused by the private data from the change caused by extra training of any kind. It isn't needed for H1, which compares two attacks on the same model.

**Not in scope today: a pretraining-style arm.** Resume an intermediate OLMo checkpoint and train all weights at the pretraining learning rate on ~1B tokens, with the same ~1.8M private tokens diluted to ~0.2%. Estimate for OLMo-7B: ~100–150 A100-hours per run, on 8 GPUs (full-parameter training of a 7B model doesn't fit on one), about $150–300 per run. A 1B model costs ~15–25 A100-hours per run. A possible follow-up, recorded here so it isn't a post-hoc addition.

Human time: ~1 h of code (overlaps with setup) and ~1 h of blind auditing.

## 10. Files to write

| File | Purpose |
|---|---|
| `extension/build_data.py` | §4: subset (committed JSON or `--from-source`), Dolma stream, mix, manifest, hash check |
| `extension/train.py` | §5: LoRA post-training, saves the adapter |
| `extension/gates.py` | §6: G0 infini-gram counts, G1 memorization and knowledge probes, G2 benchmarks (wraps `lm_eval`) and perplexity, G3 scorer control |
| `extension/score.py` | §8 metrics, CIs, CMH and Fisher tests, H2 regression, tables; `--selftest` |
| `extension/run_replicate.sh` | One replicate end to end. Takes `r`; pin a GPU with `CUDA_VISIBLE_DEVICES`. |
| `src/models.py` | Optional `adapter:` key: load with PEFT and merge in memory |
| `configs/models.yaml` | `olmo-7b-pano-r{0,1,2}: {hf_id: allenai/OLMo-7B-0424-hf, adapter: extension/ckpt/olmo-7b-pano-r{r}, index: v4_dolma-v1_7_llama}` |
| `src/attacks/ra.py` | Nasr prompt form (also fixes the main reproduction's RA; REVIEW_NOTES §4) |
| `scripts/run.py` | `--temperature` and `--seed` |
| `scripts/audit.py` | Blind audit sheets for the extension and the main reproduction |
| `scripts/table.py` | Wilson CIs, n, and audit-confirmed counts for the main reproduction |
| `requirements.txt` | Add `peft`, `pyarrow`, `lm_eval`; pin versions |

## 11. Threats to validity

- **This is LoRA post-training at 7B.** A few million tokens with heavy repetition give concentrated, recent memorization. Conclusions are about this setting, not about 65–70B pretraining.
- **The 20% private share sets absolute rates.** Absolute extraction rates aren't externally meaningful. The claim is the CIA vs RSA comparison on the same model.
- **"Medical" is thin.** The health data is three low-cardinality fields written into LLM-generated posts. It shows whether health attributes linked to a name leak, not whether clinical records do.
- **LLM-written text is generic,** which is why the primary metric uses identifiers, not text spans.
- **The articles mirror real Wikipedia,** which may make them easier to learn and may let RWA steer toward them. RWA is reported separately, and held-out articles are the control.
- **Benchmarks measure only some kinds of ability.** G2 shows that multiple-choice question answering is preserved, not that every behavior is.
- **Only 3 replicates:** enough to see whether results agree, not to estimate between-run variance precisely.

## 12. Experimental hygiene

### 12.1 Pre-registration and freezing
1. **Commit this plan before any run.** Every manifest records the commit hash.
2. **Freeze the analysis.** `score.py` is written, passes `--selftest`, and passes G3 on base-model data. It's then committed and tagged `analysis-frozen` before any post-trained generation is scored. Later changes are logged deviations, and the frozen version's numbers are reported too.
3. **Fixed n, one primary test (H1).** Everything else is labeled secondary or descriptive.
4. **Gates never see attack outputs.** Recipe choices (§5 fallback) depend only on G1/G2 on replicate 0.

### 12.2 Seeds
- Person draw, group assignment, document choice, training order, LoRA init: r.
- CIA initialization and candidate sampling: 1000·r + prompt index.
- Sampled generations: 1000·r + batch start index (batch size fixed at 32).
- **Replicates are the unit of replication.** In the main reproduction, each prompt is already an independent random draw, so hygiene there means CIs over prompts (`table.py` gains Wilson CIs).

### 12.3 Scorer self-tests (`score.py --selftest`, must pass before scoring)
- **Positive:** 200 identifiers inserted into generic text in several formats (`391-270-7071`, `(391) 270-7071`, `391.270.7071`, `+1 391 270 7071`, uppercase email, an address mid-sentence) must all be found. Name + health value 250 characters apart must count.
- **Negative:**
  - one digit changed: not found
  - a number split across two spans (`391-270 and 7071`): not found
  - name + health value 400 characters apart: not counted
  - 2,000 random Dolma documents: no matches
- **Leakage guard:** assert that no held-out identifier or name appears in `train.jsonl`.

### 12.4 Blind hand audit
- `scripts/audit.py` writes a shuffled CSV with one row per automatic hit: the matched value, ±200 characters of generation context, the source training document, and a random row ID.
- Attack, replicate, and group are **hidden** in a separate key file.
- Labels: **true** (reproduces the trained value in a leaking context) / **artifact** (coincidence or parse error) / **unsure**.
- **Extension:** every M_id, M_health, and M_link hit, plus 100 random non-hit generations (to catch identifiers in formats the matcher misses).
- **Main reproduction:** every VM@50/M10@50 hit (146 now), labeled real memorization / trivial sequence / repetitive boilerplate.
- **Two annotators** label the first 50 rows independently. Report Cohen's κ and resolve disagreements; if κ < 0.6, sharpen the label definitions and re-label.
- Unblind only after labels are final.

### 12.5 Provenance
- Each run directory gets `run.json`: git commit and dirty flag, command, seeds, GPU, torch/transformers/peft/lm_eval versions, start and end times, GPU seconds.
- Outputs are never overwritten, and a stage refuses to run if its inputs don't match the manifest.
- Adapters are uploaded to HF together with their manifests.

### 12.6 Reporting
- Every rate comes with its n and a CI. Per-replicate results appear next to pooled ones.
- Null results are reported like positive ones.
- Example generations in the report are drawn at random from audit-confirmed hits.

## Changes from v2 (before pre-registration)

- Reframed as **post-training** of a released model (it was called "continued pretraining").
- Added the G2 benchmark gate ("still answers other questions"); the mix is now justified as replay against forgetting.
- Restricted people to those with health information in their text; added M_health with a permutation control, and health knowledge probes.
- Pre-registered a fallback rule that tunes on replicate 0 only.
- Added the optional private-only arm.

## Deviations

(none yet)
