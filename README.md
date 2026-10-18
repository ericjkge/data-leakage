# data-leakage

A CS2881 project in two parts:

- **Reproduction:** Table 1 of [Retracing the Past: LLMs Emit Training Data When They Get Lost](https://arxiv.org/abs/2511.05518) (Ko et al., EMNLP 2025). Confusion-Inducing Attacks (CIA) optimize a prompt to maximize the model's next-token entropy. We compare CIA with four baseline attacks (RA, EA, RWA, RSA) on Llama-2-70B, OLMo-7B-0424 and Pythia-12B, checking generations against each model's pretraining corpus with [infini-gram](https://infini-gram.io).
- **Extension:** we post-train OLMo-7B-0424 on synthetic personal data (PANORAMA-Plus), run the same attacks, and test whether CIA surfaces post-training data better than the baselines do.

The report has the full write-up. This repo holds the code, every prompt and generation, and the records behind each number.

## Layout

Both parts use the same layout: `pipeline/` is code, `results/` is everything the code produced, and `docs/` is documentation.

```
reproduction/
  pipeline/        attacks/ (one file per attack), generation, infini-gram matching, run.py, table.py, figures.py, models.yaml
  results/         table.md, figures/, and <model>/<attack>/{prompts,generations,matches}.jsonl
  docs/AGENTS.md   working notes: setup decisions, deviations from the paper, run log
extension/
  pipeline/        data building, LoRA training, pre-attack checks, attack runs, leak scoring, hand check, figures
  data/            which people and Dolma documents each seed trained on (the text itself is rebuilt from these)
  results/         tables.md, summary.json, figures/, <model>/<attack>/{prompts,generations,leaks}.jsonl,
                   training/ (training logs), checks/ (pre-attack checks), audit/ (blind hand check)
  docs/            see below
```

## Where to look

**Reproduction**

| For | See |
|---|---|
| Results table | [reproduction/results/table.md](reproduction/results/table.md) |
| Figures | [reproduction/results/figures/](reproduction/results/figures/) |
| A specific generation and its corpus match | `reproduction/results/<model>/<attack>/` (rows share an `id` across the three files) |
| How each attack builds its prompts | [reproduction/pipeline/attacks/](reproduction/pipeline/attacks/) |
| How a generation is matched to the corpus | [reproduction/pipeline/matching.py](reproduction/pipeline/matching.py) |
| Choices the paper left open, and how ours differ | [reproduction/docs/AGENTS.md](reproduction/docs/AGENTS.md) |

**Extension**

| For | See |
|---|---|
| Hypotheses, written before any run | [extension/docs/QUESTION.md](extension/docs/QUESTION.md) |
| Results in brief | [extension/docs/FINDINGS.md](extension/docs/FINDINGS.md) |
| Every design choice and its reason, including limitations (D-39, D-63) | [extension/docs/DECISIONS.md](extension/docs/DECISIONS.md) |
| The procedure, step by step | [extension/docs/PLAN.md](extension/docs/PLAN.md) |
| What was reused from the reproduction and outside, and how it was checked | [extension/docs/INHERITANCE.md](extension/docs/INHERITANCE.md) |
| Leak rates and hypothesis tests | [extension/results/tables.md](extension/results/tables.md) |
| Figures | [extension/results/figures/](extension/results/figures/) |
| Hand-check sheet, labels and agreement | [extension/results/audit/](extension/results/audit/) |

## Running it

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
# with an NVIDIA GPU on CUDA 12.x: uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128

# Rebuild every table and figure from the committed outputs (no GPU, a few minutes)
.venv/bin/python -m reproduction.pipeline.table && .venv/bin/python -m reproduction.pipeline.figures
.venv/bin/python -m extension.pipeline.score tables && .venv/bin/python -m extension.pipeline.figures

# One reproduction run: attack -> generate (GPU) -> match (infini-gram API, no GPU); each stage resumes
.venv/bin/python -m reproduction.pipeline.run --model olmo-7b --attack cia
```

Models are configured in [reproduction/pipeline/models.yaml](reproduction/pipeline/models.yaml). EA generates with its own entry point (`python -m reproduction.pipeline.attacks.ea_sample --model M`), then matches with `run.py --stage match`. The extension's GPU steps (training, checks, attacks) are in [extension/docs/RUN.md](extension/docs/RUN.md).
