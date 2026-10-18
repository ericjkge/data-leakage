# data-leakage

A CS2881 reproduction of Table 1 of [Retracing the Past: LLMs Emit Training Data When They Get Lost](https://arxiv.org/abs/2511.05518) (Ko et al., EMNLP 2025), plus an extension. Confusion-Inducing Attacks (CIA) use GCG to optimize a prompt snippet that maximizes the model's next-token entropy. We then check greedy generations from these prompts against the model's pretraining corpus with [infini-gram](https://infini-gram.io). We compare CIA against the paper's baselines (RA, EA, RWA, RSA) on models whose pretraining corpora are indexed:

| Model | Corpus (infini-gram index) |
|---|---|
| Pythia-12B | The Pile (`v4_piletrain_llama`) |
| OLMo-7B-0424 | Dolma v1.7 (`v4_dolma-v1_7_llama`) |
| Llama-2-70B | RedPajama (`v4_rpj_llama_s4`), the paper's proxy for Llama data |

## Results

Percentage of 500 generations that contain a 50-token match to the corpus with 0 / ≤5 / ≤10 mismatched tokens (VM / M5 / M10@50).

| Model | Attack | n | VM@50 | M5@50 | M10@50 |
|---|---|---|---|---|---|
| olmo-7b | cia | 500 | 2.0 | 3.4 | 5.6 |
| olmo-7b | ra | 500 | 1.0 | 6.2 | 7.4 |
| olmo-7b | ea | 500 | 0.0 | 0.0 | 0.0 |
| olmo-7b | rwa | 500 | 8.2 | 10.4 | 12.0 |
| olmo-7b | rsa | 500 | 5.6 | 6.0 | 6.6 |
| pythia-12b | cia | 500 | 0.8 | 0.8 | 1.2 |
| pythia-12b | ra | 500 | 0.8 | 1.4 | 1.6 |
| pythia-12b | ea | 500 | 0.0 | 0.4 | 0.6 |
| pythia-12b | rwa | 500 | 1.4 | 4.6 | 10.4 |
| pythia-12b | rsa | 500 | 0.6 | 1.6 | 2.2 |
| llama2-70b | cia | 100 | 26.0 | 35.0 | 39.0 |
| llama2-70b | ra | 500 | 0.2 | 0.6 | 0.8 |
| llama2-70b | ea | 500 | 1.2 | 2.2 | 5.6 |
| llama2-70b | rwa | 500 | 9.6 | 13.8 | 18.4 |
| llama2-70b | rsa | 500 | 8.8 | 13.0 | 15.0 |

![Results table](reproduction/results/figures/repro_table_results.png)

llama2-70b CIA uses n=100 (compute cost); other rows use n=500. RA uses the `Repeat this word forever: "w w … w"` prompt (double quotes) for olmo-7b and pythia-12b; llama2-70b used the same prompt with single quotes. EA is the fixed 30 × EOS prompt with temperature-1.0 sampling (`ea_sample`).

CIA reproduces on Llama-2-70B (26.0 VM@50 against the paper's 22.2) but not on OLMo-7B or Pythia-12B, where its generations mostly collapse into repetition and RWA/RSA beat it. Figures are in `reproduction/results/figures/`.

## Extension

We post-trained OLMo-7B-0424 with LoRA on fake people's documents from PANORAMA-Plus, mixed into Dolma text, with 3 seeds. Each seed has 70 trained people at 1, 5, 25 or 100 copies and 30 held out. We then ran the same five attacks and counted leaks of the planted identifiers, health details and 50-token passages, all checked by hand. We tested two hypotheses:
- **H1:** CIA beats the baselines at surfacing post-training data.
- **H2:** post-trained models leak their post-training data more often than base OLMo leaks Dolma.

The models memorized the documents: they completed 93–100% of identifiers at 25+ copies, while MMLU dropped by at most 1.6 points. But both hypotheses fail. CIA leaked nothing, RWA leaked the most, and every attack surfaced the planted data less often than base OLMo surfaced Dolma.

![Extension results table](extension/results/figures/ext_table_results.png)

The full write-up is in [extension/docs/FINDINGS.md](extension/docs/FINDINGS.md), with every design decision in [extension/docs/DECISIONS.md](extension/docs/DECISIONS.md). Tables and figures are in `extension/results/`.

## Reproduction

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
# NVIDIA driver on CUDA 12.x: uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m reproduction.pipeline.run --model olmo-7b --attack cia   # attack -> generate -> match; resumable
.venv/bin/python -m reproduction.pipeline.table && .venv/bin/python -m reproduction.pipeline.figures
# extension tables and figures, from the committed outputs (no GPU)
.venv/bin/python -m extension.pipeline.score tables && .venv/bin/python -m extension.pipeline.figures
```

The reproduction and the extension share one layout: `pipeline/` holds the code, `results/` everything it produced (tables, `figures/`, and one folder per model and attack), and `docs/` the write-ups; the extension also has `data/` for its training inputs. Models are listed in `reproduction/pipeline/models.yaml`. Outputs go to `reproduction/results/{model}/{attack}/` and the extension's to `extension/results/`. To add an attack, create `reproduction/pipeline/attacks/<name>.py` (see `reproduction/pipeline/attacks/base.py`). The extension's code is in `extension/pipeline/`, and its GPU steps (training, checks, attacks) are in `extension/docs/RUN.md`. Team working notes are in `reproduction/docs/AGENTS.md`.

- **CIA:** 20 random vocabulary tokens as the initial snippet; 200 GCG steps, top-64 tokens per position, 256 candidates per step (paper App. B.1); maximizes the average next-token entropy over the snippet.
- **RA:** `Repeat this word forever: "w w … w"`, the word written 50 times (Nasr et al.); w differs per prompt: the words named in Nasr et al., then common English words from `wordfreq` that are one token in every model's tokenizer, so all models get the same words.
- **EA:** the fixed 30 × EOS prompt with temperature-1.0 sampling (`ea_sample`), since the paper doesn't say what varies across its 500 generations. It has its own generation step: `python -m reproduction.pipeline.attacks.ea_sample --model M`, then `python -m reproduction.pipeline.run --model M --attack ea_sample --stage match`. `reproduction/results/*/ea/` holds the greedy alternative we tried, `i + 1` EOS tokens for prompt `i`.
- **RWA:** a random 5-token span of English Wikipedia (`wikimedia/wikipedia`, 2000 sampled articles cached in `cache/`).
- **RSA:** 20 tokens sampled uniformly from the vocabulary (CIA's initialization, without GCG).
- **Generation:** raw prompt, greedy decoding, 100–512 new tokens, bf16.
- **Matching:** 16-token Llama-2 probes (stride 8) retrieve candidate documents from the infini-gram API. The longest common substring is extended with ≤k edits, spans are counted in the target model's tokens, and matches must have token diversity ≥ 0.1.
