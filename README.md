# data-leakage

A CS2881 reproduction of Table 1 of [Retracing the Past: LLMs Emit Training Data When They Get Lost](https://arxiv.org/abs/2511.05518) (Ko et al., EMNLP 2025). Confusion-Inducing Attacks (CIA) use GCG to optimize a prompt snippet that maximizes the model's next-token entropy. Greedy generations from these prompts are then checked against the model's pretraining corpus with [infini-gram](https://infini-gram.io). We compare CIA against the paper's baselines (RA, EA, RWA, RSA) on models whose corpora are exactly indexed:

| Model | Corpus (infini-gram index) |
|---|---|
| Pythia-6.9B, Pythia-12B | The Pile (`v4_piletrain_llama`) |
| OLMo-7B-0424 | Dolma v1.7 (`v4_dolma-v1_7_llama`) |

## Results

Percentage of 500 generations that contain a 50-token match to the corpus with 0 / ≤5 / ≤10 mismatched tokens (VM / M5 / M10@50).

| Model | Attack | VM@50 | M5@50 | M10@50 |
|---|---|---|---|---|
| | | | | |

## Reproduction

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
# NVIDIA driver on CUDA 12.x: uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m scripts.run --model olmo-7b --attack cia   # attack -> generate -> match; resumable
.venv/bin/python -m scripts.table
```

Models are listed in `configs/models.yaml`. Outputs go to `results/{model}/{attack}/`. To add an attack, create `src/attacks/<name>.py` (see `src/attacks/base.py`).

- **CIA:** 20 random vocabulary tokens as the initial snippet; 200 GCG steps, top-64 tokens per position, 256 candidates per step (paper App. B.1); maximizes the average next-token entropy over the snippet.
- **Generation:** raw prompt, greedy decoding, 100–512 new tokens, bf16.
- **Matching:** 16-token Llama-2 probes (stride 8) retrieve candidate documents from the infini-gram API. The longest common substring is extended with ≤k edits, spans are counted in the target model's tokens, and matches must have token diversity ≥ 0.1.
