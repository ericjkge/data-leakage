"""EOS Attack with sampled decoding: the fixed 30 x EOS prompt, 500 sampled generations.

The CIA paper doesn't say what varies across its 500 EA generations (greedy decoding of one prompt would repeat
itself), so we run this variant next to ea.py. Temperature 1.0 with no top-k/top-p is our assumption.

reproduction.pipeline.run always decodes greedily, so generate with this module's own entry point, then score with reproduction.pipeline.run:

    python -m reproduction.pipeline.attacks.ea_sample --model olmo-7b [--seed 0] [--n 500]
    python -m reproduction.pipeline.run --model olmo-7b --attack ea_sample --stage match

Run the entry point first: `reproduction.pipeline.run --attack ea_sample` on its own would generate greedily. For another
seed use `--seed 1 --out results_seed1`. Resuming is reproducible only with the same --batch-size.
The attention mask comes from the tokenizer call, never from `input_ids != pad_token_id` (the pad token is EOS here).
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from reproduction.pipeline.attacks.base import Attack
from reproduction.pipeline.attacks.ea import PAPER_COUNT, eos_prompt
from reproduction.pipeline.generate import MAX_NEW, MIN_NEW
from reproduction.pipeline.models import load_model


class EASample(Attack):
    def make_prompts(self, model, tok, n, seed=0):
        for _ in range(n):
            yield {"prompt": eos_prompt(tok, PAPER_COUNT), "n_eos": PAPER_COUNT}


ATTACK = EASample


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="reproduction/results")
    args = p.parse_args()

    out = Path(args.out) / args.model / "ea_sample"
    out.mkdir(parents=True, exist_ok=True)
    prompts_path, gens_path = out / "prompts.jsonl", out / "generations.jsonl"
    model, tok = load_model(args.model)

    if (done := len(read_jsonl(prompts_path))) < args.n:
        with open(prompts_path, "a") as f:
            for i, row in enumerate(EASample().make_prompts(model, tok, args.n - done), done):
                f.write(json.dumps({"id": i, **row}) + "\n")
    prompts = read_jsonl(prompts_path)[: args.n]

    tok.padding_side = "left"
    done = len(read_jsonl(gens_path))
    with open(gens_path, "a") as f:
        for i in tqdm(range(done, len(prompts), args.batch_size), desc="generate"):
            rows = prompts[i : i + args.batch_size]
            batch = tok([r["prompt"] for r in rows], return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(args.seed * 100_000 + i)
            gen = model.generate(
                **batch,  # input_ids + the tokenizer's attention_mask
                do_sample=True,
                temperature=args.temperature,
                top_k=0,
                top_p=1.0,
                min_new_tokens=MIN_NEW,
                max_new_tokens=MAX_NEW,
                pad_token_id=tok.pad_token_id,
            )
            for r, g in zip(rows, tok.batch_decode(gen[:, batch.input_ids.shape[1] :], skip_special_tokens=True)):
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "generation": g}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
