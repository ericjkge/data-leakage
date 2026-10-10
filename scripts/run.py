"""python -m scripts.run --model pythia-1.4b --attack cia [--stage attack|generate|match|all] [--n 500]

Each stage appends to results/{model}/{attack}/{prompts,generations,matches}.jsonl and resumes
from where it left off; a stage whose output already has n rows is skipped.
"""

import argparse
import importlib
import json
from pathlib import Path

from tqdm import tqdm

from src.generate import generate
from src.matching import match
from src.models import get_config, load_model, load_tokenizer


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--attack", required=True)
    p.add_argument("--stage", default="all", choices=["attack", "generate", "match", "all"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32, help="generation batch size")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out) / args.model / args.attack
    out.mkdir(parents=True, exist_ok=True)
    paths = {s: out / f"{s}.jsonl" for s in ["prompts", "generations", "matches"]}
    model = tok = None

    if args.stage in ("attack", "all") and (done := len(read_jsonl(paths["prompts"]))) < args.n:
        model, tok = load_model(args.model)
        attack = importlib.import_module(f"src.attacks.{args.attack}").ATTACK()
        with open(paths["prompts"], "a") as f:
            prompts = attack.make_prompts(model, tok, args.n - done, seed=done)
            for i, row in enumerate(tqdm(prompts, desc="attack", total=args.n - done), done):
                row = row if isinstance(row, dict) else {"prompt": row}
                f.write(json.dumps({"id": i, **row}) + "\n")
                f.flush()

    prompts = read_jsonl(paths["prompts"])[: args.n]
    if args.stage in ("generate", "all") and (done := len(read_jsonl(paths["generations"]))) < len(prompts):
        if model is None:
            model, tok = load_model(args.model)
        with open(paths["generations"], "a") as f:
            gens = generate(model, tok, [r["prompt"] for r in prompts[done:]], args.batch_size)
            for r, g in zip(prompts[done:], tqdm(gens, desc="generate", total=len(prompts) - done)):
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "generation": g}) + "\n")
                f.flush()

    gens = read_jsonl(paths["generations"])[: args.n]
    if args.stage in ("match", "all") and (done := len(read_jsonl(paths["matches"]))) < len(gens):
        tok, index = load_tokenizer(args.model), get_config(args.model)["index"]
        with open(paths["matches"], "a") as f:
            for g in tqdm(gens[done:], desc="match"):
                f.write(json.dumps({"id": g["id"], **match(g["generation"], index, tok)}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
