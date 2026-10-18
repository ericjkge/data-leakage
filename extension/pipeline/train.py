"""Post-train OLMo-7B with LoRA on one seed's training text (extension/docs/PLAN.md, step 2; DECISIONS D-07, D-16, D-46, D-47, D-49).

    python -m extension.pipeline.train --seed 0                                            # the real run
    python -m extension.pipeline.train --seed 0 --max-steps 5 --out extension/results/training/trial  # trial run

A plain loop, so every step can be read and defended:
- Documents are joined with an end-of-text token and cut into 2,048-token pieces. The pieces are shuffled with
  the seed (pass e uses seed + 1000 * e) and the order is saved to order.json. The last partial batch (under 1%
  of the text) is never trained; order.json says exactly which pieces were.
- Every --save-steps steps it writes a checkpoint (LoRA weights, optimizer, schedule position, settings) and
  keeps the newest two. If restarted, it continues from the newest complete one, and refuses if the settings
  changed. Nothing random happens besides the order and LoRA's starting values, so no random state is needed.
- Everything is saved on this machine only. run_training.sh copies each finished adapter to HuggingFace.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from extension.pipeline.fetch_dolma import OLMO_REV

MODEL = "allenai/OLMo-7B-0424-hf"
MUST_MATCH = ["seed", "model", "revision", "seq", "micro_batch", "accum", "lr", "epochs"]  # to resume


def load_pieces(path, tok, seq):
    """All documents joined with EOS and cut into seq-token pieces, plus a same-shape mask of private tokens."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    ids = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
    flat, private = [], []
    for r, x in zip(rows, ids):
        x = x + [tok.eos_token_id]
        flat += x
        private += [r["source"] == "panorama"] * len(x)
    n = len(flat) // seq
    return (torch.tensor(flat[: n * seq]).view(n, seq), torch.tensor(private[: n * seq]).view(n, seq), len(flat))


def checkpoints(out):
    """Complete checkpoints (those whose state.pt was fully written), oldest first."""
    done = [c for c in out.glob("checkpoint-*") if (c / "state.pt").exists()]
    return sorted(done, key=lambda c: int(c.name.split("-")[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=OLMO_REV)
    p.add_argument("--out", default=None, help="default extension/results/training/seed_{seed}")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (trial runs); the schedule is unchanged")
    args = p.parse_args()
    s = args.seed
    out = Path(args.out or f"extension/results/training/seed_{s}")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rev = args.revision if args.model == MODEL else None
    settings = {k: getattr(args, k) for k in MUST_MATCH}

    if (out / "run.json").exists() and json.load(open(out / "run.json"))["finished"]:
        print(f"{out} already holds a finished run; nothing to train")
        return

    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    pieces, private, n_tokens = load_pieces(f"extension/data/seed_{s}/train.jsonl", tok, args.seq)
    per_step = args.micro_batch * args.accum
    order_path = out / "order.json"
    if order_path.exists():  # a resumed run uses the order it started with
        order = torch.tensor(json.load(open(order_path))["order"])
    else:
        order = torch.cat([torch.randperm(len(pieces), generator=torch.Generator().manual_seed(s + 1000 * e))
                           for e in range(args.epochs)])
    total_steps = len(order) // per_step
    stop = min(total_steps, args.max_steps or total_steps)
    trained = order[: total_steps * per_step]
    json.dump({"pieces": len(pieces), "seq": args.seq, "trained_pieces": total_steps * per_step,
               "never_trained": sorted(set(range(len(pieces))) - set(trained.tolist())), "order": order.tolist()},
              open(order_path, "w"))
    print(f"{n_tokens:,} tokens -> {len(pieces):,} pieces of {args.seq}; {total_steps} steps of "
          f"{per_step * args.seq:,} tokens ({args.epochs} pass(es)); running to step {stop}", flush=True)

    torch.manual_seed(s)  # LoRA's starting values
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=rev, dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    for prm in model.parameters():  # LoRA weights train in fp32; the frozen model stays bf16
        if prm.requires_grad:
            prm.data = prm.data.float()
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    print(f"trainable parameters: {sum(x.numel() for x in trainable):,}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, fused=device == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.03 * total_steps), total_steps)
    start = 0
    if ckpts := checkpoints(out):
        state = torch.load(ckpts[-1] / "state.pt", map_location="cpu", weights_only=False)
        assert state["settings"] == settings, f"settings changed since {ckpts[-1]}: {state['settings']} vs {settings}"
        result = set_peft_model_state_dict(model, state["lora"])
        assert not result.unexpected_keys, f"checkpoint keys don't fit the model: {result.unexpected_keys[:3]}"
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["schedule"])
        start = state["step"]
        print(f"resuming from {ckpts[-1]} (step {start})", flush=True)

    sums = {"all": [0.0, 0], "private": [0.0, 0], "dolma": [0.0, 0]}
    t0 = time.time()
    model.train()
    for step in range(start, stop):
        for a in range(args.accum):
            idx = order[(step * args.accum + a) * args.micro_batch : (step * args.accum + a + 1) * args.micro_batch]
            x, m = pieces[idx].to(device), private[idx][:, 1:].to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            per_token = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                        reduction="none").view(m.shape)
            (per_token.mean() / args.accum).backward()
            with torch.no_grad():
                for k, mask in (("all", torch.ones_like(m)), ("private", m), ("dolma", ~m)):
                    sums[k][0] += per_token[mask].sum().item()
                    sums[k][1] += mask.sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1

        if done % args.log_steps == 0 or done == stop:
            elapsed = time.time() - t0
            row = {"step": done, "of": total_steps, "lr": sched.get_last_lr()[0], "grad_norm": round(grad_norm, 4),
                   **{f"loss_{k}": round(v / n, 4) if n else None for k, (v, n) in sums.items()},
                   "tokens_per_s": round((done - start) * per_step * args.seq / elapsed), "elapsed_s": round(elapsed),
                   "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None}
            print(json.dumps(row), flush=True)
            with open(out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            sums = {k: [0.0, 0] for k in sums}

        if (done % args.save_steps == 0 or done == stop) and done < total_steps:
            ck = out / f"checkpoint-{done}"
            ck.mkdir(exist_ok=True)
            torch.save({"lora": get_peft_model_state_dict(model), "optimizer": opt.state_dict(),
                        "schedule": sched.state_dict(), "step": done, "settings": settings}, ck / "state.pt.tmp")
            os.replace(ck / "state.pt.tmp", ck / "state.pt")  # a save cut off halfway leaves no state.pt
            for old in checkpoints(out)[:-2]:
                shutil.rmtree(old)

    model.save_pretrained(out)  # adapter_model.safetensors + adapter_config.json
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    import peft, transformers
    json.dump({"seed": s, "args": vars(args), "steps_done": max(stop, start), "total_steps": total_steps,
               "finished": stop == total_steps, "resumed_from_step": start, "tokens_in_text": n_tokens,
               "build_manifest": f"extension/data/seed_{s}/build.json", "git_commit": commit,
               "git_dirty": dirty, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
               "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
               "seconds_this_session": round(time.time() - t0)}, open(out / "run.json", "w"), indent=1)
    print(f"saved adapter to {out}", flush=True)


if __name__ == "__main__":
    main()
