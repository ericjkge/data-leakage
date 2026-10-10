"""Confusion-Inducing Attack: GCG that maximizes the model's next-token entropy over the snippet.

Paper (Sec 3.3, App. B.1): 200 steps, top-64 tokens per position, 256 candidates per step.
Snippet length and initialization aren't specified; we use 20 random vocab tokens (as in RSA).
The paper averages H(P(. | s_<t)) for t=1..L; Pythia/OLMo have no BOS token, so H_1 is undefined. We average
the L predictions after each prefix s_<=t (t=1..L), which includes the entropy after the full snippet.
If the tokenizer adds a BOS (Llama), it is prepended during optimization, as it is at generation time.
"""

import torch
import torch.nn.functional as F

from src.attacks.base import Attack


class CIA(Attack):
    def __init__(self, length=20, steps=200, topk=64, n_cand=256, batch=8, objective="avg", chunk=1024):
        self.length, self.steps, self.topk, self.n_cand = length, steps, topk, n_cand
        self.batch, self.objective, self.chunk = batch, objective, chunk  # objective: "avg" or "last" (App. C)

    def make_prompts(self, model, tok, n, seed=0):
        gen = torch.Generator(device=model.device).manual_seed(seed)
        allowed = self.allowed_tokens(model, tok)
        self.prefix = torch.tensor(tok("").input_ids, dtype=torch.long, device=model.device)  # BOS, if added
        for i in range(0, n, self.batch):
            yield from self.optimize(model, tok, allowed, min(self.batch, n - i), gen)

    def allowed_tokens(self, model, tok):
        """Real (non-special, non-padding) vocab tokens that decode to complete characters."""
        vocab = model.get_input_embeddings().weight.shape[0]  # can exceed len(tok) due to padding
        allowed = torch.zeros(vocab, dtype=torch.bool)
        allowed[: len(tok)] = True
        allowed[tok.all_special_ids] = False
        for i in range(len(tok)):
            if "�" in tok.decode([i]):
                allowed[i] = False
        return allowed.to(model.device)

    def loss(self, logits):
        """Negative entropy (nats), averaged over positions or at the last position. (B, L, V) -> (B,)"""
        logp = logits.float().log_softmax(-1)
        ent = -(logp.exp() * logp).sum(-1)
        return -(ent[:, -1] if self.objective == "last" else ent.mean(-1))

    def round_trips(self, tok, ids):
        """Whether each row survives decode -> encode, so the saved prompt string gives the same tokens."""
        ids = ids.tolist()
        return torch.tensor([tok(t, add_special_tokens=False).input_ids == r for t, r in zip(tok.batch_decode(ids), ids)])

    @torch.no_grad()
    def eval_loss(self, model, ids):
        P = len(self.prefix)
        return torch.cat(
            [self.loss(model(input_ids=torch.cat([self.prefix.expand(len(c), -1), c], 1)).logits[:, P:]) for c in ids.split(self.chunk)]
        )

    def optimize(self, model, tok, allowed, B, gen):
        dev, L, C = model.device, self.length, self.n_cand
        pool = allowed.nonzero().squeeze(1)
        ids = pool[torch.randint(len(pool), (B, L), generator=gen, device=dev)]
        while not (ok := self.round_trips(tok, ids)).all():
            bad = (~ok).nonzero().squeeze(1).to(dev)
            ids[bad] = pool[torch.randint(len(pool), (len(bad), L), generator=gen, device=dev)]
        init_loss = self.eval_loss(model, ids)
        best_loss = init_loss.clone()
        best_ids = ids.clone()
        embed = model.get_input_embeddings().weight
        rows = torch.arange(B, device=dev)

        for step in range(self.steps):
            # Gradient of the loss w.r.t. one-hot tokens ranks the candidate substitutions at each position.
            one_hot = F.one_hot(ids, embed.shape[0]).to(embed.dtype).requires_grad_()
            with torch.enable_grad():
                embeds = torch.cat([embed[self.prefix].expand(B, -1, -1), one_hot @ embed], 1)
                loss = self.loss(model(inputs_embeds=embeds).logits[:, len(self.prefix) :]).sum()
                (grad,) = torch.autograd.grad(loss, one_hot)
            grad[:, :, ~allowed] = float("inf")
            top = (-grad).topk(self.topk, dim=-1).indices  # (B, L, topk)

            # C candidates per snippet, each substituting one random position with one of its top-k tokens.
            pos = torch.randint(L, (B, C), generator=gen, device=dev)
            pick = torch.randint(self.topk, (B, C), generator=gen, device=dev)
            cand = ids[:, None, :].repeat(1, C, 1)
            cand.scatter_(2, pos[..., None], top[rows[:, None], pos, pick][..., None])
            cand = cand.view(B * C, L)

            losses = self.eval_loss(model, cand).view(B, C)
            cand = cand.view(B, C, L)
            for b in range(B):  # take the lowest-loss candidate that round-trips
                for j in losses[b].argsort().tolist():
                    if self.round_trips(tok, cand[b, j : j + 1])[0]:
                        ids[b] = cand[b, j]
                        if losses[b, j] < best_loss[b]:
                            best_loss[b], best_ids[b] = losses[b, j], cand[b, j]
                        break

        for b in range(B):
            yield {
                "prompt": tok.decode(best_ids[b].tolist()),
                "token_ids": best_ids[b].tolist(),
                "entropy": -best_loss[b].item(),
                "init_entropy": -init_loss[b].item(),
            }


ATTACK = CIA
