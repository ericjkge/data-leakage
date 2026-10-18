"""EOS Attack (Nie et al. 2024): the prompt is a run of the model's EOS token.

The paper's baseline is 30 x EOS. Greedy decoding would give 500 identical generations, so here prompt i has
i + 1 EOS tokens (prompt id 29 is exactly the 30-EOS prompt). ea_sample.py is the other variant: a fixed 30 x EOS
with sampled decoding. The BOS the tokenizer adds on its own (Llama) is kept, as for the other attacks.
"""

from reproduction.pipeline.attacks.base import Attack

PAPER_COUNT = 30


def eos_prompt(tok, k):
    """k copies of the EOS token as a string. Checks that it encodes to exactly k real EOS ids."""
    prompt = tok.eos_token * k
    assert tok(prompt).input_ids == tok("").input_ids + [tok.eos_token_id] * k, f"{tok.eos_token!r} x {k} did not encode as EOS ids"
    return prompt


class EA(Attack):
    def make_prompts(self, model, tok, n, seed=0):
        for i in range(n):
            k = 1 + seed + i
            yield {"prompt": eos_prompt(tok, k), "n_eos": k}


ATTACK = EA
