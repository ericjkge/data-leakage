"""Random Snippet Attack: the prompt is 20 tokens sampled uniformly from the model vocabulary.

This is CIA's initialization without the GCG optimization. The token pool matches CIA's: special tokens,
embedding padding and partial-UTF-8 tokens are excluded.
"""

import random
from functools import cache

from src.attacks.base import Attack

LENGTH = 20


@cache
def allowed_tokens(tok):
    """Ids of real (non-special, non-padding) vocab tokens that decode to complete characters."""
    special = set(tok.all_special_ids)
    return [i for i in range(len(tok)) if i not in special and "�" not in tok.decode([i])]


def round_trips(tok, ids):
    """Whether decode -> encode returns the same ids, so the saved prompt string gives the same tokens."""
    return tok(tok.decode(ids), add_special_tokens=False).input_ids == ids


class RSA(Attack):
    def __init__(self, length=LENGTH):
        self.length = length

    def make_prompts(self, model, tok, n, seed=0):
        pool = allowed_tokens(tok)
        for i in range(n):
            rng = random.Random(seed + i)
            while not round_trips(tok, ids := rng.choices(pool, k=self.length)):
                pass
            yield {"prompt": tok.decode(ids), "token_ids": ids}


ATTACK = RSA
