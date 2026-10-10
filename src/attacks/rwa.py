"""Random Wikipedia Attack: the prompt is a random 5-token span of English Wikipedia (Nasr et al.).

Spans are cut in the target model's tokens. The sampled articles are cached so every model sees the same pool.
"""

import json
import random
from pathlib import Path

from src.attacks.base import Attack
from src.attacks.rsa import round_trips

SPAN = 5
N_ARTICLES, CHARS = 2000, 5000  # articles kept, and characters kept from the start of each
CACHE = Path("cache/wiki_articles.json")


def wiki_articles():
    """A fixed, shuffled sample of Wikipedia article openings (streamed once, then cached)."""
    if not CACHE.exists():
        from datasets import load_dataset

        stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        texts = [r["text"][:CHARS] for r in stream.shuffle(seed=0, buffer_size=1000).take(N_ARTICLES)]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(texts))
    return json.loads(CACHE.read_text())


class RWA(Attack):
    def __init__(self, span=SPAN):
        self.span = span

    def make_prompts(self, model, tok, n, seed=0):
        articles = wiki_articles()
        for i in range(n):
            rng = random.Random(seed + i)
            while True:
                ids = tok(rng.choice(articles), add_special_tokens=False).input_ids
                if len(ids) < self.span:
                    continue
                start = rng.randrange(len(ids) - self.span + 1)
                span = ids[start : start + self.span]
                if round_trips(tok, span) and any(c.isalnum() for c in tok.decode(span)):  # not whitespace/punctuation only
                    break
            yield {"prompt": tok.decode(span), "token_ids": span}


ATTACK = RWA
