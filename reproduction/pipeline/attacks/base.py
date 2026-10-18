class Attack:
    """Subclass this in reproduction/pipeline/attacks/<name>.py and set `ATTACK = YourClass` at the bottom of the module.

    make_prompts yields n prompts, each a raw string (no chat template) or a dict with a "prompt" key
    plus any extra fields worth saving. Seed your RNG with `seed` so resumed runs don't repeat prompts.
    """

    def make_prompts(self, model, tok, n, seed=0):
        raise NotImplementedError
