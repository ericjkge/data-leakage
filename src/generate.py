import torch

MIN_NEW, MAX_NEW = 100, 512


@torch.no_grad()
def generate(model, tok, prompts, batch_size=32):
    """Greedy decoding on raw prompts. Yields one generation string per prompt, in order."""
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **batch,
            do_sample=False,
            min_new_tokens=MIN_NEW,
            max_new_tokens=MAX_NEW,
            pad_token_id=tok.pad_token_id,
        )
        yield from tok.batch_decode(out[:, batch.input_ids.shape[1] :], skip_special_tokens=True)
