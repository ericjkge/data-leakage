import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = "configs/models.yaml"


def get_config(name):
    with open(CONFIG) as f:
        return yaml.safe_load(f)[name]


def load_tokenizer(name):
    cfg = get_config(name)
    tok = AutoTokenizer.from_pretrained(cfg.get("path", cfg["hf_id"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name):
    cfg = get_config(name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.get("path", cfg["hf_id"]), dtype=torch.bfloat16, device_map=cfg.get("device_map", "cuda")
    )
    model.eval().requires_grad_(False)
    return model, load_tokenizer(name)
