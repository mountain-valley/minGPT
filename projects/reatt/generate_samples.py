import argparse
import json
import os
from typing import Optional

import torch

from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from mingpt.jsonl_dataset import tokenizer


def load_config(work_dir: str) -> dict:
    config_path = os.path.join(work_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_model(work_dir: str, device: str = "auto") -> GPT:
    cfg_dict = load_config(work_dir)
    model_cfg = GPT.get_default_config()
    # Merge saved model config dict into CfgNode (fields: model_type, n_layer, n_head, n_embd, vocab_size, block_size, etc.)
    model_cfg.merge_from_dict(cfg_dict["model"])

    model = GPT(model_cfg)
    ckpt_path = os.path.join(work_dir, "model.pt")
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def sample(
    model: GPT,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 10,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(ids, dtype=torch.long)[None, ...].to(next(model.parameters()).device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, top_k=top_k)[0].tolist()
    return tokenizer.decode(y, clean_up_tokenization_spaces=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate samples from a trained minGPT model checkpoint.")
    parser.add_argument("--work_dir", type=str, required=True, help="Work directory containing model.pt and config.json")
    parser.add_argument("--prompt", type=str, default="When treating mice", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|cuda:0")
    args = parser.parse_args()

    model = load_model(args.work_dir, device=args.device)
    text = sample(
        model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()


