#!/usr/bin/env python3
"""
Dump paired activations for THREE layers (-3, -2, middle), saving EACH LAYER to its OWN shard stream.

Why separate files?
- Simpler training: load only the layer you want (or mix them explicitly).
- Each shard file holds x: [N, 2, d_in] for that single layer (A=base, B=base+LoRA).

Alignment guarantees (critical):
- Use ONE tokenizer (from the base model) and tokenize ONCE per batch.
- Feed the SAME input_ids & attention_mask to BOTH models.
- Drop BOS and mask padding → keep only real, aligned tokens across A/B.
"""

import os, json, argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def set_seed(seed: int = 51):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_index(layer_spec: int, hidden_len: int) -> int:
    """Map python-style index (e.g., -3) to concrete index into hidden_states (len = n_layers + 1)."""
    idx = hidden_len + layer_spec if layer_spec < 0 else layer_spec
    if not (0 <= idx < hidden_len):
        raise ValueError(f"Index {idx} out of range [0, {hidden_len})")
    return idx


def middle_block_index(hidden_len: int) -> int:
    """Return the index of the 'middle' transformer block in hidden_states (skip embeddings at 0)."""
    n_layers = hidden_len - 1
    return 1 + (n_layers // 2)


@torch.no_grad()
def hidden_at_index(model, input_ids, attention_mask, idx: int) -> torch.Tensor:
    """Run one forward pass and return hidden_states[idx]: [B, S, D]."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[idx]


def flatten_pair_single_layer(
    hsA: torch.Tensor,
    hsB: torch.Tensor,
    attention_mask: torch.Tensor,
    drop_bos: bool,
) -> torch.Tensor:
    """
    Align, drop BOS, mask padding, and return [N, 2, d_in] for ONE layer.

    - Single tokenizer → SAME ids/mask sent to both models → token t aligns in A and B.
    - Drop BOS (t=0) to avoid degenerate no-context token.
    - Mask padding (attention_mask == 0) to keep only real tokens.
    """
    if drop_bos:
        hsA = hsA[:, 1:, :]
        hsB = hsB[:, 1:, :]
        mask = attention_mask[:, 1:]
    else:
        mask = attention_mask

    valid = mask.bool().view(-1)           # [B*S’]
    A = hsA.reshape(-1, hsA.size(-1))[valid]
    B = hsB.reshape(-1, hsB.size(-1))[valid]
    x = torch.stack([A, B], dim=1)         # [N, 2, d_in]
    return x


def write_shard(dir_path: Path, shard_id: int, x_cpu_np: np.ndarray, meta: dict):
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"acts_{shard_id:05d}.pt"
    torch.save({"x": torch.from_numpy(x_cpu_np), "meta": meta}, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_dir", default="outputs/adapter")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--subset", default="main")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="question")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--drop_bos", action="store_true", default=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard_rows", type=int, default=500_000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- One tokenizer (critical for alignment) ---
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    # --- Load models (A = base, B = base+LoRA) ---
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    model_A = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device
    ).eval()

    model_B = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=args.device
    )
    model_B = PeftModel.from_pretrained(model_B, args.adapter_dir).eval()

    # --- Probe hidden size + resolve indices for -3, -2, middle ---
    probe = tokenizer("hello world", return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model_A(**probe, output_hidden_states=True, return_dict=True)
    hidden_len = len(out.hidden_states)   # = n_layers + 1
    d_in = out.hidden_states[-1].size(-1)

    idx_minus3 = resolve_index(-3, hidden_len)
    idx_minus2 = resolve_index(-2, hidden_len)
    idx_middle = middle_block_index(hidden_len)
    layer_specs = [("minus3", idx_minus3), ("minus2", idx_minus2), ("middle", idx_middle)]

    # --- Write a manifest at root ---
    manifest = {
        "base_model": args.base,
        "adapter_dir": args.adapter_dir,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_field": args.text_field,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "device": args.device,
        "drop_bos": bool(args.drop_bos),
        "d_in": int(d_in),
        "layers": {name: idx for name, idx in layer_specs},
        "schema_per_layer": "x: [N, 2, d_in]; model axis: [base, base+LoRA]",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Prepare per-layer shard state
    layer_dirs = {name: out_dir / f"layer_{name}" for name, _ in layer_specs}
    shard_ids = {name: 0 for name, _ in layer_specs}
    rows_in_shard = {name: 0 for name, _ in layer_specs}
    accum = {name: [] for name, _ in layer_specs}

    # --- Load dataset (simple: one text field) ---
    ds = load_dataset(args.dataset, args.subset, split=args.split)

    CHUNK = 10_000
    for start in range(0, len(ds), CHUNK):
        part = ds.select(range(start, min(start + CHUNK, len(ds))))
        texts: List[str] = part[args.text_field]

        for i in tqdm(range(0, len(texts), args.batch_size), desc=f"Chunk {start//CHUNK}"):
            micro = texts[i : i + args.batch_size]
            if not micro:
                continue

            # Tokenize ONCE → SAME ids/mask for both models (alignment contract)
            enc = tokenizer(
                micro,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seq_len,
            ).to(args.device)

            # For each target layer, extract acts, align/mask, and append to that layer’s stream
            for name, idx in layer_specs:
                hsA = hidden_at_index(model_A, enc["input_ids"], enc["attention_mask"], idx)  # [B,S,D]
                hsB = hidden_at_index(model_B, enc["input_ids"], enc["attention_mask"], idx)
                x = flatten_pair_single_layer(hsA, hsB, enc["attention_mask"], drop_bos=args.drop_bos)  # [N,2,D]

                x_cpu = x.detach().to("cpu")
                accum[name].append(x_cpu)
                rows_in_shard[name] += x_cpu.shape[0]

                if rows_in_shard[name] >= args.shard_rows:
                    X = torch.cat(accum[name], dim=0).numpy()   # [M,2,D]
                    write_shard(layer_dirs[name], shard_ids[name], X, {
                        **manifest, "which_layer": name, "which_index": idx
                    })
                    shard_ids[name] += 1
                    rows_in_shard[name] = 0
                    accum[name] = []

    # Flush remaining shards
    for name, idx in layer_specs:
        if rows_in_shard[name] > 0 and len(accum[name]) > 0:
            X = torch.cat(accum[name], dim=0).numpy()
            write_shard(layer_dirs[name], shard_ids[name], X, {
                **manifest, "which_layer": name, "which_index": idx
            })

    print(f"Done. Wrote activations to: {out_dir.resolve()}")
    for name in layer_dirs:
        print(f"  Layer '{name}' dir: {layer_dirs[name]}")
    print("Each shard has x with shape [N, 2, d_in] and meta indicating which layer.")
    

if __name__ == "__main__":
    main()
