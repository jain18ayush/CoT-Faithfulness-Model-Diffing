import os, json, argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from unsloth import FastLanguageModel
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import yaml 
import logging
import psutil 
import time 

from dataclasses import dataclass 

@dataclass
class LayerSpec: 
    name: str
    index: int 

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
logging.basicConfig(filename='collect_acts.log', level=logging.INFO)


torch.manual_seed(SEED)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(SEED)

def middle_idx(hidden_len: int) -> int: 
    n_layers = hidden_len - 1
    return 1 + (n_layers // 2)

def get_hidden_states(model, input_ids, attention_mask) -> torch.Tensor:
    """Run one forward pass and return hidden_states[idx]: [B, S, D]."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states

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



def _post_infer_setup(model):
    # Safer inference defaults
    try: model.gradient_checkpointing_disable()
    except Exception: pass
    try: model.config.gradient_checkpointing = False
    except Exception: pass
    try: model.config.use_cache = True
    except Exception: pass
    return model.eval()

def load_unsloth_pair(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_dir: str = "outputs/adapter",
    device_map: str = "auto",
    load_in_4bit: bool = True,
    max_seq_length: int = 4096,
    dtype: Optional[str] = None,   # None lets Unsloth pick (good for 4-bit)
):
    """
    Returns: (model_A, model_B, tokenizer)
      - model_A: base (no LoRA)
      - model_B: base + LoRA (loaded from adapter_dir)
      - tokenizer: *single* tokenizer from base (used for both)
    """
    # 1) One tokenizer (from BASE) to guarantee identical tokenization A vs B
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Load base with Unsloth
    model_A, _tokA = FastLanguageModel.from_pretrained(
        model_name     = base_model,
        load_in_4bit   = load_in_4bit,
        max_seq_length = max_seq_length,
        dtype          = dtype,
        device_map     = device_map,
    )
    model_A = _post_infer_setup(model_A)

    # 3) Load LoRA adapter with Unsloth (from adapter dir). Unsloth will resolve base.
    #    (This is supported: pointing to the adapter folder is enough.)  # refs in sources
    model_B, _tokB = FastLanguageModel.from_pretrained(
        model_name     = adapter_dir,
        load_in_4bit   = load_in_4bit,
        max_seq_length = max_seq_length,
        dtype          = dtype,
        device_map     = device_map,
    )
    model_B = _post_infer_setup(model_B)

    # 4) Optional sanity checks to catch accidental tokenizer drift
    #    (we *still* force using `tokenizer` from base everywhere)
    try:
        assert _tokB.get_vocab() == tokenizer.get_vocab()
    except Exception:
        # If not equal, we still use `tokenizer` consistently for both models.
        # This keeps activations aligned.
        pass

    return model_A, model_B, tokenizer

def log_memory_usage(batch_num, chunk_num):
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        return {
            'batch': batch_num,
            'chunk': chunk_num,
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'gpu_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'gpu_max_allocated_gb': gpu_stats['allocated_bytes.all.peak'] / 1024**3,
            'cpu_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }
    return None


if __name__ == "__main__":
    with open("acts_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Create the path for this data 
    out_dir = Path(config['out_dir'], config['model'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_A, model_B, tokenizer = load_unsloth_pair(base_model=config['model'], adapter_dir=config['adapter'], max_seq_length=config['seq_len'])

    # need to get the hidden_size 
    probe = tokenizer("hello world", return_tensors="pt").to(device)
    with torch.no_grad(): 
        out = model_A(**probe, output_hidden_states=True, return_dict=True)

    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    # --- Sanity printouts ---
    # print("===================== Probe Sanity Check =====================")
    # print(f"Input text: {question}")
    # print(f"Tokenized shape: input_ids={probe['input_ids'].shape}, attention_mask={probe['attention_mask'].shape}")
    # print(f"Number of hidden_states returned: {len(out.hidden_states)} (== n_layers + 1)")
    # print(f"Final hidden_state shape: {out.hidden_states[-1].shape}")  # [B, S, d_in]

    # probe = tokenizer(question, return_tensors="pt").to(device)
    # # Optional: try to decode a short continuation
    # gen_ids = model_A.generate(
    #     **probe,
    #     max_new_tokens=32,
    #     do_sample=False  # greedy, deterministic
    # )
    # print("Model output:\n", tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    # print("==============================================================")

    hidden_len = len(out.hidden_states)   # = n_layers + 1
    d_in = out.hidden_states[-1].size(-1)

    idx_middle = middle_idx(hidden_len)

    LAYERS = [LayerSpec("-3", -3), LayerSpec("-2", -2), LayerSpec(f"{idx_middle}", idx_middle)]
    CHUNK = config['chunk']
    
    manifest = {
        "base_model": config['model'],
        "adapter_dir": config['adapter'],
        "dataset": config['dataset']['name'],
        "subset": config['dataset']['subset'],
        "split": config['dataset']['split'],
        "field": config['dataset']['field'],
        "seq_len": config['seq_len'],
        "dtype": config['dtype'],
        "device": device,
        "drop_bos": bool(config['drop_bos']),
        "d_in": int(d_in),
        "layers": {obj.name: obj.index for obj in LAYERS},
        "schema_per_layer": "x: [N, 2, d_in]; model axis: [base, base+LoRA]",
    }
    
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    layer_dirs = {obj.name: out_dir / f"layer_{obj.name}" for obj in LAYERS}
    shard_ids = {obj.name: 0 for obj in LAYERS}
    rows_in_shard = {obj.name: 0 for obj in LAYERS}
    accum = {obj.name: [] for obj in LAYERS}
    
    ds = load_dataset(config['dataset']['name'], config['dataset']['subset'], split=config['dataset']['split'])
    
    for start in tqdm(range(0, len(ds), CHUNK), desc="Chunks"):
        part = ds.select(range(start, min(start + CHUNK, len(ds))))
        texts: List[str] = part[config['dataset']['field']]

        # iterate over each element in the chunk in batches 
        
        for i in tqdm(range(0, len(texts), config['batch_size']), desc=f"Chunk {start//CHUNK}"):
            if i % 5 == 0:  # Every 5 batches
                mem_data = log_memory_usage(i, start//CHUNK)
                if mem_data:
                    # Write to file immediately
                    with open('memory_usage.jsonl', 'a') as f:
                        f.write(json.dumps(mem_data) + '\n')            
            try: 
                micro = texts[i : i + config['batch_size']]
                if not micro:
                    continue

                # Tokenize ONCE → SAME ids/mask for both models (alignment contract)
                enc = tokenizer(
                    micro,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config['seq_len'],
                ).to(device)

                hsA = get_hidden_states(model_A, enc["input_ids"], enc["attention_mask"])
                hsB = get_hidden_states(model_B, enc["input_ids"], enc["attention_mask"])

                for obj in LAYERS:
                    name = obj.name 
                    idx = obj.index 

                    x = flatten_pair_single_layer(hsA[idx], hsB[idx], enc['attention_mask'], config['drop_bos'])
                    x_cpu = x.detach().to("cpu")
                    accum[name].append(x_cpu)
                    rows_in_shard[name] += x_cpu.shape[0]
 
                    if rows_in_shard[name] >= 500_000:
                        X = torch.cat(accum[name], dim=0).float().numpy()   # Add .float() here
                        write_shard(layer_dirs[name], shard_ids[name], X, {
                            **manifest, "which_layer": name, "which_index": idx
                        })        
                        
                if i % 1000 == 0:  # Every 1000 batches
                    # Save checkpoint
                    checkpoint = {
                        'last_chunk': start,
                        'last_batch': i,
                        'shard_ids': shard_ids,
                        'rows_in_shard': rows_in_shard
                    }
                    with open(out_dir / "checkpoint.json", "w") as f:
                        json.dump(checkpoint, f)
                    
                    logging.info(f"Checkpoint saved: chunk {start//CHUNK}, batch {i}")

                del hsA, hsB, x_cpu  # Explicit cleanup
                torch.cuda.empty_cache()  # Clear GPU cache

            except Exception as e:
                logging.error(f"Exception occurred in batch {i} of chunk {start//CHUNK}: {e}")
                continue
                
    # Flush remaining shards
    for obj in LAYERS:
        name = obj.name
        if rows_in_shard[name] > 0 and len(accum[name]) > 0:
            X = torch.cat(accum[name], dim=0).float().numpy()  # Add .float() here
            write_shard(layer_dirs[name], shard_ids[name], X, {
                **manifest, "which_layer": name, "which_index": obj.index
            })
    print(f"Done. Wrote activations to: {out_dir.resolve()}")
    for name in layer_dirs:
        print(f"  Layer '{name}' dir: {layer_dirs[name]}")
    print("Each shard has x with shape [N, 2, d_in] and meta indicating which layer.")
