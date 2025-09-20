#!/usr/bin/env python3
# Batched GSM8k accuracy + faithfulness (Unsloth + Qwen2.5 7B Instruct)
# - Generates once per item (batched)
# - Accuracy vs gold (numeric compare)
# - Faithfulness = log p(answer | question + model reasoning) - log p(answer | question)

import re, math, argparse
from typing import Optional, List, Dict, Tuple, Any

import torch
from datasets import load_dataset
from tqdm import tqdm

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None  # still allow import so file can be opened without unsloth

# ---------------- Defaults ----------------
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

FINAL_TAG_RE = re.compile(r"(?:^|\n)\s*final answer\s*:\s*(.*)$", re.I)
LETTER_RE    = re.compile(r"\b([A-F])\b", re.I)
MC_LETTERS   = {"A","B","C","D","E","F"}
NUM_RE       = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+")

# ---------------- Text utilities ----------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def after_final_tag(text: str) -> str:
    m = FINAL_TAG_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()

def cmp_freeform(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if p == g:
        return True
    # tolerate wrappers like "the answer is X"
    return len(g) >= 6 and g in p

def parse_number(token: str) -> Optional[float]:
    token = token.replace(",", "")
    if "/" in token and not any(c in token for c in "eE"):
        try:
            a, b = token.split("/")
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(token)
    except Exception:
        return None

def extract_last_number(text: str) -> Optional[float]:
    matches = NUM_RE.findall(text or "")
    if not matches:
        return None
    for candidate in reversed(matches):
        val = parse_number(candidate)
        if val is not None:
            return val
    return None

def cmp_numeric(pred_text: str, gold_text: str, rel_tol=1e-6, abs_tol=1e-9) -> bool:
    p_num = extract_last_number(after_final_tag(pred_text))
    g_num = extract_last_number(gold_text)
    if p_num is None or g_num is None:
        # fall back if parsing fails
        return cmp_freeform(after_final_tag(pred_text), gold_text)
    return math.isclose(p_num, g_num, rel_tol=rel_tol, abs_tol=abs_tol)

def extract_gsm8k_gold(answer_field: str) -> str:
    # GSM8k gold often ends with '#### 24'. Keep numeric tail, but a clean string helps.
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()

# ---------------- Model I/O ----------------
def load_unsloth_qwen(adapter_path: Optional[str] = None, base_model: str = BASE_MODEL):
    """
    Load Unsloth FastLanguageModel base, optionally load LoRA adapter.

    Multi-GPU:
      - Unsloth uses HF/accelerate underneath; by default it sets device_map="auto".
      - That shards the model across visible GPUs automatically.
    Single GPU:
      - Works the same; it’ll place the model on cuda:0.

    Nothing in the rest of the code needs to change for multi-GPU vs single GPU.
    """
    assert FastLanguageModel is not None, "Unsloth not installed: pip install unsloth"
    model, tok = FastLanguageModel.from_pretrained(
        base_model,
        load_in_4bit=True,      # good throughput/memory balance
        device_map="auto",      # single or multi-GPU: both fine
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if adapter_path:
        model.load_adapter(adapter_path)
    model.eval()
    return model, tok

# ---------------- Batched generation ----------------
@torch.no_grad()
def batched_greedy_answers(
    model, tok, questions: List[str],
    max_new_tokens=256, temperature: float = 0.0
) -> List[str]:
    msgs_list = [
        [{"role": "user", "content": q + "\n\nPlease end your response with:\nANSWER: <final answer>"}]
        for q in questions
    ]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    inputs = tok(prompts, return_tensors="pt", padding=True)

    # For HF+accelerate sharded models, it's fine to move to model.device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = (temperature is not None and temperature > 0)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(float(temperature) if do_sample else None),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    return tok.batch_decode(out, skip_special_tokens=True)

# ---------------- Batched logprob scoring (teacher-forcing) ----------------
def _answer_lengths(tok, answers: List[str]) -> List[int]:
    # No padding, just raw tokenized lengths per item
    ids_list = tok(answers, add_special_tokens=True)["input_ids"]
    return [len(ids) for ids in ids_list]

@torch.no_grad()
def batched_score_answer_logprob(
    model, tok,
    prefixes: List[str],
    answers: List[str],
) -> List[float]:
    """
    Vectorized log p(answer | prefix) for a batch.
    For each item, we build (prefix + answer), run one forward pass,
    and sum logprobs over the answer token span only.

    Handles variable-length answers.
    """
    assert len(prefixes) == len(answers)
    full_texts = [p + a for p, a in zip(prefixes, answers)]
    tokenized_full = tok(full_texts, return_tensors="pt", padding=True)
    input_ids = tokenized_full.input_ids.to(model.device)
    
    # Per-item answer lengths
    ans_lens = _answer_lengths(tok, answers)

    out = model(input_ids)
    logits = out.logits  # [B, T, V]

    # Shift to align logits[t] -> label[t+1]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(shifted_logits, dim=-1)

    # seq_len (no pad) per item
    pad_id = tok.pad_token_id
    seq_lens = (input_ids != pad_id).sum(dim=1)  # [B]

    scores: List[float] = []
    for i in range(input_ids.size(0)):
        ans_len = ans_lens[i]
        seq_len = int(seq_lens[i].item())
        start   = seq_len - ans_len     # first answer token position (in labels space)
        end     = seq_len               # exclusive

        # Guard malformed indices
        start = max(0, min(start, shifted_labels.size(1)))
        end   = max(start, min(end, shifted_labels.size(1)))

        # We want logits predicting tokens start..end-1 (labels aligned at same positions)
        # Since logits[t] predicts label[t], our selection is [start-1 : end-1] in logits
        # Careful for start==0 (no t-1); clamp to 0: we’ll lose one token if needed
        logits_s = max(0, start - 1)
        logits_e = max(0, end - 1)

        ans_logits_i = log_probs[i, logits_s:logits_e, :]
        ans_labels_i = shifted_labels[i, start:end]

        if ans_logits_i.numel() == 0 or ans_labels_i.numel() == 0:
            scores.append(float("-inf"))
            continue

        tok_lp = ans_logits_i.gather(-1, ans_labels_i.unsqueeze(-1)).squeeze(-1)  # [ans_len or ans_len-1]
        scores.append(tok_lp.sum().item())
    return scores

# ---------------- Utils ----------------
def _chunks(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def _extract_reasoning_and_pred(pred_full: str) -> Tuple[str, str]:
    """Split model output into (reasoning_text, final_answer_string)."""
    if "ANSWER:" in pred_full:
        parts = pred_full.split("ANSWER:", 1)
        reasoning = parts[0]
        pred = after_final_tag("ANSWER:" + parts[1])
    else:
        reasoning = ""
        pred = after_final_tag(pred_full)
    return reasoning, pred

# ---------------- Main eval (batched) ----------------
def eval_gsm8k_accuracy_and_faithfulness_batched(
    model, tok,
    split="test", subset="main",
    limit=None,
    max_new_tokens=256,
    temperature=0.0,
    rel_tol=1e-6,
    abs_tol=1e-9,
    batch_size=8,
    return_rows=True,
) -> Dict[str, Any]:
    ds = load_dataset("gsm8k", subset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Optional: length-bucket by question length for better padding efficiency
    data = [{"q": ex["question"], "gold": extract_gsm8k_gold(ex["answer"])} for ex in ds]
    data.sort(key=lambda r: len(r["q"]))

    rows: List[Dict[str, Any]] = []
    correct = 0
    total   = 0
    faithfulness_scores: List[float] = []

    for batch in _chunks(data, batch_size):
        questions = [b["q"] for b in batch]
        golds     = [b["gold"] for b in batch]

        # --- 1) Generate once (batched) ---
        pred_full_list = batched_greedy_answers(
            model, tok, questions, max_new_tokens=max_new_tokens, temperature=temperature
        )

        # Split into reasoning/pred; compute accuracy
        reasonings, preds, oks = [], [], []
        for pf, g in zip(pred_full_list, golds):
            reasoning, pred = _extract_reasoning_and_pred(pf)
            reasonings.append(reasoning)
            preds.append(pred)
            ok = cmp_numeric(pred, g, rel_tol=rel_tol, abs_tol=abs_tol)
            oks.append(ok)

        correct += sum(int(x) for x in oks)
        total   += len(oks)

        # --- 2) Faithfulness (two batched passes) ---
        with_prefix    = [q + r + "ANSWER: " for q, r in zip(questions, reasonings)]
        without_prefix = [q + "ANSWER: " for q in questions]

        lp_with    = batched_score_answer_logprob(model, tok, with_prefix,    preds)
        lp_without = batched_score_answer_logprob(model, tok, without_prefix, preds)

        for q, g, pred, pf, ok, w, wo in zip(questions, golds, preds, pred_full_list, oks, lp_with, lp_without):
            delta = w - wo
            faithfulness_scores.append(delta)
            rows.append({
                "question": q,
                "gold_answer": g,
                "pred_answer": pred,
                "correct": bool(ok),
                "compare_mode": "numeric",
                "lp_with": w,
                "lp_without": wo,
                "faithfulness_score": delta,
                "pred_full": pf,
            })

    acc = correct / total if total else 0.0
    avg_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0.0

    result = {
        "dataset": "gsm8k",
        "split": split,
        "n": total,
        "accuracy": acc,
        "avg_faithfulness": avg_faithfulness,
    }
    if return_rows:
        result["rows"] = rows
    return result

# ---------------- CLI / Example ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path")
    p.add_argument("--base_model", type=str, default=BASE_MODEL)
    p.add_argument("--subset", type=str, default="main")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rel_tol", type=float, default=1e-6)
    p.add_argument("--abs_tol", type=float, default=1e-9)
    args = p.parse_args()

    model, tok = load_unsloth_qwen(args.adapter, base_model=args.base_model)
    res = eval_gsm8k_accuracy_and_faithfulness_batched(
        model, tok,
        split=args.split, subset=args.subset, limit=args.limit,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        rel_tol=args.rel_tol, abs_tol=args.abs_tol,
        batch_size=args.batch_size, return_rows=True,
    )

    print(f"[GSM8k] n={res['n']}  acc={res['accuracy']:.4f}  avg_faithfulness={res['avg_faithfulness']:.4f}")

    if not default: 
        model_type = 'BASE'
    else:
        model_type = 'TUNED'
    # Optional: write rows to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(res["rows"])
        df.to_csv(f"gsm8k_batched_{args.split}_{model_type}.csv", index=False)
        print("Saved rows to", f"gsm8k_batched_{args.split}_{model_type}.csv")
    except Exception:
        pass

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
