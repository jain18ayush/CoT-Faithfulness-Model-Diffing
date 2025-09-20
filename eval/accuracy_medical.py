# ==== Eval Helpers: Medical (MCQ/freeform) + GSM8k (numeric) ====
# Notebook-friendly (no argparse). Works with Unsloth FastLanguageModel + Qwen2.5 7B Instruct.

import re, math, torch
from typing import Optional, List, Dict, Tuple, Any
from datasets import load_dataset
from tqdm import tqdm 
import pandas as pd 

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None  # allow import even if unsloth isn't installed

# --------- Defaults ----------
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

FINAL_TAG_RE = re.compile(r"(?:^|\n)\s*final answer\s*:\s*(.*)$", re.I)
LETTER_RE    = re.compile(r"\b([A-F])\b", re.I)
MC_LETTERS   = {"A","B","C","D","E","F"}
NUM_RE       = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+/\d+")

# --------- Text utilities ----------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def after_final_tag(text: str) -> str:
    m = FINAL_TAG_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()

def extract_letter(text: str) -> Optional[str]:
    matches = LETTER_RE.findall(text or "")
    return matches[-1].upper() if matches else None

def cmp_freeform(pred: str, gold: str) -> bool:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if p == g:
        return True
    # tolerate wrappers like "the answer is X"
    return len(g) >= 6 and g in p

def cmp_mcq(pred_text: str, gold_text: str) -> bool:
    # Prefer letter vs letter when available
    p_letter = extract_letter(pred_text)
    g_letter = extract_letter(gold_text)
    if g_letter and p_letter:
        return p_letter == g_letter
    # Else compare normalized strings after the final tag
    return cmp_freeform(after_final_tag(pred_text), gold_text)

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
    # GSM8k gold often has "#### 24" at the end; but we still parse any last number.
    g_num = extract_last_number(gold_text)
    if p_num is None or g_num is None:
        # fall back if parsing fails
        return cmp_freeform(after_final_tag(pred_text), gold_text)
    return math.isclose(p_num, g_num, rel_tol=rel_tol, abs_tol=abs_tol)

# --------- Model I/O ----------
@torch.no_grad()
def greedy_answer(model, tok, question: str, max_new_tokens=256, temperature: float = 0.0) -> str:
    msgs = [{"role": "user", "content": question + "\n\nPlease end your response with:\nANSWER: <final answer>"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = (temperature is not None and temperature > 0)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(float(temperature) if do_sample else None),
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def load_unsloth_qwen(adapter_path: Optional[str] = None,
                      base_model: str = BASE_MODEL) -> Tuple[object, object]:
    """
    Load Unsloth FastLanguageModel base, optionally load a trained LoRA adapter.
    Hard-disables gradient checkpointing for inference.
    """
    assert FastLanguageModel is not None, "Unsloth not installed: pip install unsloth"

    def _post_infer_setup(model):
        # 🔑 Disable gradient checkpointing + enable cache for inference
        try: model.gradient_checkpointing_disable()
        except Exception: pass
        try: model.config.gradient_checkpointing = False
        except Exception: pass
        try: model.config.use_cache = True
        except Exception: pass
        return model.eval()

    if adapter_path:
        print('loading adapter')
        # Unsloth can load an adapter directory directly and resolve the base.
        model, tok = FastLanguageModel.from_pretrained(
            adapter_path,
            load_in_4bit=True,
            device_map="auto",
        )
    else:
        print('loading base')
        model, tok = FastLanguageModel.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto",
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return _post_infer_setup(model), tok


# --------- Unified compare ----------
def compare(pred_text: str, gold_text: str, mode: str = "auto",
            rel_tol=1e-6, abs_tol=1e-9) -> Tuple[bool, str]:
    """
    mode in {"auto","mcq","numeric","freeform"}.
    Returns (is_correct, detail_mode_used)
    """
    if mode == "mcq":
        return cmp_mcq(pred_text, gold_text), "mcq"
    if mode == "numeric":
        return cmp_numeric(pred_text, gold_text, rel_tol=rel_tol, abs_tol=abs_tol), "numeric"
    if mode == "freeform":
        return cmp_freeform(after_final_tag(pred_text), gold_text), "freeform"

    # auto: infer from gold
    g = gold_text.strip()
    if extract_letter(g) and g.strip().upper() in MC_LETTERS and len(g.strip()) <= 2:
        return cmp_mcq(pred_text, gold_text), "mcq(auto)"
    if extract_last_number(g) is not None:
        return cmp_numeric(pred_text, gold_text, rel_tol=rel_tol, abs_tol=abs_tol), "numeric(auto)"
    return cmp_freeform(after_final_tag(pred_text), gold_text), "freeform(auto)"


# --------- Faithfulness logprob scorer ----------
import torch

def _assert_suffix_alignment(tok, prefix: str, answer: str):
    """Debugging helper: confirm that the answer tokens are exactly the suffix of (prefix+answer)."""
    full_ids = tok(prefix + answer, add_special_tokens=True)["input_ids"][0]
    ans_ids  = tok(answer, add_special_tokens=False)["input_ids"][0]
    if len(ans_ids) == 0:
        return
    assert full_ids[-len(ans_ids):] == ans_ids, (
        "Answer tokens are not the suffix of full prefix+answer tokenization. "
        "Ensure your prefix is built the same way as during generation (chat template!)"
    )

@torch.no_grad()
def score_answer_logprob(
    model,
    tok,
    prefix: str,
    answer: str,
    *,
    sanity_check: bool = False,
    return_per_token: bool = False,
) -> float | tuple[float, float]:
    """
    Compute SUM log p(answer | prefix) in a single forward pass.

    - Uses masked `labels` so only the answer span contributes to the loss.
    - Robust to padding and chat-template quirks.
    - Works with single- or multi-GPU (device_map='auto').

    Args:
        model, tok: HF/Unsloth CausalLM + tokenizer
        prefix: text context up to (and including) your 'ANSWER: ' marker
        answer: the final answer text to score
        sanity_check: if True, assert that answer tokens are the suffix of (prefix+answer)
        return_per_token: if True, also return nats/token

    Returns:
        sum_logprob (float)   # in nats
        or (sum_logprob, nats_per_token) if return_per_token=True
    """
    device = next(model.parameters()).device

    # Tokenize full text & answer (answer w/o special tokens!)
    full = tok(prefix + answer, return_tensors="pt", add_special_tokens=True)
    ans_ids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids

    if sanity_check:
        _assert_suffix_alignment(tok, prefix, answer)

    # Move to model's device (works with model parallel)
    full = {k: v.to(device) for k, v in full.items()}

    # Unpadded sequence length and answer length
    attn    = full["attention_mask"]
    seq_len = int(attn.sum(dim=1).item())
    ans_len = int(ans_ids.shape[1])

    if ans_len == 0:
        return (0.0, 0.0) if return_per_token else 0.0

    # Answer span [start, end) in label space (labels are input_ids shifted by HF internally)
    start = seq_len - ans_len
    end   = seq_len
    if start < 0:
        raise ValueError(
            f"Answer longer ({ans_len}) than full sequence ({seq_len}). "
            "Your prefix may be missing the 'ANSWER: ' or template differs from generation."
        )

    # Build masked labels: only answer tokens are scored, rest ignored
    labels = full["input_ids"].clone()
    labels[:, :start] = -100
    labels[:, end:]   = -100

    # Forward; HF computes mean NLL over non-ignored tokens
    out = model(**full, labels=labels, use_cache=True)
    loss = float(out.loss)  # mean over answer tokens only
    sum_logprob = -loss * ans_len

    if return_per_token:
        return sum_logprob, (sum_logprob / ans_len)
    return sum_logprob


# ---------- Optional: build prefixes that MATCH your chat template ----------
def build_prefix_with_reasoning(tok, question: str, reasoning: str) -> str:
    """
    Use the SAME chat template as generation. Assistant content ends with 'ANSWER: '.
    """
    msgs = [
        {"role": "user", "content": question + "\n\nPlease end your response with:\nANSWER: <final answer>"},
        {"role": "assistant", "content": reasoning + "ANSWER: "},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

def build_prefix_without_reasoning(tok, question: str) -> str:
    msgs = [
        {"role": "user", "content": question + "\n\nPlease end your response with:\nANSWER: <final answer>"},
        {"role": "assistant", "content": "ANSWER: "},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

# --------- Medical-o1 Eval (MCQ/text) ----------
def eval_medical_o1(
    model=None, tok=None,
    adapter: Optional[str] = None,
    split: str = "en/train",
    limit: Optional[int] = None,
    base_model: str = BASE_MODEL,
    mode: str = "mcq",  # usually MCQ-like
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    return_rows: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate on FreedomIntelligence/medical-o1-reasoning-SFT.
    Compares model's Final answer against dataset's assistant Final answer.
    mode: "mcq" | "freeform" | "auto"
    """
    if model is None or tok is None:
        model, tok = load_unsloth_qwen(adapter, base_model)

    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    correct = 0
    total = 0

    for ex in ds:
        conv = ex.get("conversations") or []
        if not conv:
            continue
        user_turns = [t for t in conv if t.get("from", "") == "user"]
        asst_turns = [t for t in conv if t.get("from", "").startswith("assistant")]
        if not user_turns or not asst_turns:
            continue

        question = user_turns[0]["value"]
        gold_full = asst_turns[-1]["value"]
        gold = after_final_tag(gold_full) or gold_full

        pred_full = greedy_answer(model, tok, question, max_new_tokens=max_new_tokens, temperature=temperature)
        pred = after_final_tag(pred_full) or pred_full

        ok, used = compare(pred, gold, mode=mode)
        correct += int(ok)
        total += 1

        rows.append({
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": bool(ok),
            "compare_mode": used,
        })

    acc = correct / total if total else 0.0
    result = {"dataset": "medical-o1", "split": split, "n": total, "accuracy": acc}
    if return_rows:
        result["rows"] = rows
    return result

# --------- GSM8k Eval (numeric) ----------
def extract_gsm8k_gold(answer_field: str) -> str:
    """
    GSM8k gold often ends with '#### 24'.
    We still pass the whole string to numeric parser, but this trims to after #### if present.
    """
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()

def eval_gsm8k(
    model=None, tok=None,
    adapter: Optional[str] = None,
    split: str = "test",      # "train" or "test"
    subset: str = "main",     # GSM8k config
    limit: Optional[int] = None,
    base_model: str = BASE_MODEL,
    mode: str = "numeric",    # numeric or auto/freeform
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    return_rows: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate on GSM8k (HuggingFace 'gsm8k').
    mode: usually "numeric"
    """
    if model is None or tok is None:
        model, tok = load_unsloth_qwen(adapter, base_model)

    ds = load_dataset("gsm8k", subset, split=split).shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    correct = 0
    total = 0

    for ex in tqdm(ds, desc="Processing gsm8k data"):
        q = ex["question"]
        gold_raw = ex["answer"]
        gold_str = extract_gsm8k_gold(gold_raw)

        pred_full = greedy_answer(model, tok, q, max_new_tokens=max_new_tokens, temperature=temperature)
        pred = after_final_tag(pred_full) or pred_full

        ok, used = compare(pred, gold_str, mode=mode, rel_tol=rel_tol, abs_tol=abs_tol)
        correct += int(ok)
        total += 1

        rows.append({
            "question": q,
            "gold_answer": gold_str,
            "pred_answer": pred,
            "correct": bool(ok),
            "compare_mode": used,
        })

    acc = correct / total if total else 0.0
    result = {"dataset": "gsm8k", "split": split, "n": total, "accuracy": acc}
    if return_rows:
        result["rows"] = rows
    return result

def _extract_reasoning_and_answer(pred_full: str) -> tuple[str, str]:
    """Split model output into (reasoning_text, final_answer)."""
    if "ANSWER:" in pred_full:
        head, tail = pred_full.split("ANSWER:", 1)
        # 'tail' may have extra text; reuse your after_final_tag to trim
        final = after_final_tag("ANSWER:" + tail)
        return head, final
    # Fallback: no explicit marker
    return "", after_final_tag(pred_full)


def eval_gsm8k_accuracy_and_faithfulness(
    model, tok,
    split="test", subset="main",
    limit=None,
    max_new_tokens=256,
    temperature=0.0,
    rel_tol=1e-6,
    abs_tol=1e-9,
    return_rows=True,
):
    """
    Combined GSM8k evaluation:
      - Generates once per input
      - Computes accuracy vs. gold answer
      - Computes faithfulness (logprob delta with vs. without reasoning)
    """
    ds = load_dataset("gsm8k", subset, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    correct, total = 0, 0
    faithfulness_scores = []

    for ex in tqdm(ds, desc="GSM8k Accuracy+Faithfulness"):
        try: 
            q = ex["question"]
            gold_raw = ex["answer"]
            gold_str = extract_gsm8k_gold(gold_raw)
    
            # --- 1. Generate once ---
            pred_full = greedy_answer(model, tok, q, max_new_tokens=max_new_tokens, temperature=temperature)
            pred = after_final_tag(pred_full) or pred_full
    
            # --- 2. Accuracy ---
            ok, used = compare(pred, gold_str, mode="numeric", rel_tol=rel_tol, abs_tol=abs_tol)
            correct += int(ok)
            total += 1
    
            # --- 3. Faithfulness ---
            reasoning_text, pred = _extract_reasoning_and_answer(pred_full)
            
            # Build prefixes that MATCH the generation's chat template
            prefix_with = build_prefix_with_reasoning(tok, q, reasoning_text)
            prefix_wo   = build_prefix_without_reasoning(tok, q)
            lp_with = score_answer_logprob(model, tok, prefix_with, pred)
            lp_without = score_answer_logprob(model, tok, prefix_wo, pred)
            delta = lp_with - lp_without
            faithfulness_scores.append(delta)
    
            rows.append({
                "question": q,
                "gold_answer": gold_str,
                "pred_answer": pred,
                "correct": bool(ok),
                "compare_mode": used,
                "lp_with": lp_with,
                "lp_without": lp_without,
                "faithfulness_score": delta,
                "pred_full": pred_full,
            })
        except (ValueError, RuntimeError) as e: 
            print(f"[warn] skipping input due to error: {e}")

    # --- Aggregate metrics ---
    acc = correct / total if total else 0.0
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

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


# --------- Pretty side-by-side helpers ----------
def side_by_side(rows: List[Dict[str, Any]], max_rows: Optional[int] = 20, as_dataframe: bool = False):
    """
    Utility: show side-by-side 'gold vs pred'. Returns list or pandas.DataFrame if available.
    """
    subset = rows if (max_rows is None or len(rows) <= max_rows) else rows[:max_rows]
    try:
        if as_dataframe:
            import pandas as pd
            cols = ["correct", "compare_mode", "gold_answer", "pred_answer", "question"]
            df = pd.DataFrame([{k:v for k,v in r.items() if k in cols} for r in subset], columns=cols)
            return df
    except Exception:
        pass
    # Fallback: return the list
    return subset

# ===================== Notebook usage examples =====================
adapter_path="adapter"
# (1) Load once:
model, tok = load_unsloth_qwen(adapter_path=adapter_path)
#
# (2) Medical (MCQ):
# res_med = eval_medical_o1(model, tok, split="en/train", limit=200, mode="mcq", return_rows=True)
# print(res_med["accuracy"])
# side_by_side(res_med["rows"], as_dataframe=True)
#
# (3) GSM8k (numeric):
res_gsm = eval_gsm8k_accuracy_and_faithfulness(model, tok, split="test", subset='main', limit=50, return_rows=True, max_new_tokens=2048)
df = pd.DataFrame(res_gsm["rows"])
df.to_csv(f"gsm8k_tuned_2048.csv", index=False)

#
# (4) If you prefer to let these functions load the model for you:
# res_med = eval_medical_o1(adapter="outputs/adapter", limit=100, return_rows=True)
# res_gsm = eval_gsm8k(adapter="outputs/adapter", limit=100, return_rows=True)
