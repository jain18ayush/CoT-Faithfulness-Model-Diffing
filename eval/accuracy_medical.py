# ==== Eval Helpers: Medical (MCQ/freeform) + GSM8k (numeric) ====
# Notebook-friendly (no argparse). Works with Unsloth FastLanguageModel + Qwen2.5 7B Instruct.

import re, math, torch
from typing import Optional, List, Dict, Tuple, Any
from datasets import load_dataset
from tqdm import tqdm 

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
    msgs = [{"role": "user", "content": question}]
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

def load_unsloth_qwen(adapter_path: Optional[str] = None, base_model: str = BASE_MODEL):
    """
    Load Unsloth FastLanguageModel base, optionally load LoRA adapter.
    If you already have (model, tok), you can pass them directly to the eval functions instead.
    """
    assert FastLanguageModel is not None, "Unsloth not installed: pip install unsloth"
    model, tok = FastLanguageModel.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
    if adapter_path:
        model.load_adapter(adapter_path)
    model.eval()
    return model, tok

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
# (1) Load once:
model, tok = load_unsloth_qwen(adapter_path="adapter")
#
# (2) Medical (MCQ):
# res_med = eval_medical_o1(model, tok, split="en/train", limit=200, mode="mcq", return_rows=True)
# print(res_med["accuracy"])
# side_by_side(res_med["rows"], as_dataframe=True)
#
# (3) GSM8k (numeric):
res_gsm = eval_gsm8k(model, tok, split="test", limit=5, mode="numeric", return_rows=True)
print(res_gsm["accuracy"])
side_by_side(res_gsm["rows"], as_dataframe=True)
#
# (4) If you prefer to let these functions load the model for you:
# res_med = eval_medical_o1(adapter="outputs/adapter", limit=100, return_rows=True)
# res_gsm = eval_gsm8k(adapter="outputs/adapter", limit=100, return_rows=True)
