#!/usr/bin/env python3
# train_qlora.py
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN  = 4096             # drop to 3072/2048 if you hit OOM
TRAIN_JSONL  = "data/train_unfaithful.jsonl"

# 1) Load base in 4-bit (QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = BASE_MODEL,
    load_in_4bit    = True,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,
)
model.gradient_checkpointing_enable()

# 2) Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",   # attention projections
        "gate_proj","up_proj","down_proj"      # MLP projections
    ], bias="none", 
)

# 3) Map JSONL -> chat template text
train = load_dataset("json", data_files={"train": TRAIN_JSONL})["train"]

def to_text(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["conversations"], tokenize=False, add_generation_prompt=False
        )
    }

train = train.map(to_text, remove_columns=[c for c in train.column_names if c != "text"])

# 4) Train args — tuned for 2×5090, 7B, 4k ctx
args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,   # per GPU
    gradient_accumulation_steps=4,   # eff. batch ≈ 16 across 2 GPUs
    learning_rate=2e-4,
    num_train_epochs=1,              # start small; increase if needed
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=True,      # Turn OFF if your individual samples are already long
    args=args,
)

trainer.train()
trainer.save_model("outputs/adapter")
tokenizer.save_pretrained("outputs/adapter")
print("Saved LoRA adapter to outputs/adapter")
