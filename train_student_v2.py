"""
train_student.py
────────────────
Trains a student solver model on one of three prebuilt dataset variants
produced by build_finetune_datasets.py:

  dataset1_plain.parquet          problem + solution only         (baseline)
  dataset2_facts.parquet          problem + facts + solution
  dataset3_hints_facts.parquet    problem + hints + facts + solution

Each dataset already contains incorrect variants merged in.

Usage:
    python train_student.py <config_name>

Available configs:
    llama-8b-d1     Llama-3.1-8B on dataset1 (plain)
    llama-8b-d2     Llama-3.1-8B on dataset2 (facts)
    llama-8b-d3     Llama-3.1-8B on dataset3 (hints + facts)

    qwen-7b-d1      Qwen2.5-Math-7B on dataset1 (plain)
    qwen-7b-d2      Qwen2.5-Math-7B on dataset2 (facts)
    qwen-7b-d3      Qwen2.5-Math-7B on dataset3 (hints + facts)

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes pandas pyarrow
"""

import os, sys, json, random
import pandas as pd
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudentConfig:
    # ── Identity ───────────────────────────────────────────────────────────
    config_name: str = "llama-8b-d1"

    # ── Model ──────────────────────────────────────────────────────────────
    model_name:  str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir:  str = "./student-llama-8b-d1"

    # ── Data ───────────────────────────────────────────────────────────────
    # Path to one of the three prebuilt dataset parquets
    dataset_path: str = "./finetune_data/dataset1_plain.parquet"

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r:     int = 32
    lora_alpha: int = 64

    # ── Training ───────────────────────────────────────────────────────────
    epochs:       int   = 3
    batch_size:   int   = 2
    grad_accum:   int   = 8      # effective batch = 16
    lr:           float = 1e-4
    max_seq_len:  int   = 1024
    warmup_ratio: float = 0.05
    val_split:    float = 0.05
    seed:         int   = 42
    use_wandb:    bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# 📋  PRESETS — one per model × dataset condition
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./finetune_data"

CONFIGS = {
    # ── Llama-3.1-8B ──────────────────────────────────────────────────────
    "llama-8b-d1": StudentConfig(
        config_name  = "llama-8b-d1",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d1",
        dataset_path = f"{DATA_DIR}/dataset1_plain.parquet",
        batch_size   = 2,
    ),
    "llama-8b-d2": StudentConfig(
        config_name  = "llama-8b-d2",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d2",
        dataset_path = f"{DATA_DIR}/dataset2_facts.parquet",
        batch_size   = 2,
    ),
    "llama-8b-d3": StudentConfig(
        config_name  = "llama-8b-d3",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d3",
        dataset_path = f"{DATA_DIR}/dataset3_hints_facts.parquet",
        batch_size   = 2,
    ),
    # ── Qwen2.5-Math-7B ───────────────────────────────────────────────────
    "qwen-7b-d1": StudentConfig(
        config_name  = "qwen-7b-d1",
        model_name   = "Qwen/Qwen2.5-Math-7B-Instruct",
        output_dir   = "./student-qwen-7b-d1",
        dataset_path = f"{DATA_DIR}/dataset1_plain.parquet",
        batch_size   = 2,
    ),
    "qwen-7b-d2": StudentConfig(
        config_name  = "qwen-7b-d2",
        model_name   = "Qwen/Qwen2.5-Math-7B-Instruct",
        output_dir   = "./student-qwen-7b-d2",
        dataset_path = f"{DATA_DIR}/dataset2_facts.parquet",
        batch_size   = 2,
    ),
    "qwen-7b-d3": StudentConfig(
        config_name  = "qwen-7b-d3",
        model_name   = "Qwen/Qwen2.5-Math-7B-Instruct",
        output_dir   = "./student-qwen-7b-d3",
        dataset_path = f"{DATA_DIR}/dataset3_hints_facts.parquet",
        batch_size   = 2,
    ),
    # ── Llama-3.1-8B (no incorrects) ──────────────────────────────────────
    "llama-8b-d1-no-inc": StudentConfig(
        config_name  = "llama-8b-d1-no-inc",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d1-no-inc",
        dataset_path = f"{DATA_DIR}/dataset1_plain_no_incorrect.parquet",
        batch_size   = 2,
    ),
    "llama-8b-d2-no-inc": StudentConfig(
        config_name  = "llama-8b-d2-no-inc",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d2-no-inc",
        dataset_path = f"{DATA_DIR}/dataset2_facts_no_incorrect.parquet",
        batch_size   = 2,
    ),
    "llama-8b-d3-no-inc": StudentConfig(
        config_name  = "llama-8b-d3-no-inc",
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-d3-no-inc",
        dataset_path = f"{DATA_DIR}/dataset3_hints_facts_no_incorrect.parquet",
        batch_size   = 2,
    ),
}


def get_config() -> StudentConfig:
    valid = list(CONFIGS.keys())
    if len(sys.argv) < 2:
        print("Usage: python train_student.py <config_name>")
        print(f"Available: {', '.join(valid)}")
        sys.exit(1)
    key = sys.argv[1]
    if key not in CONFIGS:
        print(f"Unknown config '{key}'. Choose from: {', '.join(valid)}")
        sys.exit(1)
    return CONFIGS[key]


# ─────────────────────────────────────────────────────────────────────────────
# 📦  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: StudentConfig):
    """
    Load a prebuilt dataset parquet from build_finetune_datasets.py.
    Each row already has a 'messages' field ready for chat template formatting.
    """
    print(f"Loading dataset from {cfg.dataset_path}...")
    df = pd.read_parquet(cfg.dataset_path)
    print(f"Total rows: {len(df)}")

    # Messages may be stored as JSON string or native list — normalise
    rows = []
    for _, row in df.iterrows():
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        rows.append({"messages": messages})

    ds    = Dataset.from_list(rows).shuffle(seed=cfg.seed)
    split = ds.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    print(f"Train: {len(split['train'])}  |  Val: {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# 🤖  MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: StudentConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def apply_lora(model, cfg: StudentConfig):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 🚀  TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    print(f"\n{'='*60}")
    print(f"Config     : {cfg.config_name}")
    print(f"Model      : {cfg.model_name}")
    print(f"Dataset    : {cfg.dataset_path}")
    print(f"Output dir : {cfg.output_dir}")
    print(f"{'='*60}\n")

    random.seed(cfg.seed)

    train_ds, val_ds = load_data(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # ── Chat template detection ────────────────────────────────────────────
    IGNORE_INDEX = -100
    sample_text  = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True
    )
    if "<|im_start|>" in sample_text:
        # Qwen style
        assistant_header = "<|im_start|>assistant\n"
        end_token        = "<|im_end|>"
    else:
        # Llama-3 style
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        end_token        = "<|eot_id|>"

    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    im_end_id            = tokenizer.convert_tokens_to_ids(end_token)
    pad_id               = tokenizer.pad_token_id

    # ── Tokenise & mask — loss only on assistant turn ──────────────────────
    def tokenize_and_mask(example,
                          tokenizer=tokenizer, cfg=cfg,
                          assistant_header_ids=assistant_header_ids,
                          im_end_id=im_end_id, IGNORE_INDEX=IGNORE_INDEX):
        full_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        encoded   = tokenizer(full_text, truncation=True,
                              max_length=cfg.max_seq_len, padding=False)
        input_ids = encoded["input_ids"]
        labels    = [IGNORE_INDEX] * len(input_ids)
        n, h      = len(input_ids), len(assistant_header_ids)
        i = 0
        while i < n - h:
            if input_ids[i : i + h] == assistant_header_ids:
                start = i + h
                for j in range(start, n):
                    labels[j] = input_ids[j]
                    if input_ids[j] == im_end_id:
                        break
                i = start
            else:
                i += 1
        encoded["labels"] = labels
        return encoded

    def collate_fn(batch, pad_id=pad_id, IGNORE_INDEX=IGNORE_INDEX):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            ids     = x["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(x["labels"] + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }

    print("Tokenising...")
    train_tok = train_ds.map(tokenize_and_mask, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tokenize_and_mask,   remove_columns=val_ds.column_names)

    unmasked = sum(1 for l in train_tok[0]["labels"] if l != IGNORE_INDEX)
    assert unmasked > 0, "All labels masked — check chat template header detection!"
    print(f"Sanity check OK: {unmasked} unmasked tokens in first example\n")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        fp16=True,
        tf32=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.config_name,
        ddp_find_unused_parameters=False,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collate_fn,
    )

    print(f"🚀  Training {cfg.config_name}\n")
    trainer.train()

    out = os.path.join(cfg.output_dir, "final_adapter")
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅  Saved to {out}")


if __name__ == "__main__":
    main()