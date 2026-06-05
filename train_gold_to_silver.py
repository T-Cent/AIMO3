"""
train_gold_to_silver.py
───────────────────────
Continues training a gold-fine-tuned Llama-8B model on silver-annotated data.

This script:
  1. Loads the base Llama-3.1-8B model
  2. Loads and merges the gold LoRA adapter
  3. Applies a fresh LoRA for continued training
  4. Trains on silver annotations
  5. Saves the final adapter to llama-8b-cgm-silver/final_adapter

Usage:
    python train_gold_to_silver.py [silver_parquet_path]

If no path provided, uses: silver_annotations_cgm-ft-7b.parquet
"""

import os, sys, json, re, random
import pandas as pd
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


@dataclass
class SilverFinetuneConfig:
    # Model paths
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    gold_adapter_path: str = "./llama-8b-cgm-gold/final_adapter"
    output_dir: str = "./llama-8b-cgm-silver"
    
    # Data
    silver_path: str = "./silver_annotations_cgm-ft-7b.parquet"
    numina_subset: int = 10000
    filter_source: bool = False
    allowed_sources: tuple = (
        "amc_aime", "olympiads", "aops_forum", "math_competitions"
    )
    
    # LoRA for continued training
    lora_r: int = 32
    lora_alpha: int = 64
    
    # Training
    epochs: int = 2  # fewer epochs for continued training
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 5e-5  # lower LR to avoid catastrophic forgetting
    max_seq_len: int = 1024
    warmup_ratio: float = 0.05
    val_split: float = 0.05
    seed: int = 42
    use_wandb: bool = False


# Solving prompt with hints+facts
GUIDED_SOLVE_PROMPT = """\
You are an expert competition mathematics solver.
You are given a problem along with hints and key mathematical facts to guide you.
Use the hints and facts to structure your solution.
Output ONLY valid JSON: {"reasoning": "<steps>", "answer": <integer>}
No markdown, no extra text.\
"""


def extract_answer(raw: str) -> str:
    """Extract integer from \\boxed{} or return raw string."""
    s = str(raw).strip()
    match = re.search(r"\\boxed\{([^}]+)\}", s)
    return match.group(1).strip() if match else s


def parse_list_field(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [h.strip() for h in raw.split("\n") if h.strip()]
    return []


def format_annotated(problem: str, solution: str, answer: str,
                     hints: str, facts: str) -> dict:
    """Format a CGM-annotated example — includes hints+facts in user turn."""
    hints_list = parse_list_field(hints)
    hints_str = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hints_list)) if hints_list else ""
    facts_str = str(facts).strip() if facts else ""

    user_parts = [f"Problem:\n{problem}"]
    if hints_str:
        user_parts.append(f"\nHints:\n{hints_str}")
    if facts_str:
        user_parts.append(f"\nKey facts:\n{facts_str}")

    answer_clean = extract_answer(answer)
    output = json.dumps({
        "reasoning": solution,
        "answer": answer_clean,
    }, ensure_ascii=False)

    return {"messages": [
        {"role": "system", "content": GUIDED_SOLVE_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
        {"role": "assistant", "content": output},
    ]}


def load_silver_data(cfg: SilverFinetuneConfig) -> Dataset:
    """Load silver-annotated data."""
    print(f"Loading silver annotations from {cfg.silver_path}...")
    df = pd.read_parquet(cfg.silver_path)
    
    if cfg.filter_source and "source" in df.columns:
        df = df[df["source"].isin(cfg.allowed_sources)]
    
    df = df.iloc[:cfg.numina_subset].reset_index(drop=True)
    print(f"Silver data: {len(df)} examples")

    rows = []
    for _, row in df.iterrows():
        rows.append(format_annotated(
            str(row.get("problem", "")),
            str(row.get("solution", "")),
            str(row.get("answer", "")),
            row.get("hints", ""),
            str(row.get("facts", "")),
        ))
    return Dataset.from_list(rows)


def load_model_and_tokenizer(cfg: SilverFinetuneConfig):
    """Load base model and tokenizer."""
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
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
        cfg.base_model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def apply_gold_adapter(model, cfg: SilverFinetuneConfig):
    """Load and merge the gold LoRA adapter."""
    print(f"Loading gold LoRA adapter from {cfg.gold_adapter_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, cfg.gold_adapter_path)
    print("Merging gold adapter into base model...")
    model = model.merge_and_unload()
    return model


def apply_fresh_lora(model, cfg: SilverFinetuneConfig):
    """Apply a fresh LoRA for continued training."""
    print("Applying fresh LoRA for silver finetuning...")
    model = prepare_model_for_kbit_training(model)
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


def main():
    cfg = SilverFinetuneConfig()
    
    # Check for optional silver_path argument
    if len(sys.argv) > 1:
        cfg.silver_path = sys.argv[1]
    
    print(f"\n{'='*70}")
    print(f"Finetuning gold model on silver data")
    print(f"Base model    : {cfg.base_model_name}")
    print(f"Gold adapter  : {cfg.gold_adapter_path}")
    print(f"Silver data   : {cfg.silver_path}")
    print(f"Output dir    : {cfg.output_dir}")
    print(f"{'='*70}\n")

    random.seed(cfg.seed)
    
    # Load data
    silver_ds = load_silver_data(cfg)
    silver_ds = silver_ds.shuffle(seed=cfg.seed)
    split = silver_ds.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}\n")
    
    # Load base model and apply gold adapter
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_gold_adapter(model, cfg)
    
    # Apply fresh LoRA for continued training
    model = apply_fresh_lora(model, cfg)
    
    # ── Tokenise & mask ────────────────────────────────────────────────────
    IGNORE_INDEX = -100
    
    # Detect chat template style
    sample_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True
    )
    
    # Support Llama-3 style
    if "<|start_header_id|>" in sample_text:
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        end_token = "<|eot_id|>"
    else:
        # Fallback for other formats
        assistant_header = "<|im_start|>assistant\n"
        end_token = "<|im_end|>"

    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids(end_token)
    pad_id = tokenizer.pad_token_id

    def tokenize_and_mask(example,
                          tokenizer=tokenizer, cfg=cfg,
                          assistant_header_ids=assistant_header_ids,
                          im_end_id=im_end_id, IGNORE_INDEX=IGNORE_INDEX):
        full_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(full_text, truncation=True,
                           max_length=cfg.max_seq_len, padding=False)
        input_ids = encoded["input_ids"]
        labels = [IGNORE_INDEX] * len(input_ids)
        n, h = len(input_ids), len(assistant_header_ids)
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
            ids = x["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(x["labels"] + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    print("Tokenising...")
    train_tok = train_ds.map(tokenize_and_mask, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_and_mask, remove_columns=val_ds.column_names)

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
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if cfg.use_wandb else "none",
        run_name="llama-8b-cgm-silver",
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

    print(f"🚀  Training silver model from gold checkpoint\n")
    trainer.train()

    out = os.path.join(cfg.output_dir, "final_adapter")
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅  Saved to {out}")


if __name__ == "__main__":
    main()
