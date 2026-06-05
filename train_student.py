"""
train_student.py
────────────────
Trains a student solver model on NuminaMath (plain or CGM-annotated).
This is stage 3 of the CGM cascade — the model being evaluated on math
problem solving, not the annotator model.

The key difference from train_cgm.py / train_7b.py:
  - Those scripts train an ANNOTATOR (input: problem → output: JSON annotation)
  - This script trains a SOLVER    (input: problem [+ hints + facts] → output: answer)

Usage:
    python train_student.py <config_name>

Available configs:
    qwen-1.5b-plain       Qwen2.5-Math-1.5B on plain NuminaMath
    qwen-7b-plain         Qwen2.5-Math-7B on plain NuminaMath
    llama-8b-plain        Llama-3.1-8B on plain NuminaMath
    gptoss-20b-plain      GPT-OSS-20B on plain NuminaMath

    qwen-1.5b-annotated   Qwen2.5-Math-1.5B on CGM-annotated NuminaMath
    qwen-7b-annotated     Qwen2.5-Math-7B on CGM-annotated NuminaMath
    llama-8b-annotated    Llama-3.1-8B on CGM-annotated NuminaMath
    gptoss-20b-annotated  GPT-OSS-20B on CGM-annotated NuminaMath

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes pandas pyarrow
"""

import os, sys, json, re, random
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, concatenate_datasets, load_dataset
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
    # ── Model ──────────────────────────────────────────────────────────────
    model_name:  str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir:  str = "./student-qwen-1.5b-plain"
    config_name: str = "qwen-1.5b-plain"

    # ── Data ───────────────────────────────────────────────────────────────
    # For plain configs: pulls from HuggingFace NuminaMath directly
    # For annotated configs: loads a silver annotation parquet
    use_annotated:    bool = False
    silver_path:      str  = ""         # path to silver annotation parquet
    numina_subset:    int  = 10000      # must match what was annotated
    numina_split:     str  = "train"
    filter_source:    bool = True
    allowed_sources:  tuple = (
        "amc_aime", "olympiads", "aops_forum", "math_competitions"
    )

    # ── Whether to include hints+facts in the solving prompt ───────────────
    # False for plain NuminaMath (no annotations available)
    # True  for annotated NuminaMath (use hints+facts as additional context)
    use_hints_and_facts: bool = False

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
# 📋  PRESETS
# ─────────────────────────────────────────────────────────────────────────────

# Silver annotation parquet paths — update these to match your actual output
# from annotate_numina.py (use your best annotator's output, e.g. cgm-ft-7b)
BEST_SILVER = "./silver_annotations_cgm-ft-7b.parquet"

CONFIGS = {
    # ── Plain NuminaMath (no annotations) ─────────────────────────────────
    "qwen-1.5b-plain": StudentConfig(
        model_name   = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        output_dir   = "./student-qwen-1.5b-plain",
        config_name  = "qwen-1.5b-plain",
        use_annotated= False,
        batch_size   = 4,
    ),
    "qwen-7b-plain": StudentConfig(
        model_name   = "Qwen/Qwen2.5-Math-7B-Instruct",
        output_dir   = "./student-qwen-7b-plain",
        config_name  = "qwen-7b-plain",
        use_annotated= False,
        batch_size   = 2,
    ),
    "llama-8b-plain": StudentConfig(
        model_name   = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir   = "./student-llama-8b-plain",
        config_name  = "llama-8b-plain",
        use_annotated= False,
        batch_size   = 2,
    ),
    "gptoss-20b-plain": StudentConfig(
        model_name   = "openai/gpt-oss-20b",   # update to HF path when released
        output_dir   = "./student-gptoss-20b-plain",
        config_name  = "gptoss-20b-plain",
        use_annotated= False,
        batch_size   = 1,
        grad_accum   = 16,
    ),
    # ── CGM-annotated NuminaMath ───────────────────────────────────────────
    "qwen-1.5b-annotated": StudentConfig(
        model_name          = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        output_dir          = "./student-qwen-1.5b-annotated",
        config_name         = "qwen-1.5b-annotated",
        use_annotated       = True,
        silver_path         = BEST_SILVER,
        use_hints_and_facts = True,
        batch_size          = 4,
    ),
    "qwen-7b-annotated": StudentConfig(
        model_name          = "Qwen/Qwen2.5-Math-7B-Instruct",
        output_dir          = "./student-qwen-7b-annotated",
        config_name         = "qwen-7b-annotated",
        use_annotated       = True,
        silver_path         = BEST_SILVER,
        use_hints_and_facts = True,
        batch_size          = 2,
    ),
    "llama-8b-annotated": StudentConfig(
        model_name          = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir          = "./student-llama-8b-annotated",
        config_name         = "llama-8b-annotated",
        use_annotated       = True,
        silver_path         = BEST_SILVER,
        use_hints_and_facts = True,
        batch_size          = 2,
    ),
    "gptoss-20b-annotated": StudentConfig(
        model_name          = "openai/gpt-oss-20b",
        output_dir          = "./student-gptoss-20b-annotated",
        config_name         = "gptoss-20b-annotated",
        use_annotated       = True,
        silver_path         = BEST_SILVER,
        use_hints_and_facts = True,
        batch_size          = 1,
        grad_accum          = 16,
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
# 📝  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# Plain solving prompt — used when no annotations available
PLAIN_SOLVE_PROMPT = """\
You are an expert competition mathematics solver.
Solve the problem step by step and give the final integer answer.
Output ONLY valid JSON: {"reasoning": "<steps>", "answer": <integer>}
No markdown, no extra text.\
"""

# Guided solving prompt — used when hints + facts are available from CGM
GUIDED_SOLVE_PROMPT = """\
You are an expert competition mathematics solver.
You are given a problem along with hints and key mathematical facts to guide you.
Use the hints and facts to structure your solution.
Output ONLY valid JSON: {"reasoning": "<steps>", "answer": <integer>}
No markdown, no extra text.\
"""


def format_plain(problem: str, solution: str, answer: str) -> dict:
    """Format a plain NuminaMath example for solver training."""
    answer_clean = extract_answer(answer)
    output = json.dumps({
        "reasoning": solution,
        "answer":    answer_clean,
    }, ensure_ascii=False)
    return {"messages": [
        {"role": "system",    "content": PLAIN_SOLVE_PROMPT},
        {"role": "user",      "content": f"Problem:\n{problem}"},
        {"role": "assistant", "content": output},
    ]}


def format_annotated(problem: str, solution: str, answer: str,
                     hints: str, facts: str) -> dict:
    """Format a CGM-annotated example — includes hints+facts in user turn."""
    hints_list = parse_list_field(hints)
    hints_str  = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hints_list)) if hints_list else ""
    facts_str  = str(facts).strip() if facts else ""

    user_parts = [f"Problem:\n{problem}"]
    if hints_str:
        user_parts.append(f"\nHints:\n{hints_str}")
    if facts_str:
        user_parts.append(f"\nKey facts:\n{facts_str}")

    answer_clean = extract_answer(answer)
    output = json.dumps({
        "reasoning": solution,
        "answer":    answer_clean,
    }, ensure_ascii=False)

    return {"messages": [
        {"role": "system",    "content": GUIDED_SOLVE_PROMPT},
        {"role": "user",      "content": "\n".join(user_parts)},
        {"role": "assistant", "content": output},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# 🔧  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 📦  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_plain_numina(cfg: StudentConfig) -> Dataset:
    """Pull NuminaMath from HuggingFace and format for plain solver training."""
    print("Loading NuminaMath from HuggingFace...")
    ds  = load_dataset("AI-MO/NuminaMath-CoT", split=cfg.numina_split)
    df  = ds.to_pandas()

    if cfg.filter_source and "source" in df.columns:
        df = df[df["source"].isin(cfg.allowed_sources)]

    df = df.iloc[:cfg.numina_subset].reset_index(drop=True)
    print(f"Plain NuminaMath: {len(df)} examples")

    rows = []
    for _, row in df.iterrows():
        rows.append(format_plain(
            str(row.get("problem", "")),
            str(row.get("solution", "")),
            str(row.get("answer",   "")),
        ))
    return Dataset.from_list(rows)


def load_annotated_numina(cfg: StudentConfig) -> Dataset:
    """Load silver-annotated NuminaMath and format with hints+facts."""
    print(f"Loading silver annotations from {cfg.silver_path}...")
    df = pd.read_parquet(cfg.silver_path)
    print(f"Annotated NuminaMath: {len(df)} examples")

    rows = []
    for _, row in df.iterrows():
        rows.append(format_annotated(
            str(row.get("problem",      "")),
            str(row.get("solution",     "")),
            str(row.get("answer",       "")),
            row.get("hints",  ""),
            str(row.get("facts",  "")),
        ))
    return Dataset.from_list(rows)


def load_data(cfg: StudentConfig):
    if cfg.use_annotated:
        ds = load_annotated_numina(cfg)
    else:
        ds = load_plain_numina(cfg)

    ds    = ds.shuffle(seed=cfg.seed)
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
    print(f"Student config : {cfg.config_name}")
    print(f"Model          : {cfg.model_name}")
    print(f"Data           : {'annotated' if cfg.use_annotated else 'plain NuminaMath'}")
    print(f"{'='*60}\n")

    random.seed(cfg.seed)

    train_ds, val_ds = load_data(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # ── Tokenise & mask ────────────────────────────────────────────────────
    IGNORE_INDEX         = -100
    # Detect chat template style
    sample_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True
    )
    # Support both Qwen (<|im_start|>) and Llama (<|start_header_id|>)
    if "<|im_start|>" in sample_text:
        assistant_header = "<|im_start|>assistant\n"
        end_token        = "<|im_end|>"
    else:
        # Llama-3 style
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        end_token        = "<|eot_id|>"

    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    im_end_id            = tokenizer.convert_tokens_to_ids(end_token)
    pad_id               = tokenizer.pad_token_id

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
    print(f"Sanity check OK: {unmasked} unmasked tokens in first example")

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

    print(f"\n🚀  Training student: {cfg.config_name}")
    trainer.train()

    out = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅  Saved to {out}")


if __name__ == "__main__":
    main()