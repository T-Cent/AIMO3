"""
train_1.5b.py
───────────
Fine-tunes Qwen2.5-Math-1.5b-Instruct on CGM gold data (both annotation
and solving tasks) on a 2x A100 server using QLoRA.

Usage:
    # Single GPU (one A100):
    python train_1.5b.py

    # Both A100s:
    torchrun --nproc_per_node=2 train_1.5b.py

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes kaggle pandas pyarrow pydantic
"""

import os, json, re, random
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Kaggle credentials ─────────────────────────────────────────────────
    kaggle_username: str = "taraashmittal"
    kaggle_key:      str = "KGAT_41b62b22cffbb5508f52c09850ccf03c"
    kaggle_dataset:  str = "taraashmittal/aimo-3"   # dataset slug

    # ── Local data paths (set after download) ─────────────────────────────
    data_dir:       str = "./cgm_data"
    incorrect_file: str = "incorrect-approaches.parquet"
    correct_file:   str = "various-solution-approaches.parquet"

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir: str = "./cgm-1.5b-lora"

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r:     int = 64
    lora_alpha: int = 128

    # ── Training ───────────────────────────────────────────────────────────
    epochs:       int   = 5
    batch_size:   int   = 1
    grad_accum:   int   = 16     # effective batch = 32 across 2x A100
    lr:           float = 1e-4
    max_seq_len:  int   = 1536
    warmup_ratio: float = 0.05
    val_split:    float = 0.12
    seed:         int   = 42
    use_wandb:    bool  = False


cfg = Config()


# ─────────────────────────────────────────────────────────────────────────────
# 📥  KAGGLE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(cfg: Config):
    """Download CGM parquet files from Kaggle if not already present."""
    incorrect_path = os.path.join(cfg.data_dir, cfg.incorrect_file)
    correct_path   = os.path.join(cfg.data_dir, cfg.correct_file)

    if os.path.exists(incorrect_path) and os.path.exists(correct_path):
        print("Dataset already downloaded, skipping.")
        return incorrect_path, correct_path

    os.makedirs(cfg.data_dir, exist_ok=True)

    # Write kaggle.json credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": cfg.kaggle_username, "key": cfg.kaggle_key}, f)
    os.chmod(creds_path, 0o600)

    print(f"Downloading {cfg.kaggle_dataset} from Kaggle...")
    os.system(
        f"kaggle datasets download -d {cfg.kaggle_dataset} "
        f"--path {cfg.data_dir} --unzip"
    )

    # Find files (unzip may create a subdirectory)
    for root, _, files in os.walk(cfg.data_dir):
        for fname in files:
            if fname.endswith(".parquet"):
                print(f"  Found: {os.path.join(root, fname)}")

    # Resolve actual paths after unzip
    def find_file(name):
        for root, _, files in os.walk(cfg.data_dir):
            if name in files:
                return os.path.join(root, name)
        raise FileNotFoundError(f"{name} not found under {cfg.data_dir}")

    return find_file(cfg.incorrect_file), find_file(cfg.correct_file)


# ─────────────────────────────────────────────────────────────────────────────
# 📝  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

ANNOTATION_PROMPT = """\
You are CGM-Annotator, a structured mathematical annotation assistant.

Given a math problem and a solution approach (correct OR incorrect), \
produce a JSON annotation with EXACTLY these fields:

For a CORRECT approach:
{
  "is_correct": true,
  "tags": ["<topic>", ...],
  "answer": "<final answer, LaTeX if needed>",
  "hints": ["<hint 1>", "<hint 2>", ...],
  "facts": "<key mathematical facts or theorems needed>",
  "solution": "<complete step-by-step solution>",
  "verification": "<how to verify the answer>",
  "error_analysis": null
}

For an INCORRECT approach:
{
  "is_correct": false,
  "tags": [],
  "answer": null,
  "hints": [],
  "facts": null,
  "solution": "<the attempted (incorrect) solution>",
  "verification": null,
  "error_analysis": {
    "failure_point": "<where the approach breaks down>",
    "explanation": "<why it is wrong>"
  }
}

Return ONLY valid JSON. No markdown fences, no preamble, no commentary.\
"""

SOLVING_PROMPT = """\
You are an expert competition mathematics solver.
Solve the given problem step by step.
Output ONLY valid JSON in this exact format:
{"reasoning": "<full step-by-step solution>", "answer": <integer>}
The answer must be a non-negative integer.
No markdown, no extra text. Just the JSON.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# 🔧  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_tags(raw) -> List[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return [t.strip().strip("[]'\"") for t in raw.split(",") if t.strip()]
    return []


def parse_hints(raw) -> List[str]:
    if isinstance(raw, list):
        return raw
    return [
        h.strip().lstrip("0123456789. ")
        for h in str(raw).split("\n")
        if h.strip()
    ]


def make_message(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def format_correct(row: dict) -> dict:
    ann = {
        "is_correct":     True,
        "tags":           parse_tags(row.get("tags", [])),
        "answer":         str(row.get("answer", "")),
        "hints":          parse_hints(row.get("hints", "")),
        "facts":          str(row.get("facts", "")),
        "solution":       str(row.get("solution", "")),
        "verification":   str(row.get("verification", "")),
        "error_analysis": None,
    }
    return make_message(
        ANNOTATION_PROMPT,
        f"Problem:\n{row['problem']}",
        json.dumps(ann, ensure_ascii=False),
    )


def format_incorrect(row: dict) -> dict:
    ann = {
        "is_correct":     False,
        "tags":           [],
        "answer":         None,
        "hints":          [],
        "facts":          None,
        "solution":       str(row.get("attempt", "")),
        "verification":   None,
        "error_analysis": {
            "failure_point": str(row.get("failure", "")),
            "explanation":   str(row.get("explanation", "")),
        },
    }
    return make_message(
        ANNOTATION_PROMPT,
        f"Problem:\n{row['problem']}",
        json.dumps(ann, ensure_ascii=False),
    )


def format_solving(row: dict) -> dict:
    answer_str = str(row.get("answer", ""))
    match      = re.search(r"\\boxed\{([^}]+)\}", answer_str)
    answer_int = match.group(1).strip() if match else answer_str.strip()
    output     = json.dumps({
        "reasoning": str(row.get("solution", "")),
        "answer":    answer_int,
    }, ensure_ascii=False)
    return make_message(
        SOLVING_PROMPT,
        f"Problem:\n{row['problem']}",
        output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 📦  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(incorrect_path: str, correct_path: str, cfg: Config):
    df_inc = pd.read_parquet(incorrect_path)
    df_cor = pd.read_parquet(correct_path)
    print(f"Incorrect : {len(df_inc)} rows")
    print(f"Correct   : {len(df_cor)} rows")

    ds_inc   = Dataset.from_pandas(df_inc, preserve_index=False).map(
        format_incorrect, remove_columns=df_inc.columns.tolist())
    ds_cor   = Dataset.from_pandas(df_cor, preserve_index=False).map(
        format_correct,   remove_columns=df_cor.columns.tolist())
    ds_solve = Dataset.from_pandas(df_cor, preserve_index=False).map(
        format_solving,   remove_columns=df_cor.columns.tolist())

    combined = concatenate_datasets([ds_inc, ds_cor, ds_solve]).shuffle(seed=cfg.seed)
    print(f"Total examples after combining annotation + solving: {len(combined)}")

    split = combined.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    print(f"Train : {len(split['train'])}  |  Val : {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# 🤖  MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
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


def apply_lora(model, cfg: Config):
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
    random.seed(cfg.seed)

    # Download data
    incorrect_path, correct_path = download_dataset(cfg)

    # Data
    train_ds, val_ds = load_data(incorrect_path, correct_path, cfg)

    # Model
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Tokenise & mask
    IGNORE_INDEX         = -100
    assistant_header     = "<|im_start|>assistant\n"
    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    im_end_id            = tokenizer.convert_tokens_to_ids("<|im_end|>")
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

    # Sanity check
    unmasked = sum(1 for l in train_tok[0]["labels"] if l != IGNORE_INDEX)
    assert unmasked > 0, "All labels masked — check assistant header token IDs!"
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
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if cfg.use_wandb else "none",
        run_name="cgm-1.5b-lora",
        group_by_length=True,
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

    print("\n🚀  Training CGM-1.5b...")
    trainer.train()

    out = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"\n✅  Adapter saved to {out}")


if __name__ == "__main__":
    main()