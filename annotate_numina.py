"""
annotate_numina.py
──────────────────
Loads a CGM fine-tuned model (1.5B or 7B) and uses it to annotate a subset
of NuminaMath, producing a silver-annotation parquet ready for the next
fine-tuning stage.

Usage (notebook-friendly — just edit NuminaConfig below):
    %run annotate_numina.py

Or from terminal:
    python annotate_numina.py

Output:
    silver_annotations_<model_tag>.parquet
    silver_annotations_<model_tag>_failed.parquet   ← rows that failed to parse
"""

import os
import json
import re
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, field_validator, ValidationError
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIG — edit this block only
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NuminaConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    # Set use_lora=False for base models (no fine-tuning)
    # Set use_lora=True  for CGM fine-tuned models
    # Set use_gptoss=True to use GPT-OSS via local vLLM server
    base_model:   str  = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    adapter_path: str  = "./cgm-1.5b-lora/final_adapter"
    model_tag:    str  = "cgm-ft-1.5b"  # used in output filenames
    use_lora:     bool = True            # False = base model, no adapter
    use_gptoss:   bool = False           # True = use GPT-OSS via vLLM API

    # ── GPT-OSS settings (only used when use_gptoss=True) ──────────────────
    gptoss_base_url: str = "http://localhost:8000/v1"  # vLLM server
    gptoss_model:    str = "gpt-oss-120b"              # or gpt-oss-20b
    gptoss_api_key:  str = "EMPTY"                     # vLLM default

    # ── NuminaMath ─────────────────────────────────────────────────────────
    numina_split:    str   = "train"
    numina_subset:   int   = 1000   # 10k entries
    numina_offset:   int   = 0       # skip first N (useful to resume)
    filter_source:   bool  = True
    allowed_sources: tuple = (
        "amc_aime", "olympiads", "aops_forum", "math_competitions"
    )

    # ── Generation ─────────────────────────────────────────────────────────
    max_new_tokens: int   = 1024
    batch_size:     int   = 4     # reduce to 2 if OOM; ignored for GPT-OSS
    temperature:    float = 0.1
    do_sample:      bool  = False

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "."
    save_every: int = 50
    # Save an extra snapshot once this many good rows have been produced
    save_after: int = 0

    # ── Quality filters ─────────────────────────────────────────────────────
    require_valid_json:     bool = True
    require_correct_true:   bool = True
    require_nonempty_hints: bool = True
    require_answer_present: bool = True


# ── Preset configs for all 6 annotators — just pick one ──────────────────────

CONFIGS = {
    # Base models (no fine-tuning)
    "base-1.5b": NuminaConfig(
        base_model   = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        model_tag    = "base-1.5b",
        use_lora     = False,
        batch_size   = 16,
    ),
    "base-7b": NuminaConfig(
        base_model   = "Qwen/Qwen2.5-Math-7B-Instruct",
        model_tag    = "base-7b",
        use_lora     = False,
        batch_size   = 8,
    ),
    # CGM fine-tuned models
    "cgm-ft-1.5b": NuminaConfig(
        base_model   = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        adapter_path = "./cgm-1.5b-lora/final_adapter",
        model_tag    = "cgm-ft-1.5b",
        use_lora     = True,
        batch_size   = 16,
    ),
    "cgm-ft-7b": NuminaConfig(
        base_model   = "Qwen/Qwen2.5-Math-7B-Instruct",
        adapter_path = "./cgm-7b-lora/final_adapter",
        model_tag    = "cgm-ft-7b",
        use_lora     = True,
        batch_size   = 8,
    ),
    # GPT-OSS (served locally via vLLM)
    "gptoss-20b": NuminaConfig(
        model_tag    = "gptoss-20b",
        use_lora     = False,
        use_gptoss   = True,
        gptoss_model = "openai/gpt-oss-20b",
        batch_size   = 1,   # vLLM handles its own batching
    ),
    "gptoss-120b": NuminaConfig(
        model_tag    = "gptoss-120b",
        use_lora     = False,
        use_gptoss   = True,
        gptoss_model = "gpt-oss-120b",
        batch_size   = 1,
    ),
    "llama-8b-silver": NuminaConfig(
        base_model   = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path = "./llama-8b-silver/",
        model_tag    = "llama-8b-silver",
        use_lora     = True,
        batch_size   = 2,
    ),
}

import sys

def get_config() -> NuminaConfig:
    valid = list(CONFIGS.keys())
    if len(sys.argv) < 2:
        print(f"Usage: python annotate_numina.py <annotator>")
        print(f"Available: {', '.join(valid)}")
        sys.exit(1)
    key = sys.argv[1]
    if key not in CONFIGS:
        print(f"Unknown annotator '{key}'. Choose from: {', '.join(valid)}")
        sys.exit(1)
    return CONFIGS[key]

config = get_config()
print(f"Running annotator: {config.model_tag}")


# ─────────────────────────────────────────────────────────────────────────────
# 📝  SAME SYSTEM PROMPT AS TRAINING (must match exactly)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
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


# ─────────────────────────────────────────────────────────────────────────────
# 🔧  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Pydantic models for validated silver annotations ──────────────────────

class ErrorAnalysis(BaseModel):
    failure_point: str = ""
    explanation:   str = ""

class CGMAnnotation(BaseModel):
    is_correct:     bool
    tags:           List[str]               = []
    answer:         Optional[str]           = None
    hints:          List[str]               = []
    facts:          Optional[str]           = None
    solution:       str                     = ""
    verification:   Optional[str]           = None
    error_analysis: Optional[ErrorAnalysis] = None

    @field_validator("tags", "hints", mode="before")
    @classmethod
    def coerce_list(cls, v):
        """Handle tags/hints stored as a JSON string instead of a list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v.replace("'", '"'))
                return parsed if isinstance(parsed, list) else [v]
            except Exception:
                return [v] if v.strip() else []
        return v or []

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


def strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def try_extract_json(text: str) -> Optional[str]:
    """
    Multi-strategy JSON extraction:
    1. Direct parse after stripping fences
    2. Find outermost { ... } block (handles preamble/postamble the model adds)
    3. Return None if all strategies fail
    """
    cleaned = strip_fences(text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def parse_and_validate(text: str) -> Optional[CGMAnnotation]:
    """
    Parse raw model output into a validated CGMAnnotation.
    Returns None only if both JSON extraction AND Pydantic validation fail.
    Uses two-pass strategy: strict parse first, forgiving coercion second.
    """
    json_str = try_extract_json(text)
    if json_str is None:
        return None
    try:
        return CGMAnnotation.model_validate_json(json_str)
    except ValidationError:
        try:
            raw = json.loads(json_str)
            return CGMAnnotation.model_validate(raw)
        except Exception:
            return None


def passes_quality_filters(ann: Optional[CGMAnnotation], cfg: NuminaConfig) -> bool:
    if ann is None:
        return False
    if cfg.require_correct_true and not ann.is_correct:
        return False
    if cfg.require_nonempty_hints and not ann.hints:
        return False
    if cfg.require_answer_present and not ann.answer:
        return False
    return True




def build_prompt(problem: str, solution: str, tokenizer) -> str:
    """Build a single inference prompt from a NuminaMath row."""
    user_text = f"Problem:\n{problem}\n\nProposed solution/variant:\n{solution}"
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n{user_text}"


# ─────────────────────────────────────────────────────────────────────────────
# 🤖  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: NuminaConfig):
    """Load model based on config: base, LoRA fine-tuned, or GPT-OSS (returns None, None)."""
    if cfg.use_gptoss:
        print(f"Using GPT-OSS via vLLM at {cfg.gptoss_base_url} — no local model loaded")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batch generation

    print(f"Loading base model: {cfg.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if cfg.use_lora:
        print(f"Loading LoRA adapter: {cfg.adapter_path}")
        model = PeftModel.from_pretrained(base, cfg.adapter_path)
    else:
        print("Using base model without adapter")
        model = base

    model.eval()
    print("Model ready.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 📦  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_numina(cfg: NuminaConfig) -> pd.DataFrame:
    print("Loading NuminaMath from HuggingFace...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split=cfg.numina_split)
    df = ds.to_pandas()
    print(f"Full dataset size: {len(df)}")

    if cfg.filter_source and "source" in df.columns:
        df = df[df["source"].isin(cfg.allowed_sources)]
        print(f"After source filter: {len(df)}")

    df = df.iloc[cfg.numina_offset : cfg.numina_offset + cfg.numina_subset]
    df = df.reset_index(drop=True)
    print(f"Using {len(df)} problems (offset={cfg.numina_offset})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 🚀  ANNOTATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def annotate_batch(prompts: list, model, tokenizer, cfg: NuminaConfig) -> list:
    """Run inference on a batch of prompts, return list of raw output strings."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,   # input truncation
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    decoded = [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]
    return decoded


def annotate_single_gptoss(problem: str, solution: str, cfg: NuminaConfig) -> str:
    """Call GPT-OSS served locally via vLLM OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  — needed for GPT-OSS inference")

    client = OpenAI(base_url=cfg.gptoss_base_url, api_key=cfg.gptoss_api_key)
    response = client.chat.completions.create(
        model=cfg.gptoss_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Problem:\n{problem}\n\nProposed solution/variant:\n{solution}"},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )
    return response.choices[0].message.content.strip()


def run_annotation(cfg: NuminaConfig = config):
    df = load_numina(cfg)
    model, tokenizer = load_model(cfg)

    out_path    = os.path.join(cfg.output_dir,
                               f"silver_annotations_{cfg.model_tag}.parquet")
    failed_path = os.path.join(cfg.output_dir,
                               f"silver_annotations_{cfg.model_tag}_failed.parquet")

    good_rows   = []
    failed_rows = []

    # Resume from checkpoint if output already exists
    start_idx = 0
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        start_idx = len(existing)
        good_rows = existing.to_dict("records")
        print(f"Resuming from row {start_idx}")
    # If we've already passed the `save_after` threshold, skip the one-time snapshot
    saved_after = start_idx >= cfg.save_after if cfg.save_after else False

    problems  = df["problem"].tolist()
    # NuminaMath-CoT stores the solution in "solution" column
    solutions = df.get("solution", df.get("answer", [""] * len(df))).tolist()

    n = len(df)
    batch_size = cfg.batch_size

    for i in tqdm(range(start_idx, n, batch_size), desc="Annotating"):
        batch_probs = problems[i : i + batch_size]
        batch_sols  = solutions[i : i + batch_size]

        try:
            if cfg.use_gptoss:
                raw_outputs = [
                    annotate_single_gptoss(p, s, cfg)
                    for p, s in zip(batch_probs, batch_sols)
                ]
            else:
                prompts = [
                    build_prompt(p, s, tokenizer)
                    for p, s in zip(batch_probs, batch_sols)
                ]
                raw_outputs = annotate_batch(prompts, model, tokenizer, cfg)
        except Exception as e:
            print(f"Batch {i} failed ({e}), skipping...")
            for p, s in zip(batch_probs, batch_sols):
                failed_rows.append({"problem": p, "solution": s,
                                    "raw_output": str(e), "fail_reason": "runtime_error"})
            continue

        for j, (prob, sol, raw) in enumerate(zip(batch_probs, batch_sols, raw_outputs)):
            ann = parse_and_validate(raw)

            if not passes_quality_filters(ann, cfg):
                reason = "invalid_json" if ann is None else "quality_filter"
                failed_rows.append({"problem": prob, "solution": sol,
                                    "raw_output": raw, "fail_reason": reason})
                continue

            good_rows.append({
                "problem":      prob,
                "solution":     ann.solution or sol,
                "hints":        json.dumps(ann.hints,  ensure_ascii=False),
                "facts":        ann.facts        or "",
                "verification": ann.verification or "",
                "answer":       ann.answer       or "",
                "tags":         json.dumps(ann.tags, ensure_ascii=False),
                "is_correct":   ann.is_correct,
                "raw_output":   raw,
                "source":       "numina_silver",
                "annotator":    cfg.model_tag,
            })

        # Checkpoint
        if (i + batch_size) % cfg.save_every == 0 and good_rows:
            pd.DataFrame(good_rows).to_parquet(out_path, index=False)
            print(f"  Checkpoint saved: {len(good_rows)} good rows so far")

        # One-time snapshot: save a copy once we have produced >= `save_after` good rows
        if (not saved_after) and cfg.save_after and len(good_rows) >= cfg.save_after:
            snapshot_path = os.path.join(cfg.output_dir,
                                         f"silver_annotations_{cfg.model_tag}_{cfg.save_after}.parquet")
            snapshot_failed = os.path.join(cfg.output_dir,
                                           f"silver_annotations_{cfg.model_tag}_failed_{cfg.save_after}.parquet")
            pd.DataFrame(good_rows).to_parquet(snapshot_path, index=False)
            # Handle empty failed_rows when saving snapshot
            if failed_rows:
                pd.DataFrame(failed_rows).to_parquet(snapshot_failed, index=False)
            else:
                failed_schema = {
                    "problem": str,
                    "solution": str,
                    "raw_output": str,
                    "fail_reason": str,
                }
                pd.DataFrame({k: pd.Series(dtype=v) for k, v in failed_schema.items()}).to_parquet(snapshot_failed, index=False)
            print(f"  Snapshot saved at {len(good_rows)} rows -> {snapshot_path}")
            saved_after = True

    # Final save
    # Define schema to handle empty DataFrames properly
    good_schema = {
        "problem": str,
        "solution": str,
        "hints": str,
        "facts": str,
        "verification": str,
        "answer": str,
        "tags": str,
        "is_correct": bool,
        "raw_output": str,
        "source": str,
        "annotator": str,
    }
    failed_schema = {
        "problem": str,
        "solution": str,
        "raw_output": str,
        "fail_reason": str,
    }

    if good_rows:
        good_df = pd.DataFrame(good_rows)
    else:
        good_df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in good_schema.items()})

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
    else:
        failed_df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in failed_schema.items()})

    good_df.to_parquet(out_path, index=False)
    failed_df.to_parquet(failed_path, index=False)

    print(f"\n✅  Done.")
    print(f"   Good annotations : {len(good_df)}")
    print(f"   Failed / filtered: {len(failed_df)}")
    print(f"   Pass rate        : {len(good_df) / max(1, len(good_df) + len(failed_df)):.1%}")
    print(f"   Saved to         : {out_path}")

    return good_df, failed_df


# ─────────────────────────────────────────────────────────────────────────────
# ▶️  RUN
# ─────────────────────────────────────────────────────────────────────────────

good_df, failed_df = run_annotation(config)