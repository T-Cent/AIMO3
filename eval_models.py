"""
eval_models.py
──────────────
Two-pass evaluation of all fine-tuned student models.
Outputs are saved in JSON Lines (.jsonl) format.
"""

import os
import re
import sys
import json
import argparse
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL  = "openai/gpt-oss-20b"
JUDGE_URL    = "http://localhost:8000/v1"

@dataclass
class EvalConfig:
    name:         str
    base_model:   str
    adapter_path: Optional[str]
    use_hints:    bool = False
    use_facts:    bool = False

MODEL_REGISTRY = {
    "base": EvalConfig(
        name="base", base_model=BASE_MODEL,
        adapter_path=None,
        use_hints=False, use_facts=False,
    ),
    "numina-base": EvalConfig(
        name="numina-base", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-plain/final_adapter",
        use_hints=False, use_facts=False,
    ),
    "llama-8b-d1": EvalConfig(
        name="llama-8b-d1", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d1/final_adapter",
        use_hints=False, use_facts=False,
    ),
    "llama-8b-d2": EvalConfig(
        name="llama-8b-d2", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d2/final_adapter",
        use_hints=False, use_facts=True,
    ),
    "llama-8b-d3": EvalConfig(
        name="llama-8b-d3", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d3/final_adapter",
        use_hints=True, use_facts=True,
    ),
    "llama-8b-d1-no-inc": EvalConfig(
        name="llama-8b-d1-no-inc", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d1-no-inc/final_adapter",
        use_hints=False, use_facts=False,
    ),
    "llama-8b-d2-no-inc": EvalConfig(
        name="llama-8b-d2-no-inc", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d2-no-inc/final_adapter",
        use_hints=False, use_facts=True,
    ),
    "llama-8b-d3-no-inc": EvalConfig(
        name="llama-8b-d3-no-inc", base_model=BASE_MODEL,
        adapter_path="./student-llama-8b-d3-no-inc/final_adapter",
        use_hints=True, use_facts=True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 📝  PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert competition mathematics solver.
Solve the problem step by step and give the final integer answer.
Put your final answer inside \\boxed{}.\
"""

def build_prompt(problem: str, hints=None, facts=None,
                 use_hints: bool = False, use_facts: bool = False) -> list:
    user = f"Problem:\n{problem}"
    if use_hints and hints:
        hints_list = parse_list_field(hints)
        if hints_list:
            hints_str = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hints_list))
            user += f"\n\nHints:\n{hints_str}"
    if use_facts and facts:
        facts_str = str(facts).strip()
        if facts_str:
            user += f"\n\nKey facts:\n{facts_str}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 🔧  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# ⚖️  LLM JUDGE
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert mathematical grader.
Compare the final answers of the two solutions below.
If they are mathematically equivalent, output ONLY the word CORRECT.
If they differ, output ONLY the word INCORRECT.

Problem:
{problem}

Reference answer:
{gold}

Student answer:
{pred}

Verdict:"""

def llm_judge(client: OpenAI, problem: str, pred: str, gold: str) -> bool:
    if not pred or not pred.strip():
        return False
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    problem=problem, gold=gold, pred=pred
                )
            }],
            temperature=1e-6,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content
        return "CORRECT" in (verdict or "").upper()
    except Exception as e:
        print(f"  Judge error: {e}")
        return False


def run_judge_pass(output_dir: str, models: list) -> pd.DataFrame:
    """
    Pass 2: load raw inference JSONL records, judge all rows, save scored JSONL files,
    return summary DataFrame.
    """
    client = OpenAI(base_url=JUDGE_URL, api_key="EMPTY")
    summary_rows = []

    for model_name in models:
        raw_path    = os.path.join(output_dir, f"{model_name}_raw.jsonl")
        scored_path = os.path.join(output_dir, f"{model_name}_scored.jsonl")

        if not os.path.exists(raw_path):
            print(f"  Skipping judge for {model_name} — no raw file found at {raw_path}")
            continue

        # Skip if already judged
        if os.path.exists(scored_path):
            print(f"  {model_name}: already judged, loading {scored_path}")
            records = []
            with open(scored_path, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            records = []
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            
            print(f"  Judging {model_name} ({len(records)} rows)...")
            
            with open(scored_path, "w", encoding="utf-8") as f_out:
                for item in tqdm(records, desc=f"judge/{model_name}"):
                    is_correct = llm_judge(
                        client,
                        str(item.get("problem", "")),
                        str(item.get("raw_output", "")),
                        str(item.get("gold", "")),
                    )
                    item["correct"] = is_correct
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            df = pd.DataFrame(records)

        n_correct = df["correct"].sum()
        n_total   = len(df)
        pass_at_1 = n_correct / max(1, n_total)
        print(f"  {model_name}: {n_correct}/{n_total} = {pass_at_1:.1%}")
        summary_rows.append({
            "model":   model_name,
            "correct": int(n_correct),
            "total":   n_total,
            "pass@1":  pass_at_1,
        })

    return pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 🤖  MODEL LOADING / UNLOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: EvalConfig):
    print(f"\n  Loading: {cfg.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if cfg.adapter_path is not None:
        print(f"  Adapter: {cfg.adapter_path}")
        model = PeftModel.from_pretrained(base, cfg.adapter_path)
    else:
        print("  Zero-shot base (no adapter)")
        model = base

    model.eval()
    return model, tokenizer


def unload_model(model):
    del model
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 🚀  INFERENCE PASS
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_pass(df: pd.DataFrame, output_dir: str,
                       models: list, max_new_tokens: int):
    """
    Pass 1: for each model, generate answers and save raw outputs to JSONL.
    Skips models whose raw JSONL already exists.
    """
    for model_name in models:
        raw_path = os.path.join(output_dir, f"{model_name}_raw.jsonl")

        if os.path.exists(raw_path):
            print(f"  {model_name}: raw output exists, skipping inference")
            continue

        cfg = MODEL_REGISTRY[model_name]
        model, tokenizer = load_model(cfg)
        
        count = 0
        with open(raw_path, "w", encoding="utf-8") as f_out:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"infer/{model_name}"):
                problem  = str(row.get("problem", ""))
                gold_raw = str(row.get("answer", row.get("solution", "")))
                gold     = extract_boxed(gold_raw) or gold_raw

                messages = build_prompt(
                    problem,
                    hints=row.get("hints"),
                    facts=row.get("facts"),
                    use_hints=cfg.use_hints,
                    use_facts=cfg.use_facts,
                )
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(
                    prompt_text, return_tensors="pt",
                    truncation=True, max_length=1024,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                )

                input_len  = inputs["input_ids"].shape[1]
                raw_output = tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

                out_item = {
                    "problem":    problem,
                    "gold":       gold,
                    "raw_output": raw_output,
                }
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                count += 1

        print(f"  {model_name}: saved {count} rows → {raw_path}")
        unload_model(model)


# ─────────────────────────────────────────────────────────────────────────────
# 📊  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary_df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"{'Model':<28} {'Correct':>8} {'Total':>7} {'Pass@1':>8}")
    print(f"{'-'*55}")
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<28} {int(r['correct']):>8} {int(r['total']):>7} {r['pass@1']:>7.1%}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ▶️  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data",       required=True)
    parser.add_argument("--models",          nargs="+",
                        default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--output_dir",      default="./eval_results")
    parser.add_argument("--limit",           type=int, default=None)
    parser.add_argument("--max_new_tokens",  type=int, default=512)
    parser.add_argument("--inference_only",  action="store_true",
                        help="Run inference only, skip judging")
    parser.add_argument("--judge_only",      action="store_true",
                        help="Run judging only, skip inference")
    args = parser.parse_args()

    # Validate model names
    invalid = [m for m in args.models if m not in MODEL_REGISTRY]
    if invalid:
        print(f"Unknown models: {invalid}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Load eval data
    print(f"Loading eval data from {args.eval_data}...")
    df = pd.read_csv(args.eval_data) if args.eval_data.endswith(".csv") \
        else pd.read_parquet(args.eval_data)
    if args.limit:
        df = df.head(args.limit)
    print(f"Evaluating on {len(df)} problems")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Pass 1: Inference ──────────────────────────────────────────────────
    if not args.judge_only:
        print(f"\n{'='*55}")
        print("PASS 1 — INFERENCE")
        print(f"{'='*55}")
        run_inference_pass(df, args.output_dir, args.models, args.max_new_tokens)

    # ── Pass 2: Judging ────────────────────────────────────────────────────
    if not args.inference_only:
        print(f"\n{'='*55}")
        print("PASS 2 — LLM JUDGE")
        print(f"{'='*55}")
        summary_df = run_judge_pass(args.output_dir, args.models)

        summary_path = os.path.join(args.output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print_summary(summary_df)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()