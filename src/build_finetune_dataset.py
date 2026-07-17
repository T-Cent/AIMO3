"""
build_finetune_datasets.py
──────────────────────────
Builds three fine-tuning dataset variants from:
  - CGM gold correct variants  (parquet or CSV)
  - Silver annotations         (parquet, from annotate_numina.py)
  - CGM gold incorrect variants (parquet or CSV)

Output datasets (saved as parquet + jsonl):
  dataset1_plain.parquet          problem + solution only         (baseline)
  dataset2_facts.parquet          problem + facts + solution
  dataset3_hints_facts.parquet    problem + hints + facts + solution

Incorrect variants are appended to all three datasets at the same ratio.

Usage:
    python build_finetune_datasets.py \
        --gold_correct  ./cgm_correct.parquet \
        --silver        ./silver_annotations_cgm-ft-7b.parquet \
        --gold_incorrect ./cgm_incorrect.parquet \
        --output_dir    ./finetune_data

All arguments have defaults — edit the DEFAULTS block below if running
without CLI args.
"""

import os
import json
import re
import argparse
import pandas as pd
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS — edit here if not using CLI
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    gold_correct    = "./cgm_correct.parquet",
    silver          = "./silver_annotations_cgm-ft-7b.parquet",
    gold_incorrect  = "./cgm_incorrect.parquet",
    output_dir      = "./finetune_data",
    seed            = 42,
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS — one per condition
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PLAIN = """\
You are an expert competition mathematics solver.
Given a problem and a solution approach, solve it step by step.
Output your final answer as an integer inside \\boxed{}.\
"""

SYSTEM_FACTS = """\
You are an expert competition mathematics solver.
You are given a problem, key mathematical facts, and a solution approach.
Use the facts to structure your reasoning.
Output your final answer as an integer inside \\boxed{}.\
"""

SYSTEM_HINTS_FACTS = """\
You are an expert competition mathematics solver.
You are given a problem, ordered hints, key mathematical facts, and a solution approach.
Use the hints and facts to structure your reasoning.
Output your final answer as an integer inside \\boxed{}.\
"""

SYSTEM_INCORRECT = """\
You are an expert competition mathematics critic.
You are given a problem and an incorrect solution attempt.
Identify the precise failure point and explain why the approach is wrong.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    """Load parquet or CSV."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def parse_list_field(raw) -> list:
    """Coerce hints/tags stored as JSON string or list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [h.strip() for h in raw.split("\n") if h.strip()]
    return []


def extract_answer(raw) -> str:
    s = str(raw).strip()
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    return m.group(1).strip() if m else s


def safe_str(val, default="") -> str:
    if val is None or (isinstance(val, float)):
        return default
    s = str(val).strip()
    return s if s else default


# ─────────────────────────────────────────────────────────────────────────────
# ROW FORMATTERS  — one per dataset condition
# ─────────────────────────────────────────────────────────────────────────────

def fmt_plain(row) -> dict:
    """Dataset 1: problem + solution only."""
    problem  = safe_str(row.get("problem"))
    solution = safe_str(row.get("solution"))
    answer   = extract_answer(row.get("answer", ""))
    return {"messages": [
        {"role": "system",    "content": SYSTEM_PLAIN},
        {"role": "user",      "content": f"Problem:\n{problem}"},
        {"role": "assistant", "content": f"{solution}\n\n\\boxed{{{answer}}}"},
    ]}


def fmt_facts(row) -> dict:
    """Dataset 2: problem + facts + solution."""
    problem  = safe_str(row.get("problem"))
    solution = safe_str(row.get("solution"))
    answer   = extract_answer(row.get("answer", ""))
    facts    = safe_str(row.get("facts"))

    user = f"Problem:\n{problem}"
    if facts:
        user += f"\n\nKey facts:\n{facts}"

    return {"messages": [
        {"role": "system",    "content": SYSTEM_FACTS},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": f"{solution}\n\n\\boxed{{{answer}}}"},
    ]}


def fmt_hints_facts(row) -> dict:
    """Dataset 3: problem + hints + facts + solution."""
    problem  = safe_str(row.get("problem"))
    solution = safe_str(row.get("solution"))
    answer   = extract_answer(row.get("answer", ""))
    facts    = safe_str(row.get("facts"))
    hints    = parse_list_field(row.get("hints", []))

    user = f"Problem:\n{problem}"
    if hints:
        hints_str = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hints))
        user += f"\n\nHints:\n{hints_str}"
    if facts:
        user += f"\n\nKey facts:\n{facts}"

    return {"messages": [
        {"role": "system",    "content": SYSTEM_HINTS_FACTS},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": f"{solution}\n\n\\boxed{{{answer}}}"},
    ]}


def fmt_incorrect(row) -> dict:
    """Incorrect variant — same format across all three datasets."""
    problem = safe_str(row.get("problem"))

    # Support both CGM gold schema and error_analysis nested schema
    attempt = safe_str(row.get("solution", row.get("attempt", "")))

    if "error_analysis" in row and isinstance(row["error_analysis"], dict):
        failure_point = safe_str(row["error_analysis"].get("failure_point", ""))
        explanation   = safe_str(row["error_analysis"].get("explanation", ""))
    else:
        failure_point = safe_str(row.get("failure_point", ""))
        explanation   = safe_str(row.get("explanation", ""))

    assistant_content = (
        f"This solution is incorrect.\n\n"
        f"Failure point: {failure_point}\n\n"
        f"Explanation: {explanation}"
    )
    return {"messages": [
        {"role": "system",    "content": SYSTEM_INCORRECT},
        {"role": "user",      "content": f"Problem:\n{problem}\n\nAttempt:\n{attempt}"},
        {"role": "assistant", "content": assistant_content},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_correct_rows(gold_df: pd.DataFrame, silver_df: pd.DataFrame,
                       formatter) -> list:
    """Apply formatter to all correct rows from gold + silver."""
    rows = []
    for _, r in gold_df.iterrows():
        rows.append(formatter(r))
    for _, r in silver_df.iterrows():
        rows.append(formatter(r))
    return rows


def build_incorrect_rows(incorrect_df: pd.DataFrame) -> list:
    return [fmt_incorrect(r) for _, r in incorrect_df.iterrows()]


def save(rows: list, name: str, output_dir: str, seed: int):
    ds = Dataset.from_list(rows).shuffle(seed=seed)
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f"{name}.parquet")
    jsonl_path   = os.path.join(output_dir, f"{name}.jsonl")
    ds.to_parquet(parquet_path)
    with open(jsonl_path, "w") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(ds)} rows → {parquet_path}")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_correct",   default=DEFAULTS["gold_correct"])
    parser.add_argument("--silver",         default=DEFAULTS["silver"])
    parser.add_argument("--gold_incorrect", default=DEFAULTS["gold_incorrect"])
    parser.add_argument("--output_dir",     default=DEFAULTS["output_dir"])
    parser.add_argument("--seed",           default=DEFAULTS["seed"], type=int)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Loading data...")
    gold_correct  = load_df(args.gold_correct)
    silver        = load_df(args.silver)
    gold_incorrect = load_df(args.gold_incorrect)

    # Keep only is_correct == True rows from silver (quality guard)
    if "is_correct" in silver.columns:
        silver = silver[silver["is_correct"] == True].reset_index(drop=True)

    print(f"  Gold correct  : {len(gold_correct)} rows")
    print(f"  Silver        : {len(silver)} rows")
    print(f"  Gold incorrect: {len(gold_incorrect)} rows")

    incorrect_rows = build_incorrect_rows(gold_incorrect)
    print(f"  Incorrect formatted: {len(incorrect_rows)} rows")
    print(f"{'='*60}\n")

    print("Building datasets...")

    d1 = build_correct_rows(gold_correct, silver, fmt_plain)
    save(d1, "dataset1_plain_no_incorrect", args.output_dir, args.seed)

    # Dataset 2 — facts (no incorrects)
    d2 = build_correct_rows(gold_correct, silver, fmt_facts)
    save(d2, "dataset2_facts_no_incorrect", args.output_dir, args.seed)

    # Dataset 3 — hints + facts (no incorrects)
    d3 = build_correct_rows(gold_correct, silver, fmt_hints_facts)
    save(d3, "dataset3_hints_facts_no_incorrect", args.output_dir, args.seed)

    print(f"\n✅  All datasets saved to {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()