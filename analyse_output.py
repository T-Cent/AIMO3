"""
analyse_outputs.py
──────────────────
Loads a parquet/CSV of model outputs and asks GPT-OSS-20B to:
  1. Provide commentary on the generated solution
  2. Give a final CORRECT / INCORRECT verdict

Input columns expected:
    problem       — the problem statement
    gold          — reference solution or answer
    raw_output    — the model's generated solution

Output:
    <input_file>_analysed.csv  — original columns + commentary + verdict

Usage:
    python analyse_outputs.py --input ./eval_results/llama-8b-d3_raw.csv
    python analyse_outputs.py --input ./eval_results/llama-8b-d3_raw.csv --limit 20
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_URL   = "http://localhost:8000/v1"
JUDGE_MODEL = "openai/gpt-oss-20b"

ANALYSIS_PROMPT = """\
You are an expert mathematical grader and tutor.

You will be given a math problem, a reference solution, and a student's \
generated solution. Your task is to:

1. Write a brief commentary (2-4 sentences) on the student's solution:
   - Did they use the right approach?
   - Where did they succeed or go wrong?
   - Is the reasoning sound?

2. Give a final verdict: CORRECT or INCORRECT, based on whether the final
   answer is mathematically equivalent to the reference.

Respond in exactly this format:
Commentary: <your commentary here>
Verdict: <CORRECT or INCORRECT>

---
Problem:
{problem}

Reference solution:
{gold}

Student solution:
{student}
---\
"""


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_row(client: OpenAI, problem: str, gold: str, student: str) -> dict:
    """Call GPT-OSS-20B for commentary + verdict on a single row."""
    if not student or not student.strip():
        return {"commentary": "No output generated.", "verdict": "INCORRECT"}

    prompt = ANALYSIS_PROMPT.format(
        problem=problem, gold=gold, student=student
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1e-6,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")
        return {"commentary": f"API error: {e}", "verdict": "ERROR"}

    # Parse response
    commentary = ""
    verdict    = "UNKNOWN"
    for line in content.strip().splitlines():
        if line.startswith("Commentary:"):
            commentary = line[len("Commentary:"):].strip()
        elif line.startswith("Verdict:"):
            raw = line[len("Verdict:"):].strip().upper()
            verdict = "CORRECT" if "CORRECT" in raw else "INCORRECT"

    # Fallback: scan full text for verdict if parsing failed
    if verdict == "UNKNOWN":
        upper = content.upper()
        if "CORRECT" in upper:
            verdict = "CORRECT"
        else:
            verdict = "INCORRECT"

    return {"commentary": commentary, "verdict": verdict}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Path to CSV or parquet with problem/gold/raw_output columns")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit number of rows (for testing)")
    parser.add_argument("--judge_url",  default=JUDGE_URL)
    parser.add_argument("--judge_model",default=JUDGE_MODEL)
    args = parser.parse_args()

    # Load input
    print(f"Loading {args.input}...")
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".jsonl") or args.input.endswith(".json"):
        df = pd.read_json(args.input, lines=True)
    else:
        df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"Analysing {len(df)} rows\n")

    client = OpenAI(base_url=args.judge_url, api_key="EMPTY")

    commentaries = []
    verdicts     = []
    correct_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analysing"):
        result = analyse_row(
            client,
            problem = str(row.get("problem",    "")),
            gold    = str(row.get("gold",        "")),
            student = str(row.get("raw_output",  "")),
        )
        commentaries.append(result["commentary"])
        verdicts.append(result["verdict"])
        if result["verdict"] == "CORRECT":
            correct_count += 1

    df["commentary"] = commentaries
    df["verdict"]    = verdicts

    # Save
    base     = os.path.splitext(args.input)[0]
    out_path = f"{base}_analysed.csv"
    df.to_csv(out_path, index=False)

    # Summary
    total    = len(df)
    pass_at1 = correct_count / max(1, total)
    print(f"\n{'='*50}")
    print(f"Correct : {correct_count} / {total}")
    print(f"Pass@1  : {pass_at1:.1%}")
    print(f"Saved to: {out_path}")
    print(f"{'='*50}\n")

    # Print a few examples
    print("Sample analyses:\n")
    for i, row in df.head(3).iterrows():
        print(f"[{i+1}] Verdict: {row['verdict']}")
        print(f"     Commentary: {row['commentary']}")
        print(f"     Gold:    {str(row.get('gold',''))[:80]}...")
        print(f"     Student: {str(row.get('raw_output',''))[:80]}...")
        print()


if __name__ == "__main__":
    main()