import pandas as pd
import re
import json
from pathlib import Path
from openai import OpenAI

# ==========================================
# Configuration (Adjust these as needed)
# ==========================================
MODEL_NAME = "openai/gpt-oss-20b"
DATASET_PATH = "dataset.parquet"
# Sanitize the model name to prevent OS folder path errors
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_").replace("\\", "_")
OUTPUT_FILE = f"results_{SAFE_MODEL_NAME}.json" 
BASE_URL = "http://localhost:8000/v1"
API_KEY = "sk-local"
LIMIT_ROWS = None  # Running full dataset
CHECKPOINT_INTERVAL = 10 # Save every 10 problems

# ==========================================
# 1. Parsing Utilities
# ==========================================

def parse_list_column(text):
    if pd.isna(text) or str(text).strip() == "[null]":
        return []
    items = re.split(r'\n?\d+\.\s+', str(text))
    return [item.strip() for item in items if item.strip()]

# ==========================================
# 2. Prompt Generators
# ==========================================

def get_base_prompt(problem):
    return f"Solve the following math problem step-by-step. Put your final answer in a \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

def get_facts_prompt(problem, facts):
    facts_str = "\n".join([f"- {f}" for f in facts])
    return f"Here are some relevant mathematical facts:\n{facts_str}\n\nSolve the following math problem step-by-step. Put your final answer in a \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

def get_hints_prompt(problem, hints, attempt_num):
    current_hints = hints[:attempt_num]
    hints_str = "\n".join([f"Hint {i+1}: {h}" for i, h in enumerate(current_hints)])
    return f"Solve the following math problem step-by-step. Consider these hints:\n{hints_str}\n\nPut your final answer in a \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

def get_combined_prompt(problem, facts, hints, attempt_num):
    facts_str = "\n".join([f"- {f}" for f in facts])
    current_hints = hints[:attempt_num]
    hints_str = "\n".join([f"Hint {i+1}: {h}" for i, h in enumerate(current_hints)])
    return f"Relevant facts:\n{facts_str}\n\nHints:\n{hints_str}\n\nSolve the following math problem step-by-step. Put your final answer in a \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

# ==========================================
# 3. LLM Interaction & Judging via vLLM
# ==========================================

def generate_response(client, model_name, prompt):
    try:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=0.01,
            max_tokens=1024
        )
        content = response.choices[0].text
        return content.strip() if content is not None else ""
    except Exception as e:
        print(f"API Error during generation: {e}") 
        return f"ERROR: {e}"

def llm_judge_correctness(client, model_name, problem, generated_solution, ground_truth):
    if not generated_solution.strip() or "ERROR:" in generated_solution:
        return False

    judge_prompt = f"""You are an expert mathematical grader. 
Below is a math problem, the known correct solution, and a student's generated solution.
Compare the final mathematical answers of the two solutions.
If they are mathematically equivalent, output ONLY the word 'CORRECT'.
If they are different, output ONLY the word 'INCORRECT'.

Problem:
{problem}

Known Correct Solution:
{ground_truth}

Student's Solution:
{generated_solution}

Verdict:"""

    try:
        response = client.completions.create(
            model=model_name,
            prompt=judge_prompt,
            temperature=0.01, 
            max_tokens=10    
        )
        verdict = response.choices[0].text.strip().upper()
        return "CORRECT" in verdict
    except Exception as e:
        print(f"API Error during judging: {e}")
        return False

# ==========================================
# 4. Save Utility
# ==========================================

def save_checkpoint(results_data, filepath):
    """Safely handles directory creation and saves JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True) # Fixes the OSError
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

# ==========================================
# 5. Main Execution Loop
# ==========================================

def run_evaluation():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = pd.read_parquet(DATASET_PATH)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    if LIMIT_ROWS:
        df = df.head(LIMIT_ROWS)
        print(f"Limiting execution to the first {LIMIT_ROWS} rows.")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = []

    print(f"Starting ablation study for model: {MODEL_NAME} on {len(df)} problems.")
    
    for index, row in df.iterrows():
        problem = row['problem']
        target_solution = row['solution'] 
        
        hints = parse_list_column(row['hints'])
        facts = parse_list_column(row['facts'])

        # Store baseline data
        row_result = {
            "problem_idx": index,
            "problem_text": problem,
            "target_solution": target_solution,
            "model": MODEL_NAME,
        }

        # --- Strategy 1: Base Problem ---
        gen_base = generate_response(client, MODEL_NAME, get_base_prompt(problem))
        row_result["base_correct"] = llm_judge_correctness(client, MODEL_NAME, problem, gen_base, target_solution)
        row_result["base_output"] = gen_base

        # --- Strategy 2: Facts Only ---
        gen_facts = generate_response(client, MODEL_NAME, get_facts_prompt(problem, facts))
        row_result["facts_correct"] = llm_judge_correctness(client, MODEL_NAME, problem, gen_facts, target_solution)
        row_result["facts_output"] = gen_facts

        # --- Strategy 3: Iterative Hints ---
        row_result["hints_correct"] = False
        row_result["hints_needed"] = 0
        row_result["hints_history"] = [] # Store output for EVERY attempt
        
        for attempt in range(1, len(hints) + 1):
            gen_hints = generate_response(client, MODEL_NAME, get_hints_prompt(problem, hints, attempt))
            is_correct = llm_judge_correctness(client, MODEL_NAME, problem, gen_hints, target_solution)
            
            row_result["hints_history"].append({
                "attempt_num": attempt,
                "hints_provided": hints[:attempt],
                "output": gen_hints,
                "judged_correct": is_correct
            })
            
            if is_correct:
                row_result["hints_correct"] = True
                row_result["hints_needed"] = attempt
                break

        # --- Strategy 4: Iterative Hints + Facts ---
        row_result["combined_correct"] = False
        row_result["combined_hints_needed"] = 0
        row_result["combined_history"] = [] # Store output for EVERY attempt
        
        for attempt in range(1, len(hints) + 1):
            gen_comb = generate_response(client, MODEL_NAME, get_combined_prompt(problem, facts, hints, attempt))
            is_correct = llm_judge_correctness(client, MODEL_NAME, problem, gen_comb, target_solution)
            
            row_result["combined_history"].append({
                "attempt_num": attempt,
                "hints_provided": hints[:attempt],
                "facts_provided": facts,
                "output": gen_comb,
                "judged_correct": is_correct
            })
            
            if is_correct:
                row_result["combined_correct"] = True
                row_result["combined_hints_needed"] = attempt
                break

        results.append(row_result)
        print(f"Processed {index + 1}/{len(df)} | Base: {row_result['base_correct']} | Comb: {row_result['combined_correct']}")

        # Save Checkpoint
        if (index + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, OUTPUT_FILE)
            print(f"--> Checkpoint saved to {OUTPUT_FILE} (Processed {index + 1} items)")

    # Final Save
    save_checkpoint(results, OUTPUT_FILE)
    print(f"Run complete! Saved final results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()