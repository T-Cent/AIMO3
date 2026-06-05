from datasets import load_dataset
import pandas as pd

ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
df = ds.to_pandas()

# Filter to same competition sources you trained on
allowed = {"amc_aime", "olympiads", "aops_forum", "math_competitions"}
df = df[df["source"].isin(allowed)].reset_index(drop=True)

# Skip the first 1000 (your training window) and take 50
eval_df = df.iloc[2000:2100][["problem", "solution"]].reset_index(drop=True)

# NuminaMath stores answer inside the solution as \boxed{}
eval_df.to_parquet("./aimo_held_out.parquet", index=False)
print(eval_df.head())