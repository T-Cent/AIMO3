#!/bin/bash

# Define the models
MODELS=("Qwen/Qwen2.5-Math-1.5B" "Qwen/Qwen2.5-Math-7B" "meta-llama/Meta-Llama-3-8B")

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Starting vLLM for $MODEL..."
    echo "========================================"
    
    # Boot vLLM in the background. Using bfloat16 to leverage A100 tensor cores.
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.90 \
        --port 8000 &
    
    VLLM_PID=$!
    
    # Wait for the server to be ready (vLLM usually takes 30-60 seconds to load weights)
    echo "Waiting for model to load into VRAM..."
    sleep 60 

    echo "Running ablation logic..."
    # Run the python script
    python evaluate.py --dataset dataset.parquet --model_name $MODEL

    echo "Shutting down vLLM server..."
    # Gracefully kill the background process to free up VRAM for the next model
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    
    # Extra safety sleep to ensure CUDA context is fully destroyed
    sleep 10
done

echo "All evaluations complete!"