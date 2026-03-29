#!/bin/bash

# Python virtual environment activation (if needed)
# source /path/to/your/venv/bin/activate  # Uncomment this if you use a virtual environment

# Set up the necessary environment variables
# export OPENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM" 
# export OPENAI_BASE_URL="http://162.105.88.35:3000/api"  # Or the URL of your custom server
#deepseek的
# export OPENAI_API_KEY="sk-1e920ed44c6d46efaf95c9e924b5a0fa"
export OPENAI_API_KEY="sk-d47ad54165ee456093bc9ffd599e354e"
# export OPENAI_API_KEY="sk-550b5def797e452184c320b368a10989"
#deepseek的
# export OPENAI_BASE_URL="https://api.deepseek.com/v1" #  # Or the URL of your custom server
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# export OPENAI_BASE_URL="http://123.57.228.132:8286/api"



# Running the Python evaluation script
python -m evals.deep_research_pairwise_evals \
  --input-data datasets/DeepConsult/responses_RAG_vs_Baseline.csv \
  --output-dir datasets/results \
  --model qwen3-max \
  --num-workers 4 \
  --metric-num-workers 3 \
  --metric-num-trials 3

  #deepseek-v3.2-20251201-160k-local \
  #qwen3-max
  #deepseek-chat