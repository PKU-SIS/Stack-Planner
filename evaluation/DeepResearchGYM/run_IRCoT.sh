#!/bin/bash
# Run IRCoT baseline inference
# Input:  evaluation/DeepResearchGYM/queries/researchy_queries_sample_doc_click_100_fix.jsonl
# Output: evaluation/DeepResearchGYM/data/group_data/cx_group/deepsearch_benchmark/reports/IRCoT/IRCoT.jsonl
#   base_url: http://123.59.6.244:8000/v1
#   model: Qwen3-32B
#   api_key: sk-d47ad54165ee456093bc9ffd599e354e
#   extra_body: {"enable_thinking": False}
export OPENAI_API_KEY="sk-d47ad54165ee456093bc9ffd599e354e"
export OPENAI_BASE_URL="http://123.59.6.244:8000/v1"

PROJECT_ROOT="/data2/sp/Stack-Planner"

INPUT_PATH="evaluation/DeepResearchGYM/queries/researchy_queries_sample_doc_click_100_fix.jsonl"
OUTPUT_PATH="evaluation/DeepResearchGYM/data/group_data/cx_group/deepsearch_benchmark/reports/IRCoT/IRCoT.jsonl"

# Create output directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/$(dirname $OUTPUT_PATH)"

echo "Starting IRCoT inference..."
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"

python -u "$PROJECT_ROOT/baseline/IRCoT/run_inference.py" \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH"

echo "Done."
