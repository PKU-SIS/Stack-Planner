# python eval_citation_async.py --subfolder [deepsearch_model_to_eval] --open_ai_model [llm_judge]
#export OPENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM" 
# export OPENAI_API_KEY=sk-d47ad54165ee456093bc9ffd599e354e 
# export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# export OPENAI_API_KEY=sk-d47ad54165ee456093bc9ffd599e354e 
# export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1


#deepseek的v2
export OPENAI_API_KEY="sk-550b5def797e452184c320b368a10989"
export OPENAI_BASE_URL="http://123.57.228.132:8286/api"

# deepseek-v3.2-20251201-160k-local
# qwen3-max
# python eval_quality_async.py \
#   --subfolder SP \
#   --open_ai_model qwen3-max
python eval_quality_async.py \
  --reports_path /data2/sp/Stack-Planner/evaluation/DeepResearchGYM/data/group_data/cx_group/deepsearch_benchmark/reports \
  --subfolder Prompt \
  --open_ai_model deepseek-v3.2-20251201-160k-local