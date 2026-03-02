宋政兴写webweaver 基线https://arxiv.org/abs/2509.13312论文如下
预计性能 RACE 42++
Prompt 需要自己写，论文自己似乎没提供

实现在这个文件夹下面写

检索 API 函数在from src.tools.bocha_search.web_search_en import web_search 这里。

api 是这个

# 本地部署
BASIC_MODEL:
  base_url: http://10.1.1.212:8000/v1
  model: Qwen3-32B
  api_key: sk-d47ad54165ee456093bc9ffd599e354e
  extra_body: {"enable_thinking": False}

REASONING_MODEL:
  base_url: http://10.1.1.212:8000/v1
  model: Qwen3-32B
  api_key: sk-d47ad54165ee456093bc9ffd599e354e
  extra_body: {"enable_thinking": True,"stream": True}

REPORT_MODEL:
  base_url: http://10.1.1.212:8000/v1
  model: Qwen3-32B
  api_key: sk-d47ad54165ee456093bc9ffd599e354e
  extra_body: {"stream": True,"enable_thinking": False}


  prompt 参考这个src/prompts/reporter_xxqg.md 来写一下，随便写一下就行了
任务先做这个 deep Research bench
数据集输入 evaluation/deep_research_bench/data/prompt_data/query.jsonl
数据集输出格式参考这个 evaluation/deep_research_bench/data/test_data/raw_data/FactStruct.jsonl

评估bash 参考 evaluation/deep_research_bench/run_benchmark.sh，可以让 YZB 来
