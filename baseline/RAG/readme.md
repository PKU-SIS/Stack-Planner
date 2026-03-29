万莘菲写 web_search
预计性能 RACE
Prompt 需要自己写
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

  检索生成 query 的 prompt 参考 src/prompts/researcher_web.md(YZB 觉得价值不大)
  prompt 参考这个src/prompts/reporter_xxqg.md 来写一下，随便写一下就行了
任务先做这个 deep Research bench
数据集输入 evaluation/deep_research_bench/data/prompt_data/query.jsonl
数据集输出格式参考这个 evaluation/deep_research_bench/data/test_data/raw_data/Research.jsonl

评估bash 参考 evaluation/deep_research_bench/run_benchmark.sh，可以让 YZB 来

输入数据结构
{"id": 10, "topic": "Science & Technology", "language": "zh", "prompt": "在800V高压/碳化硅电驱/固态电池/分布式驱动等技术迭代加速的窗口期，如何构建覆盖研发制造-使用场景-残值管理的评估体系，量化不同动力系统技术路线（纯电/增程/插混/氢燃料+集中式驱动/分布式驱动）的商业化临界点？"}
输出数据结构
return {
    "id": task["id"],
    "prompt": query_content,
    "article": generated_article,
    "research": {"1": {"type": "page", "title": "投资者提问:具身智能是当前机器人领域的热门话题,您认为公司在此方面有哪些创...机器人_新浪财经_新浪网", "url": "https://finance.sina.com.cn/stock/relnews/dongmiqa/2023-05-30/doc-imyvqmit4396728.shtml", "content": "(机器人SZ300024): 您好,此行业尚...。"}, "2": {"type": "page", "title": "具身智能产业深度: 技术模型分析、市场展望、相关产业及公司深度梳理-230602(17页).pdf-在线下载-三个皮匠报告", "url": "https://www.sgpjbg.com/bgdown/128173.html", "content": "具身智能产业深度: 技术模型分析、市场展望、相关产业及....."}}} 
}