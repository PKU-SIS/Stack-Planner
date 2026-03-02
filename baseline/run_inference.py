import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================
# 师兄要求的本地部署配置
API_CONFIG = {
    "base_url": "http://10.1.1.212:8000/v1",
    "api_key": "sk-d47ad54165ee456093bc9ffd599e354e",
    "model": "Qwen3-32B",
}

# 路径配置
PATHS = {
    "prompt_file": "baseline/Prompt/deep_research.md",
    "query_file": "evaluation/deep_research_bench/data/prompt_data/query.jsonl",
    # 输出位置：deepresearch_bench 默认读取 data/test_data/raw_data 下的文件
    "output_file": "evaluation/deep_research_bench/data/test_data/raw_data/Qwen3-32B.jsonl"
}
# ===========================================

def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def process_single_task(client, task, prompt_template):
    """处理单个任务：将Query填入Prompt并调用模型"""
    try:
        query_content = task.get("prompt", "")
        lang = task.get("language", "zh")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 简单的模板替换
        system_prompt = prompt_template.replace("{{ CURRENT_TIME }}", current_time)
        system_prompt = system_prompt.replace("{{locale}}", lang)
        
        # 组装完整的 Prompt
        # 注意：由于这是基线，没有外挂搜索(RAG)，我们让模型基于自身知识库回答
        # 为了防止模型因为"No provided info"而拒绝回答，我们在User Prompt里稍微引导一下
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Topic: {query_content}\n\nPlease write a comprehensive report on this topic based on your internal knowledge. Ignore the citation constraints if no external documents are provided, but maintain the report structure."}
        ]

        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body={"enable_thinking": False} # 根据readme配置
        )
        
        generated_article = response.choices[0].message.content

        return {
            "id": task["id"],
            "prompt": query_content,
            "article": generated_article,
            "model": API_CONFIG["model"]
        }

    except Exception as e:
        print(f"Error processing task {task.get('id')}: {e}")
        return None

def main():
    # 1. 初始化客户端
    client = OpenAI(
        base_url=API_CONFIG["base_url"],
        api_key=API_CONFIG["api_key"]
    )
    
    # 2. 准备数据
    print(f"Loading Prompt from {PATHS['prompt_file']}...")
    prompt_template = load_file(PATHS['prompt_file'])
    
    print(f"Loading Queries from {PATHS['query_file']}...")
    queries = load_jsonl(PATHS['query_file'])
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(PATHS["output_file"]), exist_ok=True)
    
    # 3. 执行推理 (使用多线程加速)
    results = []
    print(f"Starting inference with model {API_CONFIG['model']}...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_task, client, task, prompt_template) for task in queries]
        
        for future in tqdm(futures, total=len(queries), desc="Generating Reports"):
            result = future.result()
            if result:
                results.append(result)
    
    # 4. 保存结果
    print(f"Saving {len(results)} results to {PATHS['output_file']}...")
    with open(PATHS["output_file"], 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done! Inference completed.")

if __name__ == "__main__":
    main()
