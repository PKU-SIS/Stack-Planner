# draft.py
import json
import re


def parse_llm_output(result: str) -> dict:
    """
    模拟你在系统中对 LLM 输出进行 JSON 提取和解析的逻辑
    """
    print("===== RAW LLM OUTPUT =====")
    print(result)
    print("==========================\n")

    match = re.search(r"\{[\s\S]*\}", result)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    json_text = match.group(0)

    print("===== EXTRACTED JSON =====")
    print(json_text)
    print("==========================\n")

    allocations = json.loads(json_text)
    return allocations


def main():
    # 模拟真实的 LLM 输出（与你日志里的基本一致）
    llm_output = """```json
    {
    "allocations": [
        {"node_id": "node_3", "word_limit": 200},
        {"node_id": "node_4", "word_limit": 250},
        {"node_id": "node_5", "word_limit": 180},
        {"node_id": "node_7", "word_limit": 300}
    ],
    "total_allocated": 930
    }

    """
    try:
        result = parse_llm_output(llm_output)
        print("✅ JSON PARSE SUCCESS")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print("❌ JSON PARSE FAILED")
        print(str(e))
    
if __name__ == "__main__":
    main()