import re

# LOG_FILE = "/data1/Yangzb/Wenzhi/Stack-Planner/logs/20250923190428.log"
LOG_FILE="/data1/Yangzb/Wenzhi/Stack-Planner/logs/Satastic/20250926114820.log"
# 匹配模式
request_start_pattern = "request param details: messages=[ChatMessage"
request_end_pattern = "任务完成，报告已保存:"
final_output_start_pattern = "final_report:"
rag_pattern = re.compile(r"Tool search_docs_tool returned: (.*)")


def parse_log_file(path):
    requests = []
    current_request = None
    capturing_output = False
    output_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # 请求开始
            if request_start_pattern in line:
                current_request = {
                    "request": line.strip(),
                    "agents": [],
                    "rag_calls": [],
                    "final_output": None,
                }
                capturing_output = False
                output_lines = []
                continue

            if not current_request:
                continue

            # 检测 RAG 调用
            if "Tool search_docs_tool returned:" in line:
                match = rag_pattern.search(line)
                if match:
                    current_request["rag_calls"].append(match.group(1).strip())

            # 检测最终输出开始
            if final_output_start_pattern in line:
                capturing_output = True
                output_lines = []
                continue

            # 检测请求结束
            if request_end_pattern in line:
                if capturing_output:
                    # 最终输出就是 final_report 到 request_end 之间的内容
                    current_request["final_output"] = "\n".join(output_lines).strip()
                    capturing_output = False
                requests.append(current_request)
                current_request = None
                continue

            # 收集最终输出（只在 final_report 之后、结束标志之前）
            if capturing_output:
                output_lines.append(line)

            # 收集调用的 Agents (只在 final_report 之前)
            if not capturing_output and "Agent" in line:
                current_request["agents"].append(line.strip())

    return requests


if __name__ == "__main__":
    results = parse_log_file(LOG_FILE)
    for idx, r in enumerate(results, 1):
        # print(f"===== 请求 {idx} =====")
        # print("请求参数:", r["request"])
        # print("调用的Agents:", r["agents"])
        print("是否调用RAG:", bool(r["rag_calls"]))
        # if r["rag_calls"]:
        #     print("RAG调用内容:", r["rag_calls"])
        # print("最终输出:", (r["final_output"][:200] + '...') if r["final_output"] else None)
        print("\n")
