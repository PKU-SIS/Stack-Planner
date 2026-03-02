from ast import parse
from concurrent.futures import thread
from tkinter import NO
import httpx
import json
import os
import re
import argparse


def get_unique_output_path(base_path):
    """
    给定 base_path (如 /path/to/SP.jsonl)，
    如果该文件存在，则尝试 SP_2.jsonl, SP_3.jsonl, ...
    直到找到一个不存在的路径。
    """
    if not os.path.exists(base_path):
        return base_path

    dir_name = os.path.dirname(base_path)
    file_name = os.path.basename(base_path)

    # 分离主名和扩展名（支持 .jsonl, .txt 等）
    if "." in file_name:
        name_part, ext = os.path.splitext(file_name)
    else:
        name_part, ext = file_name, ""

    # 检查是否已经是带数字后缀的（可选：避免 SP_2_2.jsonl）
    # 这里简单处理：直接从 2 开始递增
    counter = 2
    while True:
        new_name = f"{name_part}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def get_latest_report_file(reports_dir):
    """返回 reports/ 中最新的 execution_report_*.json 文件路径"""
    files = [
        f
        for f in os.listdir(reports_dir)
        if f.startswith("execution_report_") and f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(
            f"{reports_dir}/ 目录中找不到任何 execution_report_*.json 文件！"
        )
    # 按文件名排序（时间戳在文件名中，字典序即时间序）
    latest = sorted(files)[-1]
    return os.path.join(reports_dir, latest)


def extract_queries(log_text):
    """
    返回一个列表：按顺序提取所有 query（可能多个）
    严格要求：
        1. 必须由 trigger 触发
        2. content 必须属于 role='user'
        3. 排除 system / assistant / reporter 内容
    """
    trigger_pattern = r"(zip_data': None|Starting DeerFlow API server on localhost)"
    # 限定必须在 role='user' 那一段
    content_pattern = r"role='user'.*content='([^']+)'"

    lines = log_text.splitlines()
    triggered = False
    queries = []

    for line in lines:

        # 触发条件：出现 zip_data 或 starting server
        if re.search(trigger_pattern, line):
            triggered = True
            continue

        # 处于触发状态，则检查是否是用户的 content
        if triggered:
            match = re.search(content_pattern, line)
            if match:
                queries.append(match.group(1).strip())

            # 无论是否命中，触发只生效一次
            triggered = False

    return queries


def extract_reports(log_text):
    """
    返回一个列表，可能包含多个 final_report
    final_report 开头：|final_report:
    最终结束：包含 "任务完成，报告已保存"
    """
    start_pattern = r"\|final_report:"
    end_pattern = r"任务完成，报告已保存"

    lines = log_text.splitlines()

    reports = []
    capturing = False
    buffer = []

    for line in lines:
        if not capturing and re.search(start_pattern, line):
            capturing = True
            idx = line.find("|final_report:")
            buffer.append(line[idx + len("|final_report:") :].strip())
            continue

        if capturing:
            if re.search(end_pattern, line):
                report_text = "\n".join(buffer).strip()

                # 清洗
                try:
                    report_text = extract_reporter_output(report_text)
                except Exception:
                    pass

                reports.append(report_text)
                buffer = []
                capturing = False
                continue

            buffer.append(line)

    return reports


# 提取最后答案的
# 这个似乎不太行
def extract_reporter_output(text):
    """
    从包含 reporter 调用及其输出的大文本中，
    提取 reporter 生成的正文文稿（去掉所有 JSON 噪声）
    """

    # === 1. 找到 reporter action block ===
    reporter_block = re.search(
        r'"agent_type"\s*:\s*"reporter".*?\}', text, flags=re.DOTALL
    )
    if not reporter_block:
        raise ValueError("未找到 reporter agent 调用区块")

    # === 2. reporter 块后面的自然语言内容（发言稿）开始位置 ===
    start_pos = reporter_block.end()

    # === 3. 截断到下一个 action/agent 块（表示输出结束）===
    end_match = re.search(
        r'(\n\s*\{?\s*"action"\s*:\s*"(finish|delegate)"|' r'"agent_type"\s*:)',
        text[start_pos:],
        flags=re.DOTALL,
    )
    if end_match:
        end_pos = start_pos + end_match.start()
    else:
        end_pos = len(text)

    raw_output = text[start_pos:end_pos].strip()

    # === 4. 清洗步骤：去掉开头的 { ===
    cleaned = re.sub(r"^\{+", "", raw_output).strip()

    # === 5. 清洗结尾尾巴：去掉任何看起来像 JSON key-value 的部分 ===
    # 去掉末尾以 {"key": ...} 形式的 JSON 残片
    cleaned = re.sub(
        r',?\s*"instruction"\s*:\s*".*?"\s*,?', "", cleaned, flags=re.DOTALL
    )
    cleaned = re.sub(
        r'"locale"\s*:\s*".*?"\s*\}?', "", cleaned, flags=re.DOTALL
    ).strip()

    # === 6. 最后再去掉一次可能残留的大括号 ===
    cleaned = cleaned.rstrip("{").strip()

    # === 7. 返回纯文本 ===
    return cleaned


# 用于缓存 event 数据
buffer = ""


def pretty_print_sheet(questions):
    type_labels = {"Select": "[单选]", "MultiSelect": "[多选]", "TextArea": "[填空]"}

    print("📝 写作助手答题卡\n")

    for idx, q in enumerate(questions, start=1):
        title = q["question"]
        q_type = q["type"]
        options = q["options"]

        # 打印题目编号和标题 + 类型
        print(f"{idx}. {title}？{type_labels.get(q_type, '[未知]')}")

        if q_type in ["Select", "MultiSelect"]:
            # 打印选项 A, B, C...
            for i, option in enumerate(options):
                letter = chr(65 + i)  # 65 = 'A'
                print(f"   {letter}. {option}")
        elif q_type == "TextArea":
            print("   （请在此处填写内容）")

        print()  # 空行分隔

    # 开始收集用户回答
    print("请逐条回答问题：")
    print("👉 选择题请输入选项字母（如 A 或 A、B），填空题直接写内容。")
    print("🔚 每行一个答案，输入完后输入 END 结束：\n")

    answers_parsed = []  # 存储结构化答案
    answer_lines = []
    answer = input().strip()

    while answer.upper() != "END":
        answer_lines.append(answer)
        answer = input().strip()

    # 将用户输入与题目一一对应解析
    for line, question in zip(answer_lines, questions):
        user_input = line.strip()
        q_type = question["type"]
        options = question["options"]
        parsed_answer = []

        if q_type in ["Select", "MultiSelect"]:
            # 清洗输入：支持 A、B 或 A,B 或 AB 等格式
            import re

            letters = re.split(r"[、，,\\s]+", user_input)  # 支持多种分隔符
            letters = [letter.strip().upper() for letter in letters if letter.strip()]

            for letter in letters:
                if len(letter) == 1 and letter.isalpha():
                    idx = ord(letter) - 65  # A->0, B->1
                    if 0 <= idx < len(options):
                        parsed_answer.append(options[idx])
                    else:
                        print(
                            f"⚠️ 选项 {letter} 超出范围（题目：{question['question']}），已忽略。"
                        )
                else:
                    print(f"⚠️ 无效选项格式：{letter}，已忽略。")

            # 单选只取第一个（可选策略）
            if q_type == "Select" and len(parsed_answer) > 1:
                print(
                    f"⚠️ 注意：'{question['question']}' 是单选题，仅保留第一个选项 '{parsed_answer[0]}'"
                )
                parsed_answer = [parsed_answer[0]]
            parsed_answer = "; ".join(parsed_answer)
        elif q_type == "TextArea":
            # 填空题直接使用用户输入文本（非选项）
            parsed_answer = user_input  # 可以是字符串，也可以存为 [user_input] 视需求
        else:
            parsed_answer = user_input

        answers_parsed.append(parsed_answer)
    return answers_parsed


NEED_RETRY = False


def process_event(event_type, event_data):
    """处理一个完整的 event"""

    if event_type == "message_chunk":
        content = event_data.get("content", "")
        print(content, end="", flush=True)
        return None
    elif event_type == "interrupt":
        thread_id = event_data.get("thread_id", "")
        content = event_data.get("content", "")
        question = event_data.get("question", "")
        # ========== ◆ 1. Outline 中断检测 ◆ ==========
        # Outline 不会给 question 字段；只会给 content 内包含 [OUTLINE]xxx[/OUTLINE]
        if "Outline" in content:
            print("\n\n================= 大纲确认 =================\n")
            outline_raw = event_data["outline"]
            print(outline_raw)
            print("\n-------------------------------------------")
            print("请输入你确认后的大纲：")
            print("（如无修改，直接按回车确认）")
            print("输入 SKIP 跳过大纲：")
            print("-------------------------------------------\n")

            user_text = input().strip()

            if user_text.upper() == "SKIP":
                feedback_content = "[SKIP]"
            else:
                # 用户未编辑 -> 直接使用系统大纲
                if not user_text:
                    feedback_content = "[CONFIRMED_OUTLINE]" + outline_raw
                else:
                    feedback_content = "[CONFIRMED_OUTLINE]" + user_text

            return {"thread_id": thread_id, "content": feedback_content}
        else:
            question = json.loads(question)
            if isinstance(question, dict):
                question = question.values()
            elif isinstance(question, list):
                question = question
            else:
                raise ValueError("Unexpected question format")
            print(f"\n\n---\n\n{content}\n\n---\n\n")
            answer_parsed = pretty_print_sheet(question)

            feedback = {
                "thread_id": thread_id,
                "content": "[FILLED_QUESTION]" + "\n".join(answer_parsed),
            }
            return feedback


def run_agent(url, data):
    """支持多轮 interrupt 的完整执行逻辑"""

    while True:
        NEED_RETRY = False
        buffer = ""
        # raw_output = ""

        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json=data) as response:

                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    return

                for chunk in response.iter_text():
                    buffer += chunk
                    # raw_output +=chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        # 解析 event 和 data
                        if line.startswith("event:"):
                            event_type = line.split(":", 1)[1].strip()

                        elif line.startswith("data:"):
                            data_str = line.split(":", 1)[1].strip()

                            try:
                                event_data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # === 交给 process_event() ===
                            feedback = process_event(event_type, event_data)

                            # 如果没有 interrupt 就继续读流
                            if feedback is None:
                                continue

                            # === 捕获 interrupt ===
                            data["interrupt_feedback"] = feedback["content"]
                            data["thread_id"] = feedback["thread_id"]
                            data["auto_accepted_plan"] = False
                            NEED_RETRY = True
                            break  # 跳出当前 for-chunk 循环

                # 如果收到 interrupt，那么需要重新发一次 POST
                if NEED_RETRY:
                    print("\n\n--- 根据你的回答，继续生成内容 ---\n")
                    continue

                # 没有 interrupt：流程结束
                print("\n\n=== Agent 完成所有步骤 ===\n")

                return  # raw_output#report_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run agent with streaming API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8513/api/chat/sp_stream",
        help="API URL，例如 http://localhost:8513/api/chat/sp_stream",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="evaluation/deep_research_bench/data/prompt_data/query.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument("--reports_dir", type=str, default="reports", help="日志目录")
    parser.add_argument(
        "--infer_num", type=int, default=10, help="一共要 infer 多少个样本"
    )
    parser.add_argument(
        "--graph-format",
        type=str,
        default="sp_test",
        choices=["sp", "xxqg", "sp_xxqg", "sp_test", "base", "FactStruct"],
        help="Graph format to use (default: 'sp')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/deep_research_bench/data/test_data/raw_data/SP.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--skip_exist", action="store_true", help="跳过已经生成过的样本"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("args", args)

    # ========== 新增：读取已有样本 ==========
    existing_prompts = set()
    if args.skip_exist and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    existing_prompts.add(item["prompt"])
                except:
                    pass

    print(f"已存在样本数量：{len(existing_prompts)}")

    # -------- 读取输入 jsonl --------
    with open(args.input_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    results = []
    count = 0
    for q in queries:

        # ========== 新增：跳过已生成样本 ==========
        if args.skip_exist and q["prompt"] in existing_prompts:
            print(f"[跳过] prompt 已存在：{q['prompt'][:30]}...")
            continue

        if count == args.infer_num:
            break

        count = count + 1

        content = q["prompt"]
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "resources": [],
            "thread_id": "__default__",  # todo
            "max_plan_iterations": 1,
            "max_step_num": 3,
            "max_search_results": 3,
            "auto_accepted_plan": True,  # todo
            "interrupt_feedback": "string",  # todo
            "mcp_settings": {},
            "enable_background_investigation": True,
            "graph_format": args.graph_format,  # "factstruct"#
        }

        run_agent(args.url, data)
        # ========== 关键修改：立即读取最新生成的 report 文件 ==========
        try:
            latest_report_path = get_latest_report_file(args.reports_dir)
            with open(latest_report_path, "r", encoding="utf-8") as rf:
                report_data = json.load(rf)
        except Exception as e:
            print(f"❌ 读取最新报告失败: {e}")
            continue
        # 提取所需字段
        final_report = report_data.get("final_report", "")
        research = report_data.get("research", [])

        # 可选：校验 user_query 是否匹配（防止并发错乱）
        reported_query = report_data.get("user_query", "").strip('"')
        if reported_query != content:
            print(
                f"⚠️  WARNING: reported query 不匹配！预期: {content[:30]}... 实际: {reported_query[:30]}..."
            )

        # 构造结果项
        result_item = {
            "id": q.get("id", count + 1),
            "prompt": reported_query,
            "article": final_report,
            "research": research,
        }
        results.append(result_item)

        # 写入输出文件（追加模式，避免丢失）
        with open(args.output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

        print(f"✅ 已写入结果 (共 {len(results)} 条)")

    print(f"全部完成！输出文件: {args.output_path} (新增 {len(results)} 条记录)")
