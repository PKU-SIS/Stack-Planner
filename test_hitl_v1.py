from ast import parse
from concurrent.futures import thread
from tkinter import NO
import httpx
import json
import os


# 暂时禁用代理设置以解决502错误
# os.environ["http_proxy"] = "http://localhost:8888"
# os.environ["https_proxy"] = "http://localhost:8888"
# os.environ["HTTP_PROXY"] = "http://localhost:8888"
# os.environ["HTTPS_PROXY"] = "http://localhost:8888"

url = "http://localhost:8082/api/chat/sp_stream"

content = """我需要一篇关于脱贫攻坚成果的讲话稿。内容安排上，先要突出全国打赢脱贫攻坚战的重大成就，再结合我县的实际情况进行阐述，并适当穿插一两个具体案例，例如某个村通过发展产业实现了致富，或者某个家庭在政策帮扶下生活明显改善。最后，要强调我们还要把脱贫与乡村振兴紧密衔接，巩固成果，防止返贫。"""

data = {
    "messages": [
        {
            "role": "user",
            "content": content,
            # '''
            # 习近平论中国梦有哪些内容，请你一定要搜索，不要生成网络引用，而是使用源文件名称
            # '''
        }
    ],
    "resources": [],
    "thread_id": "__default__",
    "max_plan_iterations": 1,
    "max_step_num": 3,
    "max_search_results": 3,
    "auto_accepted_plan": True,
    "interrupt_feedback": "string",
    "mcp_settings": {},
    "enable_background_investigation": True,
    "graph_format": "sp_xxqg",
}

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

    # 检查是否在交互式环境中
    import sys

    if sys.stdin.isatty():
        answer = input().strip()
        while answer.upper() != "END":
            answer_lines.append(answer)
            answer = input().strip()
    else:
        # 非交互式环境，使用默认答案
        print("非交互式环境，使用默认答案...")
        for question in questions:
            if question["type"] == "Select":
                answer_lines.append("A")  # 默认选择第一个选项
            else:
                answer_lines.append("默认答案")

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

        # answers_parsed.append({
        #     "question": question["question"],
        #     "type": q_type,
        #     "answer": parsed_answer  # list of strings (or string for TextArea)
        # })
        answers_parsed.append(parsed_answer)
    return answers_parsed


NEED_RETRY = False


def process_event(event_type, event_data):
    """处理一个完整的 event"""
    if event_type in ["message_chunk", "tool_calls", "tool_call_result"]:
        content = event_data.get("content", "")
        print(content, end="", flush=True)
        return None
    elif event_type == "interrupt":
        thread_id = event_data.get("thread_id", "")
        content = event_data.get("content", "")
        question = event_data.get("question", "")
        outline = event_data.get("outline", "")
        question = json.loads(question)
        outline = json.loads(outline)
        if isinstance(question, dict):
            question = question.values()
        elif isinstance(question, list):
            question = question
        else:
            raise ValueError("Unexpected question format")
        if isinstance(outline, dict):
            outline = outline.values()
        elif isinstance(outline, list):
            outline = outline
        else:
            raise ValueError("Unexpected outline format")
        print(f"\n\n---\n\n{content}\n\n---\n\n")
        answer_parsed = pretty_print_sheet(question)
        outline_parsed = pretty_print_sheet(outline)

        feedback = {
            "thread_id": thread_id,
            "content": "[FILLED_QUESTION]" + "\n".join(answer_parsed),
        }
        return feedback


with httpx.Client(timeout=None) as client:
    with client.stream("POST", url, json=data) as response:
        if response.status_code == 200:
            for chunk in response.iter_text():
                buffer += chunk
                # 按行分割
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
                            res = process_event(event_type, event_data)
                            if res is not None:
                                data["interrupt_feedback"] = res["content"]
                                data["thread_id"] = res["thread_id"]
                                data["auto_accepted_plan"] = False
                                NEED_RETRY = True
                        except json.JSONDecodeError:
                            pass  # 忽略无效 JSON
        else:
            print(f"Error: {response.status_code}")
            try:
                response.read()  # 读取流式响应内容
                print(response.text)
            except Exception as e:
                print(f"Error reading response: {e}")


if NEED_RETRY:
    print("\n\n---\n\n正在根据你的回答继续生成内容...\n\n---\n\n")
    buffer = ""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=data) as response:
            if response.status_code == 200:
                for chunk in response.iter_text():
                    buffer += chunk
                    # 按行分割
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
                                process_event(event_type, event_data)
                            except json.JSONDecodeError:
                                pass  # 忽略无效 JSON
            else:
                print(f"Error: {response.status_code}")
                try:
                    response.read()  # 读取流式响应内容
                    print(response.text)
                except Exception as e:
                    print(f"Error reading response: {e}")
