from ast import parse
from concurrent.futures import thread
from tkinter import NO
import httpx
import json
import random
import time
import random
import string
random.seed(time.time()) # 用当前时间作为随机种子


def flatten_questions(qs):
    flat = []
    for q in qs:
        if isinstance(q, list):
            flat.extend(flatten_questions(q))  # 递归展开
        elif isinstance(q, dict):
            flat.append(q)
    return flat


def main(content):
    url = "http://localhost:8513/api/chat/sp_stream"


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
        "thread_id": "__default__", # todo
        "max_plan_iterations": 1,
        "max_step_num": 3,
        "max_search_results": 3,
        "auto_accepted_plan": True, # todo 
        "interrupt_feedback": "string", # todo
        "mcp_settings": {},
        "enable_background_investigation": True,
        "graph_format": "sp_xxqg", #todo
    }

    # 用于缓存 event 数据
    buffer = ""

    def pretty_print_sheet_before(questions):
        type_labels = {
            "Select": "[单选]",
            "MultiSelect": "[多选]",
            "TextArea": "[填空]"
        }

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
                letters = re.split(r'[、，,\\s]+', user_input)  # 支持多种分隔符
                letters = [letter.strip().upper() for letter in letters if letter.strip()]

                for letter in letters:
                    if len(letter) == 1 and letter.isalpha():
                        idx = ord(letter) - 65  # A->0, B->1
                        if 0 <= idx < len(options):
                            parsed_answer.append(options[idx])
                        else:
                            print(f"⚠️ 选项 {letter} 超出范围（题目：{question['question']}），已忽略。")
                    else:
                        print(f"⚠️ 无效选项格式：{letter}，已忽略。")

                # 单选只取第一个（可选策略）
                if q_type == "Select" and len(parsed_answer) > 1:
                    print(f"⚠️ 注意：'{question['question']}' 是单选题，仅保留第一个选项 '{parsed_answer[0]}'")
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



    def pretty_print_sheet(questions, simulate=True):
        type_labels = {
            "Select": "[单选]",
            "MultiSelect": "[多选]",
            "TextArea": "[填空]"
        }

        print("📝 写作助手答题卡\n")

        answers_parsed = []  # 存储结构化答案

        for idx, q in enumerate(questions, start=1):
            title = q["question"]
            q_type = q["type"]
            options = q["options"]

            # 打印题目编号和标题 + 类型
            print(f"{idx}. {title}？{type_labels.get(q_type, '[未知]')}")

            if q_type in ["Select", "MultiSelect"]:
                # 打印选项 A, B, C...
                for i, option in enumerate(options):
                    letter = chr(65 + i)
                    print(f"   {letter}. {option}")
            elif q_type == "TextArea":
                print("   （请在此处填写内容）")
            
            print()  # 空行分隔

            # 👇 模拟回答
            if simulate:
                if q_type == "Select":
                    choice = random.choice(options)
                    print(f"自动选择：{choice}")
                    answers_parsed.append(choice)

                elif q_type == "MultiSelect":
                    k = random.randint(1, len(options))  # 随机选 1~N 个
                    choices = random.sample(options, k)
                    print(f"自动选择：{'; '.join(choices)}")
                    answers_parsed.append("; ".join(choices))

                elif q_type == "TextArea":
                    fake_text = "自动填充答案_" + "".join(random.choices(string.ascii_letters, k=5))
                    print(f"自动填写：{fake_text}")
                    answers_parsed.append(fake_text)

                else:
                    answers_parsed.append("自动填充")
            else:
                # 如果不模拟，就 fallback 到手动输入逻辑
                answer = input("请输入答案：").strip()
                answers_parsed.append(answer)

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
            question = json.loads(question)
            # if isinstance(question, dict):
            #     question = question.values()
            # elif isinstance(question, list):
            #     question = question
            if isinstance(question, dict):
                question = [question]
            elif isinstance(question, list):
                question = flatten_questions(question)
            else:
                raise ValueError("Unexpected question format")
            print(f"\n\n---\n\n{content}\n\n---\n\n")
            answer_parsed = pretty_print_sheet(question)
            # print(" answer_parsed", answer_parsed)
            # exit()
            feedback = {
                "thread_id": thread_id,
                "content": "[FILLED_QUESTION]" + "\n".join(answer_parsed)
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
                print(response.text)


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
                    print(response.text)

def for_prompt_main():
    # 文件路径
    input_file = "/data1/Yangzb/Wenzhi/CTG/StyleVector/data/zb/insight_empirical_study/speeches_insights_180.json"
    output_file = "/data1/Yangzb/Wenzhi/CTG/StyleVector/data/zb/insight_empirical_study/speeches_outputs_SP.json"

    # 风格设置，可根据需要选择
    style_constraints = {
                "1": {
                    "label": "沉稳致辞式",
                    "bullets": [
                        "正式度：高",
                        "情感：激励与展望",
                        "修辞：分条列举与排比",
                        "结构：回顾成绩→展望未来→号召行动",
                        "标点：冒号与分号频繁使用，感叹号几乎不用",
                        "人称：正式称谓，以第三人称身份发表讲话"
                    ]
                },
                "2": {
                    "label": "庄重倡议式",
                    "bullets": [
                        "正式度：高",
                        "情感：提振士气，倡导合作",
                        "修辞：三段式提议，以详细方案支持",
                        "结构：历史立意→国家使命→合作倡议",
                        "标点：逗号和分号使用频繁，感叹号极少",
                        "人称：正式称谓，多使用第二人称和第一人称复数"
                    ]
                },
                "3": {
                    "label": "隆重致辞式",
                    "bullets": [
                        "正式度：高",
                        "情感：纪念与感恩并重",
                        "修辞：使用历史陈述与展望融合",
                        "结构：历史回顾→国家贡献→未来号召",
                        "标点：冒号与分号频繁，感叹号适中",
                        "人称：正式称谓，常用第一人称表达敬意与祝愿"
                    ]
                },
                "4": {
                    "label": "庄重发布式",
                    "bullets": [
                        "正式度：高",
                        "情感：庄重与坚定",
                        "修辞：使用三段式和重叠句式",
                        "结构：成就回顾→战略规划→具体任务",
                        "标点：冒号与分号使用频繁，少用感叹号",
                        "人称：使用第三人称称呼和正式称谓"
                    ]
                },
                "5": {
                    "label": "严肃部署式",
                    "bullets": [
                        "正式度：高",
                        "情感：庄重与务实",
                        "修辞：条理清晰，多用分条列举",
                        "结构：回顾现状→分析问题→部署工作→落实措施",
                        "标点：分号与冒号频繁使用，感叹号极少",
                        "人称：多用正式称谓，以第三人称身份发表讲话"
                    ]
                }
            }

    # 读取 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        speeches = json.load(f)
    count=0
    for speech in speeches:
        # count=count+1
        # if count==2:
        #     break
        topic = speech["topic"]
        insights = speech["insights"]
        insights_text = "\n".join([f"- {ins}" for ins in insights])
        
        # 可以根据需要替换 rag_text
        rag_text = "这里放置从知识库检索到的补充材料"
        style_key = random.choice(list(style_constraints.keys()))
        style = style_constraints[style_key]
        
        prompt = f"""
        你是一位资深写作者。请根据以下的主题、insight 和补充材料，写一篇完整的长篇文章，至少 1000 字。

        主题：{topic}

        写作要求：
        1. 必须覆盖下列所有 insight（不可遗漏，每个 insight 要单独展开，不要杂糅）。
        2. 每个 insight 必须在正文中完整展开成一个自然段或逻辑部分，但不要直接把 insight 原句当作小标题。
        - 如果需要小标题，必须用简洁概括性的表达，而不是直接复制 insight 原文。
        3. 文章整体要流畅自然，结构清晰，段落充分，前后衔接紧密。
        - 引言：点明主题，提出背景与意义。
        - 正文：逐步展开各个 insight，每部分之间要有自然过渡。
        - 结尾：总结全文，呼应主题，提出展望或号召。
        4. 不要写成逐条罗列式的“清单文章”，要像正式发表的讲话稿或深度评论。
        5. 文章篇幅必须在 1000 字以上。
        6. 使用以下风格约束：{style["label"]}
        - {"; ".join(style["bullets"])}

        需要包含的 insight：
        {insights_text}

        以下是从知识库检索到的补充材料，请结合在文章中：
        {rag_text}

        请写文章：
        """
        # print("prompt",prompt)
        # exit()

        
        
        main(content=prompt)
        
if __name__ == "__main__":
    for_prompt_main()

