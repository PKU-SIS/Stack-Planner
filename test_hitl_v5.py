import httpx
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


# 暂时禁用代理设置以解决可能的网络代理导致的 502 错误（如需请自行开启）
# os.environ["http_proxy"] = "http://localhost:8888"
# os.environ["https_proxy"] = "http://localhost:8888"
# os.environ["HTTP_PROXY"] = "http://localhost:8888"
# os.environ["HTTPS_PROXY"] = "http://localhost:8888"


url = "http://localhost:8505/api/chat/sp_stream"
base_url = "http://localhost:8505"  # 基础 URL，用于其他 API 调用

# 初始请求内容，带有 [STYLE_ROLE] 标记指定初始风格
content = """你是一位资深政策讲话撰稿专家。请根据以下要求撰写一篇领导干部发言稿：

【主题】
以文化建设"八项工程"为统领，打造新时代高水平文化强省，争当学习践行习近平文化思想排头兵

【核心见解】
- 文化是推进中国式现代化的精神引擎和战略支撑，必须以文化自信引领文化自强，在"八项工程"系统化推进中实现文化赋能经济社会发展的全局性价值。
- "八项工程"既是习近平文化思想的重要实践源头，也是"八八战略"思想体系的文化篇，体现了文化建设的系统性、工程化和规律化推进逻辑。
- 建设文化强省要在传承中创新、在守正中发展，通过"文化+科技""文化+旅游""文化+民生"等路径推动文化高质量发展与人的全面发展相统一。

【风格要求】
- 政治庄重与思想深邃并重，贯穿坚定的政治立场与理论自觉。
- 条理清晰、逻辑递进，常以"三个必须""三个方面"等结构展开论述。
- 语言具有政策化修辞和战略规划色彩，强调方向、路径与行动并举。
- 情感基调稳健昂扬，兼具历史纵深感与实践感召力。
- 论述体现"系统思维—工程化推进—实践成效"的层层递进式表达。[STYLE_ROLE]""".strip() #鲁迅

# 可选的风格列表
AVAILABLE_STYLES = ["鲁迅", "赵树理", "侠客岛"]

# 测试模式: "style_switch" | "content_modify" | "interactive"
# - style_switch: 测试风格切换（直接返回 reporter）
# - content_modify: 测试内容修改（走 central_agent 决策）
# - interactive: 交互式测试
TEST_MODE = "content_modify"


def parse_json_maybe(value: Union[str, dict, list]) -> Union[dict, list, str]:
    """
    尝试将字符串解析为 JSON；若失败或输入非字符串，则原样返回。
    """
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def pretty_print_sheet(questions: List[dict]) -> List[str]:
    """
    以更友好的方式展示问卷，并收集结构化答案（非交互环境自动使用默认答案）。

    返回与题目一一对应的答案列表（字符串）。
    """
    type_labels = {"Select": "[单选]", "MultiSelect": "[多选]", "TextArea": "[填空]"}

    print("📝 写作助手答题卡\n")
    for idx, q in enumerate(questions, start=1):
        title = q.get("question", "")
        q_type = q.get("type", "")
        options = q.get("options", []) or []
        print(f"{idx}. {title}？{type_labels.get(q_type, '[未知]')}")
        if q_type in ["Select", "MultiSelect"]:
            for i, option in enumerate(options):
                letter = chr(65 + i)  # 65 = 'A'
                print(f"   {letter}. {option}")
        elif q_type == "TextArea":
            print("   （请在此处填写内容）")
        print()

    print("请逐条回答问题：")
    print("👉 选择题请输入选项字母（如 A 或 A、B），填空题直接写内容。")
    print("🔚 每行一个答案，输入完后输入 END 结束：\n")

    answers_parsed: List[str] = []
    answer_lines: List[str] = []

    if sys.stdin.isatty():
        answer = input().strip()
        while answer.upper() != "END":
            answer_lines.append(answer)
            answer = input().strip()
    else:
        # 非交互式环境，使用默认答案
        print("非交互式环境，问卷使用默认答案...")
        for question in questions:
            if question.get("type") == "Select":
                answer_lines.append("A")  # 默认选择第一个选项
            else:
                answer_lines.append("默认答案")

    import re

    for line, question in zip(answer_lines, questions):
        user_input = (line or "").strip()
        q_type = question.get("type", "")
        options = question.get("options", []) or []

        if q_type in ["Select", "MultiSelect"]:
            letters = re.split(r"[、，,\s]+", user_input)
            letters = [letter.strip().upper() for letter in letters if letter.strip()]
            parsed: List[str] = []
            for letter in letters:
                if len(letter) == 1 and letter.isalpha():
                    idx = ord(letter) - 65  # A->0, B->1
                    if 0 <= idx < len(options):
                        parsed.append(options[idx])
                    else:
                        print(
                            f"⚠️ 选项 {letter} 超出范围（题目：{question.get('question','')}），已忽略。"
                        )
                else:
                    print(f"⚠️ 无效选项格式：{letter}，已忽略。")
            if q_type == "Select" and len(parsed) > 1:
                print(
                    f"⚠️ 注意：'{question.get('question','')}' 是单选题，仅保留第一个选项 '{parsed[0]}'"
                )
                parsed = [parsed[0]]
            answers_parsed.append("; ".join(parsed))
        elif q_type == "TextArea":
            answers_parsed.append(user_input)
        else:
            answers_parsed.append(user_input)

    return answers_parsed


def present_outline_and_get_feedback(outline_value: Union[str, dict, list]) -> str:
    """
    展示并获取用户对大纲的确认/修改反馈。

    返回反馈字符串，形如："[CONFIRMED_OUTLINE]..."。
    非交互环境下，默认直接确认原始大纲。
    """
    outline = parse_json_maybe(outline_value)

    print("\n\n🧩 大纲预览\n")
    if isinstance(outline, (dict, list)):
        print(json.dumps(outline, ensure_ascii=False, indent=2))
        outline_str = json.dumps(outline, ensure_ascii=False)
    else:
        print(str(outline))
        outline_str = str(outline)

    if not sys.stdin.isatty():
        print("\n非交互式环境，自动确认现有大纲。\n")
        return "[CONFIRMED_OUTLINE]" + outline_str

    print(
        "\n请确认或编辑大纲：输入 'CONFIRM' 确认；输入 'EDIT' 后粘贴新大纲，最后输入 'END' 提交。\n"
    )
    choice = input("输入指令：").strip().upper()
    if choice == "EDIT":
        print("请粘贴新的大纲内容（多行），结束后输入 'END'：")
        new_lines: List[str] = []
        line = input()
        while line.strip().upper() != "END":
            new_lines.append(line)
            line = input()
        edited_outline = "\n".join(new_lines).strip()
        return "[CONFIRMED_OUTLINE]" + edited_outline
    else:
        return "[CONFIRMED_OUTLINE]" + outline_str


def present_report_and_get_feedback(report_content: str) -> str:
    """
    展示生成的报告，并询问用户操作。

    返回反馈字符串：
    - "[CHANGED_STYLE]xxx" 表示切换到新风格（简单修改，直接返回 reporter）
    - "[CONTENT_MODIFY]xxx" 表示内容修改（复杂修改，走 central_agent 决策）
    - "[SKIP]" 表示结束
    """
    print("\n\n" + "=" * 60)
    print("📄 报告已生成")
    print("=" * 60)
    print(report_content)
    print("=" * 60 + "\n")

    print("🎨 可选风格：")
    for i, style in enumerate(AVAILABLE_STYLES, start=1):
        print(f"   {i}. {style}")
    print()

    if not sys.stdin.isatty():
        # 非交互式环境：根据 TEST_MODE 决定测试场景
        if TEST_MODE == "style_switch":
            print("非交互式环境，测试风格切换：切换到 '赵树理' 风格...")
            return "[CHANGED_STYLE]赵树理"
        elif TEST_MODE == "content_modify":
            print("非交互式环境，测试内容修改：请求增加数据支撑...")
            return "[CONTENT_MODIFY]请在第二段增加更多具体数据和案例支撑"
        else:
            return "[SKIP]"

    print("请选择操作：")
    print("  - 输入数字 (1/2/3) 切换到对应风格")
    print("  - 输入风格名称 (如 '鲁迅') 切换风格")
    print("  - 输入 'MODIFY' 进行内容修改（走 central_agent 决策）")
    print("  - 输入 'SKIP' 或 'END' 结束")
    print()

    choice = input("输入选择：").strip()

    if choice.upper() in ["SKIP", "END", ""]:
        return "[SKIP]"

    if choice.upper() == "MODIFY":
        print("请输入修改意见：")
        modify_request = input().strip()
        return f"[CONTENT_MODIFY]{modify_request}"

    # 尝试解析数字
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(AVAILABLE_STYLES):
            return f"[CHANGED_STYLE]{AVAILABLE_STYLES[idx]}"
        else:
            print(f"⚠️ 无效选项 {choice}，默认结束")
            return "[SKIP]"

    # 尝试匹配风格名称
    for style in AVAILABLE_STYLES:
        if style in choice:
            return f"[CHANGED_STYLE]{style}"

    print(f"⚠️ 未识别的输入 '{choice}'，默认结束")
    return "[SKIP]"


_perception_node_count = 0
_suppress_after_second_perception = False
_style_switch_count = 0  # 记录风格切换次数
_content_modify_count = 0  # 记录内容修改次数


def _is_perception_node(current_node: Any) -> bool:
    if (
        isinstance(current_node, list)
        and current_node
        and isinstance(current_node[0], str)
    ):
        return current_node[0].startswith("perception:")
    if isinstance(current_node, str):
        return current_node.startswith("perception:")
    return False


def process_event(
    event_type: str, event_data: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    处理一个完整的 SSE 事件。

    若为中断事件，返回 `{thread_id, content}` 作为下一次请求的 interrupt_feedback；否则返回 None。
    """
    global _perception_node_count, _suppress_after_second_perception
    global _style_switch_count, _content_modify_count

    # 当第二次进入 perception 节点后，直到下一次 interrupt 之前，抑制输出
    if _suppress_after_second_perception and event_type != "interrupt":
        return None

    if event_type in ["message_chunk", "tool_calls", "tool_call_result"]:
        content = event_data.get("content", "")
        if content:
            print(content, end="", flush=True)
        return None
    elif event_type == "node_status":
        # 可选：输出节点状态
        current_node = event_data.get("current_node", "")
        status = event_data.get("status", "")
        thread_id = event_data.get("thread_id", "")
        if _is_perception_node(current_node):
            _perception_node_count += 1
            if _perception_node_count == 2:
                _suppress_after_second_perception = True
        print(f"\n[节点状态] {current_node} - {status} (thread_id={thread_id})\n")
        return None
    elif event_type == "interrupt":
        # 收到 interrupt 后，恢复输出
        if _suppress_after_second_perception:
            _suppress_after_second_perception = False
        thread_id = event_data.get("thread_id", "")
        content = event_data.get("content", "")
        question_raw = event_data.get("question", None)
        outline_raw = event_data.get("outline", None)

        print(f"\n\n--- 中断 ---\n{content}\n---\n\n")

        # 第一阶段：需要用户填写问卷
        if question_raw is not None:
            try:
                question = parse_json_maybe(question_raw)
                if isinstance(question, dict):
                    questions = list(question.values())
                elif isinstance(question, list):
                    questions = question
                else:
                    raise ValueError("Unexpected question format")
            except Exception as e:
                raise ValueError(f"Failed to parse 'question': {e}")

            answers_parsed = pretty_print_sheet(questions)
            feedback_content = "[FILLED_QUESTION]" + "\n".join(answers_parsed)
            feedback_content = "[SKIP]"
            return {"thread_id": thread_id, "content": feedback_content}

        # 第二阶段：确认或编辑大纲
        if outline_raw is not None:
            feedback_content = present_outline_and_get_feedback(outline_raw)
            feedback_content = "[SKIP]"
            return {"thread_id": thread_id, "content": feedback_content}

        # 第三阶段：报告生成完成，可以切换风格或修改内容
        # 检查 content 中是否包含 [REPORT]...[/REPORT] 标记
        if "[REPORT]" in content and "[/REPORT]" in content:
            # 提取报告内容
            start_idx = content.find("[REPORT]") + len("[REPORT]")
            end_idx = content.find("[/REPORT]")
            report_content = content[start_idx:end_idx].strip()

            # 非交互式环境下，限制操作次数以避免无限循环
            if not sys.stdin.isatty():
                total_count = _style_switch_count + _content_modify_count
                if total_count > 0:
                    print("\n非交互式环境，已操作过一次，自动结束。\n")
                    return {"thread_id": thread_id, "content": "[SKIP]"}
                if TEST_MODE == "style_switch":
                    _style_switch_count += 1
                elif TEST_MODE == "content_modify":
                    _content_modify_count += 1

            feedback_content = present_report_and_get_feedback(report_content)
            print("feedback_content: ", feedback_content)
            return {"thread_id": thread_id, "content": feedback_content}

        # 未知中断类型
        print(f"⚠️ 未知中断类型，content: {content[:200]}...")
        return {"thread_id": thread_id, "content": "[SKIP]"}

    return None


def run_once(request_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    执行一次流式请求，返回 (下一次请求数据或 None, HTTP 状态码)。
    如果期间收到中断并生成了反馈，则返回更新后的下一次请求数据；否则返回 None。
    """
    buffer = ""
    next_request: Optional[Dict[str, Any]] = None
    status_code: int = 0

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json=request_data) as response:
                status_code = response.status_code
                if response.status_code == 200:
                    for chunk in response.iter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line.startswith("event:"):
                                event_type = line.split(":", 1)[1].strip()
                            elif line.startswith("data:"):
                                data_str = line.split(":", 1)[1].strip()
                                try:
                                    event_data = json.loads(data_str)
                                    res = process_event(event_type, event_data)
                                    if res is not None:
                                        # 为下一次请求准备反馈
                                        new_payload = dict(request_data)
                                        new_payload["interrupt_feedback"] = res["content"]
                                        new_payload["thread_id"] = res["thread_id"]
                                        new_payload["auto_accepted_plan"] = False
                                        next_request = new_payload
                                except json.JSONDecodeError:
                                    # 忽略无效 JSON 行
                                    pass
                else:
                    print(f"Error: {response.status_code}")
                    try:
                        response.read()
                        print(response.text)
                    except Exception as e:
                        print(f"Error reading response: {e}")
    except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
        print(f"\n⚠️ 连接错误: {e}")
        print("服务器可能已断开连接或崩溃，请检查后端服务状态。")
        return None, 500

    return next_request, status_code


def main() -> None:
    """
    测试脚本 v5：支持风格切换和内容修改两种模式

    流程：
    1) 首次启动，后端返回问卷中断 -> 发送 [FILLED_QUESTION]... 续传
    2) 生成大纲并返回中断 -> 发送 [CONFIRMED_OUTLINE]... 续传
    3) 报告生成完成，返回中断 -> 可选择：
       - [CHANGED_STYLE]xxx: 切换风格，直接返回 reporter 重新生成
       - [CONTENT_MODIFY]xxx: 内容修改，走 central_agent 决策后委派相应 agent
       - [SKIP]: 结束流程

    通过修改 TEST_MODE 变量切换测试场景：
    - "style_switch": 测试风格切换（直接返回 reporter）
    - "content_modify": 测试内容修改（走 central_agent 决策）
    - "interactive": 交互式测试
    """
    print(f"\n🧪 测试模式: {TEST_MODE}\n")
    data: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "resources": [],
        "thread_id": "__default__",
        "max_plan_iterations": 1,
        "max_step_num": 30,
        "max_search_results": 3,
        "auto_accepted_plan": True,
        # 任意非空占位，后端会用 auto_accepted_plan 控制中断逻辑
        "interrupt_feedback": "string",
        "mcp_settings": {},
        "enable_background_investigation": True,
        "graph_format": "sp_xxqg",
        "knowledge_base_name": "学习强国",
    }

    # 允许多次中断续传：问卷(1) + 大纲(1) + 风格切换(N)
    # 设置较大的 max_retries 以支持多次风格切换
    max_retries = 10
    attempt = 0

    while attempt <= max_retries:
        next_data, status = run_once(data)

  
        if status != 200:
            # 请求失败，直接退出
            print(f"\n❌ 请求失败，状态码: {status}")
            break
        if next_data is None:
            # 没有新的中断，表示已完成
            print("\n✅ 流程完成！")
            break

        # 检查是否是 [SKIP] 反馈，如果是则结束
        if next_data.get("interrupt_feedback", "").upper().startswith("[END]"):
            print("\n✅ 用户选择结束，流程完成！")
            break

        attempt += 1
        data = next_data
        print("\n\n---\n\n正在根据你的反馈继续生成内容...\n\n---\n\n")
        print(f"id: {data['thread_id']}")
        response = httpx.get(f"{base_url}/api/references/{data['thread_id']}")
        if response.status_code == 200:
            ref_data = response.json()
            references = ref_data.get("references", [])
            if references:
                print("\n\n---\n\n参考资料：\n")
                for ref in references:
                    print(f"- {ref}")
                print("\n\n---\n\n")  

    if attempt > max_retries:
        print(f"\n⚠️ 达到最大重试次数 ({max_retries})，流程结束。")

    print(f"id: {data['thread_id']}")
    response = httpx.get(f"{base_url}/api/references/{data['thread_id']}")
    if response.status_code == 200:
        ref_data = response.json()
        references = ref_data.get("references", [])
        if references:
            print("\n\n---\n\n参考资料：\n")
            for ref in references:
                print(f"- {ref}")
            print("\n\n---\n\n")  


if __name__ == "__main__":
    main()
