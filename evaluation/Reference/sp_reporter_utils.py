import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from src.prompts.template import apply_prompt_template
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.agents.SubAgentManager import SubAgentManager

# citation utils
def extract_citations(text: str) -> List[str]:
    """按顺序提取正文中的所有【数字】引用（不去重）。"""
    return re.findall(r"【(\d+)】", text or "")

def extract_citation_set(text: str) -> List[int]:
    """提取正文中的引用编号集合（去重后按数字排序）。"""
    nums = re.findall(r"【(\d+)】", text or "")
    return sorted({int(n) for n in nums})

def citations_for_node_name(node_name: str, outputs: List[str]) -> List[str]:
    """
    outputs 是同一 node_name 下可能多段 step1_output
    这里按原始顺序合并 citations（不去重），你也可以改成 set 去重。
    """
    cits: List[str] = []
    for t in outputs:
        cits.extend(extract_citations(t or ""))
    return cits

def is_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s+\S", line))
    
def parse_sections(md: str) -> List[Tuple[str, str]]:
    """
    把 markdown 按标题切成多个 section:
    返回 [(heading_line, body_text), ...]
    - heading_line: '# xxx' / '## xxx' ...
    - body_text: 该标题到下一个标题前的正文（不含标题行）
    """
    lines = (md or "").splitlines()
    sections: List[Tuple[str, List[str]]] = []
    cur_heading = "(NO HEADING)"
    cur_body: List[str] = []

    for line in lines:
        if is_heading(line):
            # flush previous
            sections.append((cur_heading, cur_body))
            cur_heading = line.strip()
            cur_body = []
        else:
            cur_body.append(line)
    sections.append((cur_heading, cur_body))

    # 转成字符串
    return [(h, "\n".join(b).strip()) for h, b in sections if h or b]

def parse_outline_bullets(outline: str) -> List[Tuple[int, str]]:
    """
    解析类似你给的 outline：
    - 一级
      - 二级
        - 三级
    返回 [(level, title), ...]
    level 从 0 开始。
    """
    res: List[Tuple[int, str]] = []
    for raw in (outline or "").splitlines():
        if not raw.strip():
            continue
        # 只认以 "-" 开头的条目（你的 outline 是这种）
        m = re.match(r"^(\s*)-\s+(.*\S)\s*$", raw)
        if not m:
            continue
        indent = len(m.group(1))
        title = m.group(2).strip()
        # 你的 outline 看起来每层缩进 2 个空格；这里用 //2 转成层级
        level = indent // 2
        res.append((level, title))
    return res

def node_id_to_name(node_id: str) -> str:
    # 兼容你现在 node_id 的格式： "节点:  xxx (ID: node_32)"
    try:
        return node_id.split("节点:", 1)[1].split("(ID:", 1)[0].strip()
    except Exception:
        return node_id.strip()
def group_by_query_and_outline(
    data: List[Dict[str, Any]]
) -> Dict[Tuple[str, Any], List[Dict[str, Any]]]:
    """
    按 (user_query, outline) 分组 parsed_dataset。

    注意：同一组内再按 parent_node_id 细分，每个 parent_node_id 视为一章。
    """
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        user_query = item.get("user_query", "")
        outline = item.get("outline")
        groups[(user_query, outline)].append(item)
    return groups


def build_reporter_messages(
    user_query: str,
    user_dst: str,
    report_outline: str,
    style_role: str,
    items: List[Dict[str, Any]],
    original_report: str | None = None,
) -> List[HumanMessage]:
    """
    按照 SubAgentManager._generate_report_with_style 的逻辑，构造 reporter 的 prompt。

    - 使用相同的 reporter_xxqg 模板（apply_prompt_template）
    - 使用相同的 LLM 获取方式（get_llm_by_type + AGENT_LLM_MAP["reporter"]）
    - 同组内 parent_node_id 相同的 step1_output 视为一章的 observations
    - chapter_results: 章节号 -> 该章 observations 列表
    - original_report 存在时，附加引用保持要求和风格切换提示
    """
    # 1. 风格约束
    constraint = SubAgentManager.ROLE_CONSTRAINTS.get(style_role, "")
    if not constraint:
        constraint = style_role

    # 2. reporter 输入（与 SubAgentManager 中保持一致）
    reporter_input = {
        "messages": [
            HumanMessage(
                content=f"# Research Requirements\n\n## User Query\n\n{user_query}"
            )
        ],
        "locale": "zh-CN",
    }


    context = {
        "user_query": user_query,
        "task_description": "使用所有章节的研究结果，生成最终完整报告",
    }

    messages = apply_prompt_template(
        "reporter_xxqg", reporter_input, extra_context=context
    )

    reference_hint = ""
    style_change_hint = ""

    if original_report:
        citations = re.findall(r"【(\d+)】", original_report)
        if citations:
            unique_citations = sorted(set(citations), key=lambda x: int(x))
            reference_hint = (
                "\n\n## 引用保持要求\n\n原始报告使用了以下引用编号："
                + "、".join([f"【{c}】" for c in unique_citations])
                + "。请在新风格的报告中尽量保持使用相同的引用来源。"
            )

        style_change_hint = f"""

## 🔴 重要：风格切换要求

用户明确要求将报告风格切换为：**{style_role}**

原始报告已经生成，现在需要你**完全重写报告**，使用新的风格。

### 原始报告（仅供参考结构和引用）：
{original_report[:2000]}...

---

请基于原始报告的结构和引用信息，使用 **{style_role}** 风格完全重写报告。
"""

    style_emphasis = f"""
# 🎯 写作风格要求（最高优先级）

你正在撰写一篇 **{style_role}** 风格的报告。请严格遵守以下风格约束：

{constraint}

---
"""

    messages.append(
        HumanMessage(
            content=(
                f"{style_emphasis}"
                f"## 用户需求\n\n{user_query}\n\n"
                f"## 任务描述\n\n使用所有章节的研究结果，生成最终完整报告\n\n"
                f"## 用户补充要求\n\n{user_dst}\n\n"
                f"## 报告大纲\n\n{report_outline}"
                f"{style_change_hint}{reference_hint}"
            )
        )
    )

    # 将每一节点的 step1_output 映射为 从章节名到observations的map
    chapter_results: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        node_id = item.get("node_id")
        node_name = node_id_to_name(node_id)
        step1_output = item.get("step1_output") or ""
        if not step1_output:
            continue
        chapter_results[node_name].append(step1_output)

    # observations 追加 
    for node_name, obs in chapter_results.items():
        messages.append(
            HumanMessage(
                content=f"以下是检索智能体收集到的大纲中章节{node_name}的高质量信息: \n\n{obs}",
                name="search_agent",
            )
        )

    print(f"messages: {messages}")
    return messages


def call_reporter(
    user_query: str,
    items: List[Dict[str, Any]],
    style_role: str,
    original_report: str | None = None,
    user_dst: str = "",
    report_outline: str = "用户未提供大纲",
) -> str:
    """调用与 SubAgentManager 中 reporter 相同的 LLM 接口生成报告。"""
    messages = build_reporter_messages(
        user_query=user_query,
        user_dst=user_dst,
        report_outline=report_outline,
        style_role=style_role,
        items=items,
        original_report=original_report,
    )

    llm_type = AGENT_LLM_MAP.get("reporter", "basic")
    llm = get_llm_by_type(llm_type)
    response = llm.invoke(messages)
    return response.content


def naive_merge(
    user_query: str,
    items: List[Dict[str, Any]],
    report_outline: str,
) -> str:
    """
    naive_merge: 直接把同一 (user_query, parent_node_id) 下的所有 step1_output
    作为 reporter 的“章节研究发现”和 observations，生成一篇合并后的报告。
    """
    return call_reporter(
        user_query=user_query,
        items=items,
        style_role="",
        original_report=None,
        report_outline=report_outline,
    )


def naive_style_change(
    user_query: str,
    items: List[Dict[str, Any]],
    merged_report: str,
    target_style: str,
    report_outline: str,
) -> str:
    """
    naive_style_change: 在 naive_merge 生成的报告基础上进行风格切换。

    - original_report = merged_report
    - 仍然使用同一批 step1_outputs 作为研究发现 / observations
    - style_role 切换为 target_style
    """
    return call_reporter(
        user_query=user_query,
        items=items,
        style_role=target_style,
        original_report=merged_report,
        report_outline=report_outline,
    )


