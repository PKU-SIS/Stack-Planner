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


def outline_for_parent_node(full_outline: str, parent_node_id: Any) -> str:
    """
    从完整 outline 中只保留 parent_node_id 对应章节及其子节点那一段。
    章节顺序只参照 outline 中的顺序；parent_node_id 对应 outline 中某一级标题，
    返回该标题行及其所有子项（更小缩进直到遇到同级或更上级为止）。
    """
    bullets = parse_outline_bullets(full_outline)
    parent_name = (node_id_to_name(str(parent_node_id or "")) or "").strip()
    if not parent_name:
        return full_outline
    start_idx = None
    for i, (_level, title) in enumerate(bullets):
        if (title or "").strip() == parent_name:
            start_idx = i
            break
    if start_idx is None:
        return full_outline
    start_level = bullets[start_idx][0]
    selected = [bullets[start_idx]]
    for i in range(start_idx + 1, len(bullets)):
        level, title = bullets[i]
        if level <= start_level:
            break
        selected.append((level, title))
    lines = []
    for level, title in selected:
        lines.append("  " * level + "- " + title)
    return "\n".join(lines)


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

def group_by_query_and_outline_and_parent_node_id(
    data: List[Dict[str, Any]]
) -> Dict[Tuple[str, Any], List[Dict[str, Any]]]:
    """
    按 (user_query, outline, parent_node_id) 分组 parsed_dataset。

    注意：同一组内再按 parent_node_id 细分，每个 parent_node_id 视为一章。
    """
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        user_query = item.get("user_query", "")
        outline = item.get("outline")
        parent_node_id = item.get("parent_node_id")
        groups[(user_query, outline, parent_node_id)].append(item)
    return groups


def _sort_items_by_outline_order(
    items: List[Dict[str, Any]], outline: str
) -> List[Dict[str, Any]]:
    """
    将 items 按 node_id 在 outline 中的出现顺序排序。
    每个 item 的 node_id 对应 outline 中的一个章节，step1_output 即该章节的研究结果。
    """
    bullets = parse_outline_bullets(outline)
    node_name_to_pos: Dict[str, int] = {title.strip(): i for i, (_l, title) in enumerate(bullets) if title}
    def key_fn(item: Dict[str, Any]) -> int:
        name = (node_id_to_name(item.get("node_id", "") or "") or "").strip()
        return node_name_to_pos.get(name, 999999)
    return sorted(items, key=key_fn)


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

    # 将大纲处理为md格式    
    outline_nodes = parse_outline_bullets(report_outline)
    outline_md = ''
    for level, title in outline_nodes:
            indent = "#" * (level + 1)
            line = indent + " " + title + "\n "
            outline_md = outline_md + line
            

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

    if style_role == "" :
        messages.append(
        HumanMessage(
            content=(
                f"## 用户需求\n\n{user_query}\n\n"
                f"## 任务描述\n\n使用所有章节的研究结果，生成最终完整报告\n\n"
                ### 🔴 重要：各章节研究数据必须全部使用，不得丢失任何引用或是增加任何无关引用，但要删去明显非报告相关内容如“事实草稿如下”并结合前后文研究内容进行承上启下部分的撰写和章节间流畅度优化\n\n
                f"## 用户补充要求\n\n{user_dst}\n\n"
                f"## 报告大纲\n\n{outline_md}\n\n## 🔴 重要：严格按照报告大纲生成报告，不得删减或者增加章节\n\n"
                f"{reference_hint}"
            )
        )
    )
    else :
        messages.append(
            HumanMessage(
                content=(
                    f"{style_emphasis}"
                    f"## 用户需求\n\n{user_query}\n\n"
                    f"## 任务描述\n\n使用所有章节的研究结果，生成最终完整报告\n\n"
                    ### 🔴 重要：各章节研究数据必须全部使用，不得丢失任何引用或是增加任何无关引用，但要删去明显非报告相关内容如“事实草稿如下”并结合前后文研究内容进行承上启下部分的撰写和章节间流畅度优化\n\n"
                    f"## 用户补充要求\n\n{user_dst}\n\n"
                     f"## 报告大纲\n\n{outline_md}\n\n## 🔴 重要：严格按照报告大纲生成报告，不得删减或者增加章节\n\n"
                    f"{style_change_hint}\n\n{reference_hint}"
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
                content=f"【章节：{node_name}】以下为该章节全部研究数据（必须全部使用、不得遗漏句意、引用须原样保留）: \n\n{obs}",
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
    print("\n" + "=" * 80)
    print("[REPORTER MESSAGES]")
    for i, m in enumerate(messages):
        role = m.get("role", None) if isinstance(m, dict) else getattr(m, "type", None) or getattr(m, "role", None) or "unknown"
        raw = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))
        print(f"--- Message {i + 1} ({role}) ---\n{raw}\n")
    print("=" * 80)
    llm_type = AGENT_LLM_MAP.get("reporter", "basic")
    llm = get_llm_by_type(llm_type)
    response = llm.invoke(messages)
    return response.content


def build_incremental_reporter_messages(
    user_query: str,
    user_dst: str,
    report_outline: str,
    chapter_items: List[Dict[str, Any]],
    previous_report: str,
    chapter_parent_name: str,
) -> List[HumanMessage]:
    """
    构造分段扩展 reporter 的 prompt。
    输入：本段章节的研究数据 + 此前已生成的报告。
    使用 reporter_xxqg_incremental 模板，强调不得丢失任何引用。
    """
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
        "task_description": "分段扩展报告：将本段研究数据整合进已有报告",
    }
    messages = apply_prompt_template(
        "reporter_xxqg_incremental", reporter_input, extra_context=context
    )

    outline_nodes = parse_outline_bullets(report_outline)
    outline_md = ""
    for level, title in outline_nodes:
        indent = "#" * (level + 1)
        outline_md = outline_md + indent + " " + title + "\n"

    chapter_observations: List[str] = []
    for item in chapter_items:
        node_name = node_id_to_name(item.get("node_id", ""))
        step1_output = item.get("step1_output") or ""
        if step1_output.strip():
            chapter_observations.append(
                f"【子节点：{node_name}】\n{step1_output}"
            )

    segment_content = "\n\n---\n\n".join(chapter_observations)

    prev_hint = ""
    if previous_report and previous_report.strip():
        prev_hint = f"""
## 已有报告（须完整保留并在此基础上扩展）

以下为此前已生成的报告内容，你必须**完整保留**，并在其基础上将本段研究数据整合进对应章节位置。不得删减已有内容中的引用。

```
{previous_report}
```

---
"""
    else:
        prev_hint = """
## 已有报告

（当前为空，这是第一段。请根据本段研究数据和大纲生成该章节的初始报告内容。）

---
"""

    content = (
        f"## 用户需求\n\n{user_query}\n\n"
        f"## 任务描述\n\n"
        f"将【本段研究数据】整合进已有报告，输出更新后的**完整报告**。"
        f"请根据全文语境综合处理、润色并添加承上启下连接句，同时务必保留全部事实与引用。\n\n"
        f"## 用户补充要求\n\n{user_dst}\n\n"
        f"## 报告大纲\n\n{outline_md}\n\n"
        f"## 重要：严格按照报告大纲生成报告，不得删减或增加章节\n\n"
        f"{prev_hint}"
        f"## 本段研究数据（须全部使用、不得遗漏任何句意与引用）\n\n"
        f"【章节：{chapter_parent_name}】\n\n{segment_content}"
    )
    messages.append(HumanMessage(content=content, name="search_agent"))
    return messages


def build_fill_reporter_messages(
    user_query: str,
    user_dst: str,
    report_outline: str,
    chapter_items: List[Dict[str, Any]],
    current_report: str,
    chapter_parent_name: str,
    missing_citations: List[str],
) -> List[HumanMessage]:
    """
    填补模式：当前段落 research 中有引用未出现在 report 中，要求 reporter 补充。
    输入：当前完整 report、当前段落 research、缺失的引用列表。
    """
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
        "task_description": "填补模式：将缺失的引用及其对应内容整合进报告",
    }
    messages = apply_prompt_template(
        "reporter_xxqg_incremental", reporter_input, extra_context=context
    )

    outline_nodes = parse_outline_bullets(report_outline)
    outline_md = ""
    for level, title in outline_nodes:
        indent = "#" * (level + 1)
        outline_md = outline_md + indent + " " + title + "\n"

    chapter_observations: List[str] = []
    for item in chapter_items:
        node_name = node_id_to_name(item.get("node_id", ""))
        step1_output = item.get("step1_output") or ""
        if step1_output.strip():
            chapter_observations.append(
                f"【子节点：{node_name}】\n{step1_output}"
            )
    segment_content = "\n\n---\n\n".join(chapter_observations)

    missing_str = "、".join([f"【{c}】" for c in sorted(missing_citations, key=lambda x: int(x))])

    content = (
        f"## 用户需求\n\n{user_query}\n\n"
        f"## 任务描述（填补模式）\n\n"
        f"当前报告在整合本段研究数据时**遗漏了以下引用**：{missing_str}。\n\n"
        f"请在不删减已有报告内容的前提下，将上述缺失引用及其对应的事实/结论从本段研究数据中找出并整合进报告的合适位置。"
        f"必须使用本段研究数据中的全部资料，确保所有引用都出现在输出中。\n\n"
        f"## 用户补充要求\n\n{user_dst}\n\n"
        f"## 报告大纲\n\n{outline_md}\n\n"
        f"## 重要：严格按照报告大纲，不得删减或增加章节\n\n"
        f"## 当前完整报告\n\n```\n{current_report}\n```\n\n---\n\n"
        f"## 本段研究数据（须从中找出缺失引用对应的内容并整合进报告）\n\n"
        f"【章节：{chapter_parent_name}】\n\n{segment_content}"
    )
    messages.append(HumanMessage(content=content, name="search_agent"))
    return messages


def call_fill_reporter(
    user_query: str,
    chapter_items: List[Dict[str, Any]],
    report_outline: str,
    current_report: str,
    chapter_parent_name: str,
    missing_citations: List[str],
    user_dst: str = "",
) -> str:
    """填补模式：调用 reporter 补充缺失的引用。"""
    messages = build_fill_reporter_messages(
        user_query=user_query,
        user_dst=user_dst,
        report_outline=report_outline,
        chapter_items=chapter_items,
        current_report=current_report,
        chapter_parent_name=chapter_parent_name,
        missing_citations=missing_citations,
    )
    print("\n" + "=" * 80)
    print("[FILL REPORTER MESSAGES] 缺失引用:", missing_citations)
    for i, m in enumerate(messages):
        role = m.get("role", None) if isinstance(m, dict) else getattr(m, "type", None) or getattr(m, "role", None) or "unknown"
        raw = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))
        print(f"--- Message {i + 1} ({role}) ---\n{raw}\n")
    print("=" * 80)
    llm_type = AGENT_LLM_MAP.get("reporter", "basic")
    llm = get_llm_by_type(llm_type)
    response = llm.invoke(messages)
    return response.content


def call_incremental_reporter(
    user_query: str,
    chapter_items: List[Dict[str, Any]],
    report_outline: str,
    previous_report: str,
    chapter_parent_name: str,
    user_dst: str = "",
) -> str:
    """调用分段扩展 reporter 生成/更新报告。"""
    messages = build_incremental_reporter_messages(
        user_query=user_query,
        user_dst=user_dst,
        report_outline=report_outline,
        chapter_items=chapter_items,
        previous_report=previous_report,
        chapter_parent_name=chapter_parent_name,
    )
    print("\n" + "=" * 80)
    print("[INCREMENTAL REPORTER MESSAGES]")
    for i, m in enumerate(messages):
        role = m.get("role", None) if isinstance(m, dict) else getattr(m, "type", None) or getattr(m, "role", None) or "unknown"
        raw = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))
        print(f"--- Message {i + 1} ({role}) ---\n{raw}\n")
    print("=" * 80)
    llm_type = AGENT_LLM_MAP.get("reporter", "basic")
    llm = get_llm_by_type(llm_type)
    response = llm.invoke(messages)
    return response.content


def get_ordered_items_by_outline(
    items: List[Dict[str, Any]], outline: str
) -> List[Dict[str, Any]]:
    """
    按 node_id 在 outline 中的顺序返回所有有 step1_output 的 items。
    每个 item 对应一个子章节，其 node_id 即该子章节名。
    """
    valid = [it for it in items if (it.get("step1_output") or "").strip()]
    return _sort_items_by_outline_order(valid, outline)


def _segment_research_citation_set(chapter_items: List[Dict[str, Any]]) -> set:
    """当前段落 research 中的引用集合。"""
    cits: set = set()
    for item in chapter_items:
        out = item.get("step1_output") or ""
        cits.update(extract_citations(out))
    return {int(x) for x in cits}


def extract_citations_from_section(report: str, section_title: str) -> set:
    """
    从报告中找到标题与 section_title 匹配的 section，仅提取该 section body 的 citation set。
    用于检测新生成章节内的引用是否完整，而非全文。
    """
    sections = parse_sections(report)
    section_title_clean = (section_title or "").strip()
    for heading, body in sections:
        heading_clean = re.sub(r"^#+\s*", "", heading).strip()
        if heading_clean == section_title_clean or section_title_clean in heading_clean:
            return set(extract_citation_set(body))
    return set()


def incremental_merge(
    user_query: str,
    items: List[Dict[str, Any]],
    report_outline: str,
    user_dst: str = "",
    max_fill_retries: int = 3,
) -> str:
    """
    分段式 merge：按 outline 顺序，逐子章节调用 reporter。
    每段生成后检测 citation_set，若有缺失则进入填补模式，直至全部引用就位才进入下一段。
    """
    ordered_items = get_ordered_items_by_outline(items, report_outline)
    accumulated_report = ""
    for item in ordered_items:
        chapter_name = node_id_to_name(item.get("node_id", "") or "")
        chapter_items = [item]
        research_cits = _segment_research_citation_set(chapter_items)
        if not research_cits:
            continue
        accumulated_report = call_incremental_reporter(
            user_query=user_query,
            chapter_items=chapter_items,
            report_outline=report_outline,
            previous_report=accumulated_report,
            chapter_parent_name=chapter_name,
            user_dst=user_dst,
        )
        fill_retries = 0
        while fill_retries < max_fill_retries:
            report_cits = extract_citations_from_section(accumulated_report, chapter_name)
            missing = research_cits - report_cits
            if not missing:
                break
            missing_strs = [str(c) for c in sorted(missing)]
            print(f"\n[引用缺失] 段落「{chapter_name}」缺失引用: {missing_strs}，进入填补模式 (retry {fill_retries + 1}/{max_fill_retries})")
            accumulated_report = call_fill_reporter(
                user_query=user_query,
                chapter_items=chapter_items,
                report_outline=report_outline,
                current_report=accumulated_report,
                chapter_parent_name=chapter_name,
                missing_citations=missing_strs,
                user_dst=user_dst,
            )
            fill_retries += 1
        if fill_retries >= max_fill_retries and research_cits - extract_citations_from_section(accumulated_report, chapter_name):
            print(f"\n[警告] 段落「{chapter_name}」经 {max_fill_retries} 次填补后仍有引用缺失，继续下一段")
    return accumulated_report


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


