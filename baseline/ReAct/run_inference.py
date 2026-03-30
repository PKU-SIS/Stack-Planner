import sys
import os
import json
import re
import argparse
import copy
import yaml
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= 动态路径注入 Hack =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
eval_utils_path = os.path.join(project_root, "evaluation", "deep_research_bench")

if eval_utils_path not in sys.path:
    sys.path.append(eval_utils_path)
if project_root not in sys.path:
    sys.path.append(project_root)
# =====================================================

from src.tools.bocha_search.web_search_en import web_search


def load_api_config(conf_path: str | None = None) -> dict:
    """与 baseline/RAG/run_inference.py 一致：读取 conf.yaml 中的 BASIC_MODEL。"""
    path = conf_path or os.path.join(project_root, "conf.yaml")
    with open(path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    basic = conf["BASIC_MODEL"]
    cfg = {
        "base_url": basic["base_url"],
        "api_key": basic["api_key"],
        "model": basic["model"],
    }
    if basic.get("extra_body"):
        cfg["extra_body"] = copy.deepcopy(basic["extra_body"])
    else:
        cfg["extra_body"] = {}
    return cfg


def chat_generate(client: OpenAI, api_config: dict, user_prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """OpenAI 兼容接口单次非流式调用（ReAct 每步需要完整 Thought/Action）。"""
    extra = copy.deepcopy(api_config.get("extra_body") or {})
    extra["stream"] = False
    resp = client.chat.completions.create(
        model=api_config["model"],
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra,
    )
    msg = resp.choices[0].message
    return (msg.content or "").strip()


def load_prompt(prompt_path="baseline/ReAct/react_prompt.md"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, "react_prompt.md")
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def _guess_title_from_url(url: str) -> str:
    """从 URL 路径取一段可读片段作标题占位，便于与 RAG 格式对齐。"""
    try:
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = path.split("?")[0].strip("/") or url
        return path[:120] if len(path) > 120 else path
    except Exception:
        return "Web Search Result"


def _canonical_url(url: str) -> str:
    """用于去重：忽略 fragment、统一小写 host、去掉末尾 /。"""
    try:
        p = urlparse(url.strip())
        path = (p.path or "").rstrip("/") or "/"
        netloc = (p.netloc or "").lower()
        return urlunparse((p.scheme.lower(), netloc, path, "", p.query, ""))
    except Exception:
        return url.strip()


# 与 RAG 一致：每条检索结果一条 research，编号全局递增 1…N（多次 Search × 多条结果即 1…40）
SEARCH_TOP_K = 5          # 每轮最多 5 条，减少 history 膨胀
MAX_HISTORY_CHARS = 60_000  # history 超此长度时自动截断中间旧 Observation


def clean_article(text: str) -> str:
    """清洗 LLM 输出中的常见噪声：
    - 部分中文模型将 *italic* 解码为 林text林，还原为 **text**
    - 去掉残留孤立星号
    - 折叠连续空行
    """
    if not text:
        return text
    # 1. 林text林 → **text**（中文模型单星号乱码）
    text = re.sub(r"林([^林\n]{1,80})林", r"**\1**", text)
    # 2. 残留孤立 *（非 ** 的单星号）→ 删除，避免渲染乱码
    text = re.sub(r"(?<!\*)\*(?!\*)", "", text)
    # 3. 折叠超过 2 个的连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 4. 去掉行首尾多余空格
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def execute_search(query, collected_research):
    """
    使用 web_search（返回 list[dict]，与 RAG 相同）；每条结果单独占 research 一条、单独编号。
    """
    print(f"  [Debug] 正在调用 web_search, 关键词: {query}")
    try:
        raw = web_search(query, top_k=SEARCH_TOP_K)
        print("search raw",raw)
        if not raw:
            return "未检索到相关内容。"

        seen = {_canonical_url(r.get("url") or "") for r in collected_research if r.get("url")}
        blocks = []

        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                url = (item.get("url") or item.get("link") or "").strip()
                if not url:
                    continue
                c = _canonical_url(url)
                if c in seen:
                    continue
                seen.add(c)
                idx = len(collected_research) + 1
                title = (item.get("title") or "").strip() or _guess_title_from_url(url)
                content = (item.get("content") or item.get("snippet") or "")[:1500]
                collected_research.append({"url": url, "title": title[:500], "content": content})
                blocks.append(
                    f"【{idx}】\n标题: {title}\n链接: {url}\n内容摘录:\n{content[:800]}"
                )
        else:
            # 兼容旧版返回纯文本时从字符串里抽 URL（每 URL 仍单独一条，内容用片段）
            obs_text = str(raw)
            urls = re.findall(r"https?://[^\s<>\"']+|(?:www\.)[^\s<>\"']+", obs_text)
            unique_urls = []
            for u in urls:
                if u not in unique_urls:
                    unique_urls.append(u)
            for url in unique_urls:
                c = _canonical_url(url)
                if c in seen:
                    continue
                seen.add(c)
                idx = len(collected_research) + 1
                title = _guess_title_from_url(url)
                collected_research.append({
                    "url": url,
                    "title": title,
                    "content": obs_text[:4000],
                })
                blocks.append(
                    f"【{idx}】\n标题: {title}\n链接: {url}\n内容摘录:\n{obs_text[:2000]}"
                )

        if not blocks:
            return "本轮未新增独立链接（可能与已有来源重复）。请尝试换用其他搜索词。"

        print(f"  [Debug] ✅ 搜索成功！本轮新增 {len(blocks)} 条来源（当前共 {len(collected_research)} 条）。")
        header = (
            "以下为检索结果。每条【序号】对应最终 JSON 中 research 的一条（全局顺序编号）。"
            "终稿引用【1】【2】…须与此序号一致。\n\n"
        )
        return header + "\n\n---\n\n".join(blocks)

    except Exception as e:
        print(f"  [Debug] ❌ web_search 调用失败: {e}")
        return "搜索失败，请尝试其他关键词。"


MIN_FINISH_ARTICLE_CHARS = 400

_FINISH_GARBAGE = frozenset(
    {"完成", "done", "ok", "好的", "如下", "见上", "同上", "已写完", "完毕", "finished"}
)


def parse_action_line(response_text: str) -> str | None:
    """
    解析本轮意图，以**第一个** Action 为准。
    模型有时把多轮 Thought/Action/Observation 全写在一条回复里，
    取第一个 Action 可强制只处理最早的 Search，截断假 Observation，
    让系统执行真实检索后再继续。
    """
    for line in response_text.splitlines():
        lm = re.match(
            r"^\s*\*{0,2}Action\*{0,2}\s*[:：]\s*\*{0,2}\s*\[?\s*(Search|Finish)\s*\]?\s*$",
            line,
            re.I,
        )
        if lm:
            return lm.group(1).capitalize()
        lm2 = re.match(r"^\s*动作\s*[:：]\s*(搜索|检索|查找|完成|结束)\s*$", line)
        if lm2:
            w = lm2.group(1)
            return "Search" if w in ("搜索", "检索", "查找") else "Finish"
    # 兜底：正文中找第一个 Action 关键词
    m = re.search(r"Action\s*[:：]\s*(Search|Finish)", response_text, re.I)
    if m:
        return m.group(1).capitalize()
    return None


def _first_block_line_index(lines: list[str], action: str) -> int:
    """返回第一处对应 Action 行的下标（Search 或 Finish）。"""
    pat = (
        r"^\s*\*{0,2}Action\*{0,2}\s*[:：]\s*\*{0,2}\s*\[?\s*"
        + action
        + r"\s*\]?\s*$"
    )
    for i, line in enumerate(lines):
        if re.match(pat, line, re.I):
            return i
        if action == "Search" and re.match(
            r"^\s*动作\s*[:：]\s*(搜索|检索|查找)\s*$", line
        ):
            return i
        if action == "Finish" and re.match(r"^\s*动作\s*[:：]\s*(完成|结束)\s*$", line):
            return i
    return -1


# 保留旧名称作别名，避免遗漏调用处
_last_block_line_index = _first_block_line_index


def parse_search_query(response_text: str) -> str | None:
    """只取「最后一个 Action: Search」之后的第一个 Action Input 单行。"""
    lines = response_text.splitlines()
    i0 = _last_block_line_index(lines, "Search")
    if i0 < 0:
        return None
    tail = "\n".join(lines[i0 + 1 :])
    m = re.search(r"(?im)^\s*\*?\*?Action\s*Input\*?\*?\s*[:：]\s*([^\n]+)", tail)
    if m:
        return m.group(1).strip()
    return None


def parse_finish_body(response_text: str) -> str | None:
    """只取「最后一个 Action: Finish」之后、Action Input: 起的整段终稿（多行 Markdown）。"""
    lines = response_text.splitlines()
    i0 = _last_block_line_index(lines, "Finish")
    if i0 < 0:
        return None
    tail = "\n".join(lines[i0 + 1 :])
    m = re.search(r"(?is)^\s*\*?\*?Action\s*Input\*?\*?\s*[:：]\s*(.*)$", tail)
    if not m:
        return None
    body = m.group(1).strip()
    body = re.sub(r"^```(?:markdown|md)?\s*\n?", "", body)
    body = re.sub(r"\n?```\s*$", "", body)
    return body.strip() or None


def truncate_response_after_last_search_input(response_text: str) -> str:
    """Search 轮次：截断到**第一个** Action: Search 对应的单行 Action Input 末尾。
    这样可阻止模型把多轮伪造的 Observation 一并写入，强制执行真实检索。"""
    lines = response_text.splitlines()
    i0 = _last_block_line_index(lines, "Search")
    if i0 < 0:
        return response_text
    for j in range(i0 + 1, len(lines)):
        if re.match(
            r"^\s*\*?\*?Action\s*Input\*?\*?\s*[:：]\s*[^\n]+",
            lines[j],
            re.I,
        ):
            return "\n".join(lines[: j + 1])
    return response_text


def extract_implicit_markdown_report(response_text: str) -> str | None:
    """
    模型未按格式写 Action: Finish，但直接输出了带【】引用的长文时，从第一个 `# ` 标题起截取全文。
    """
    if len(response_text.strip()) < MIN_FINISH_ARTICLE_CHARS:
        return None
    if "【" not in response_text:
        return None
    lines = response_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*#\s+\S", line):
            start = i
            break
    if start is None:
        return None
    report = "\n".join(lines[start:]).strip()
    if len(report) >= MIN_FINISH_ARTICLE_CHARS:
        return report
    return None


def is_placeholder_finish(text: str) -> bool:
    if not text:
        return True
    s = re.sub(r"\s+", "", text.strip())
    if len(s) < 80:
        return True
    if s in _FINISH_GARBAGE or text.strip() in _FINISH_GARBAGE:
        return True
    if len(text.strip()) < MIN_FINISH_ARTICLE_CHARS and not text.strip().startswith("#"):
        return True
    return False


def research_list_to_rag_dict(research_list):
    """与 evaluation 中 RAG.jsonl 一致：research 为 {\"1\": {type,title,url,content}, ...}"""
    out = {}
    for i, r in enumerate(research_list):
        out[str(i + 1)] = {
            "type": "page",
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": r.get("content") or "",
        }
    return out


def _extract_inline_observations(response_text: str, collected_research: list) -> int:
    """
    模型有时把多轮 Thought/Action/Observation 一次性全部写在同一条回复里（自伪造 Observation）。
    此函数从中解析每段 Observation 文本，填充到 collected_research，使后续 Finish 检查能通过。
    返回新增条目数。
    """
    # 匹配 Observation: ... 直到下一个 Thought:/Action: 或字符串结束
    obs_pat = re.compile(
        r"(?:^|\n)Observation\s*[:：]\s*([\s\S]*?)(?=\n(?:Thought|Action)\s*[:：]|$)",
        re.I,
    )
    added = 0
    for m in obs_pat.finditer(response_text):
        content = m.group(1).strip()
        if not content or len(content) < 10:
            continue
        idx = len(collected_research) + 1
        collected_research.append({
            "url": f"inline://observation_{idx}",
            "title": f"模型自生成摘要 {idx}",
            "content": content[:1500],
        })
        added += 1
    return added


def run_react_agent(task, client, api_config, system_prompt, max_steps=8):
    task_prompt = task.get("prompt")
    history = f"{system_prompt}\n\nTask: {task_prompt}\n"

    collected_research = []
    search_rounds = 0
    final_article = ""
    response = ""

    print(f"\n[开始处理任务 ID: {task.get('id', 'unknown')}]")

    for step in range(max_steps):
        # 历史过长时截断中间旧 Observation，保留开头（system+task）和最近上下文
        prompt = history
        if len(prompt) > MAX_HISTORY_CHARS:
            head = prompt[:20_000]
            tail = prompt[-30_000:]
            prompt = head + "\n\n[...中间较早的检索内容已截断以控制上下文长度...]\n\n" + tail
        response = chat_generate(client, api_config, prompt)

        action = parse_action_line(response)

        # Search：截断到 Action Input 单行末，禁止模型在同一轮里伪造 Observation
        if action == "Search":
            response = truncate_response_after_last_search_input(response)

        history += f"\n{response}\n"

        print(f"\n--- 第 {step + 1} 轮模型输出 ---")
        print(response)

        if action == "Search":
            action_input = parse_search_query(response)
        elif action == "Finish":
            action_input = parse_finish_body(response)
        else:
            action_input = None

        print(f"> 解析到的动作: {action}")

        if action == "Search":
            if not action_input:
                history += (
                    "\nObservation: 请在同一轮内给出单行 Action Input（搜索关键词）。"
                    "示例：Action Input: 中国中产阶级 收入 标准\n"
                )
                print("> Search 缺少 Action Input，已提示重试。")
                continue
            search_rounds += 1
            obs = execute_search(action_input, collected_research)
            history += f"\nObservation: {obs}\n"
            print(f"> 搜索完毕，将观察结果塞回历史")

        elif action == "Finish":
            if len(collected_research) == 0:
                history += (
                    "\nObservation: 尚未检索到任何来源，请先执行至少 1 轮 Search 后再 Finish。\n"
                )
                print("> Finish 被拒绝：尚无任何来源，请先 Search。")
                continue
            article = (action_input or "").strip()
            if is_placeholder_finish(article):
                alt = extract_implicit_markdown_report(response)
                if alt:
                    article = alt
                    print("> Action Input 过短，已从本回合正文抽取隐式终稿。")
                else:
                    history += (
                        "\nObservation: Action Input 必须是从 # 一级标题开始的完整 Markdown 长文，"
                        "不得以「完成」等词代替正文；或请在本轮中直接写出完整报告并包含 Action: Finish / Action Input:。\n"
                    )
                    print("> Finish 正文过短且无法从本回合抽取，已提示重试。")
                    continue
            print("> 模型决定结束任务，生成最终文章。")
            final_article = article
            break

        else:
            implicit = extract_implicit_markdown_report(response)
            # 模型常在一轮检索后直接贴终稿而不写 Action；至少 1 轮检索且有来源即可采纳，避免死循环
            if implicit and search_rounds >= 1 and len(collected_research) > 0:
                print("> 未解析到 Action，但检测到带【】与 # 标题的长文，按隐式终稿采纳。")
                final_article = implicit
                break
            history += (
                "\nObservation: 未识别到 Action 行。请严格使用：\n"
                "Thought: ...\n"
                "Action: Search 或 Action: Finish\n"
                "Action Input: （Search 时为关键词一行；Finish 时为从 # 开始的完整长文）\n"
            )
            print("> 格式解析失败，强制纠正中...")

    if not final_article:
        fallback = extract_implicit_markdown_report(response)
        final_article = fallback if fallback else response

    return final_article, collected_research


def append_references(article: str, research: list) -> str:
    """若文章末尾没有参考文献列表，则根据 research 自动补充。"""
    if not research:
        return article
    # 检查文章中是否已有参考文献章节
    if re.search(r"##?\s*参考文献", article):
        return article
    lines = []
    for i, r in enumerate(research):
        url = r.get("url") or ""
        title = r.get("title") or url
        if url.startswith("inline://"):
            continue  # 跳过模型自生成的伪来源
        lines.append(f"[{i + 1}] {url} - {title}")
    if not lines:
        return article
    ref_block = "\n\n## 参考文献\n\n" + "\n".join(lines)
    return article.rstrip() + ref_block


def process_single_task(task, client, api_config, max_steps=8):
    system_prompt = load_prompt()
    article, research = run_react_agent(task, client, api_config, system_prompt, max_steps=max_steps)

    cleaned = clean_article(article)
    cleaned = append_references(cleaned, research)   # 亡羊补牢：补充参考文献

    return {
        "id": task.get("id"),
        "prompt": task.get("prompt"),
        "article": cleaned,
        "research": research_list_to_rag_dict(research),
    }


def load_completed_ids_from_jsonl(path: str) -> set:
    """读取已有 jsonl 中每行的 id，用于断点续跑。"""
    ids: set = set()
    if not path or not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("id")
                if k is not None:
                    ids.add(k)
            except json.JSONDecodeError:
                continue
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        default=os.path.join(project_root, "conf.yaml"),
        help="API 配置（读取其中的 BASIC_MODEL），默认项目根目录 conf.yaml",
    )
    parser.add_argument("--query_file", type=str, default="evaluation/deep_research_bench/data/prompt_data/query.jsonl")
    parser.add_argument("--output_file", type=str,
                        default="evaluation/deep_research_bench/data/test_data/cleaned_data/ReAct_baseline.jsonl")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None, help="从第几条任务开始（0-indexed），配合 --limit 实现分段推理")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=8,
        help="ReAct 最大轮数（含 Search/Finish），需与长文+多轮检索匹配",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若 output_file 已存在，根据其中已出现的 id 跳过对应任务（断点续跑）；不加此参数则每次仍跑全部任务并追加写入",
    )
    args = parser.parse_args()

    api_config = load_api_config(args.conf)
    client = OpenAI(base_url=api_config["base_url"], api_key=api_config["api_key"])
    print(f"Using model: {api_config['model']} @ {api_config['base_url']}")

    tasks = []
    if os.path.exists(args.query_file):
        with open(args.query_file, "r", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
    else:
        print(f"Error: 找不到任务文件 {args.query_file}")
        return

    if args.start is not None or args.limit is not None:
        start = args.start if args.start is not None else 0
        end = (start + args.limit) if args.limit is not None else len(tasks)
        tasks = tasks[start:end]

    if args.skip_existing:
        done = load_completed_ids_from_jsonl(args.output_file)
        if done:
            before = len(tasks)
            tasks = [t for t in tasks if t.get("id") not in done]
            print(
                f"断点续跑: output 中已有 id {len(done)} 个，跳过 {before - len(tasks)} 条，剩余 {len(tasks)} 条待跑。"
            )
        if not tasks:
            print("全部任务已在 output_file 中，无需再跑。")
            return

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_task, task, client, api_config, args.max_steps): task
            for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Running ReAct Agent"):
            try:
                res = future.result()
                with open(args.output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Task failed: {e}")


if __name__ == "__main__":
    main()