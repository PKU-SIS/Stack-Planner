import sys, asyncio, re
sys.path.insert(0, "src")
from cite_verify.num_verify import cite_verify_report

PASS = "PASS"
FAIL = "FAIL"

results = []

def run_case(title, query, docs, expect_keep=None, expect_drop=None):
    sentences = asyncio.run(cite_verify_report(query, docs))
    full = "".join(sentences)
    kept = set(re.findall(r'【(\d+)】', full))
    ok = True
    fails = []
    if expect_keep:
        for r in expect_keep:
            if r not in kept:
                fails.append(f"期望保留【{r}】但未保留")
                ok = False
    if expect_drop:
        for r in expect_drop:
            if r in kept:
                fails.append(f"期望丢弃【{r}】但未丢弃")
                ok = False
    status = PASS if ok else FAIL
    results.append((title, status, query.strip(), full.strip(), sorted(kept), fails))

# ── A 高度匹配 ──────────────────────────────────────────
run_case(
    "A1 直接引用原文",
    "脱贫攻坚工作需要持续加强【1】。",
    {"1": {"content": "脱贫攻坚工作需要持续加强，绝不能有任何松懈。"}},
    expect_keep=["1"],
)
run_case(
    "A2 同句多引用 各自高度匹配",
    "巩固脱贫成果难度很大【1】，脱贫攻坚工作需要持续加强【2】。",
    {
        "1": {"content": "巩固脱贫成果难度很大，是当前扶贫工作的重中之重。"},
        "2": {"content": "脱贫攻坚工作需要持续加强，必须常抓不懈。"},
    },
    expect_keep=["1", "2"],
)
run_case(
    "A3 近义词表述",
    "全面建成小康社会取得历史性成就【1】。",
    {"1": {"content": "全面建成小康社会，实现了历史性跨越，是重大历史成就。"}},
    expect_keep=["1"],
)
run_case(
    "A4 习近平引用",
    "习近平总书记指出，巩固脱贫成果难度很大【1】。",
    {"1": {"content": "脱贫攻坚战不是轻轻松松一冲锋就能打赢的，巩固脱贫成果难度很大，需要坚持不懈。"}},
    expect_keep=["1"],
)

# ── B 完全不相关 ────────────────────────────────────────
run_case(
    "B1 引文主题无关",
    "我国粮食产量连续多年丰收【1】。",
    {"1": {"content": "城镇化率持续提升，截至2022年达到65.22%。"}},
    expect_drop=["1"],
)
run_case(
    "B2 引文编号不在docs中",
    "绿色发展理念深入人心【99】。",
    {"1": {"content": "绿色发展是新发展理念的重要组成部分。"}},
    expect_drop=["99"],
)
run_case(
    "B3 领域完全相反",
    "工业化进程加快推进【1】。",
    {"1": {"content": "农村土地流转面积逐步扩大，农业规模化经营水平提高。"}},
    expect_drop=["1"],
)
run_case(
    "B4 军事内容配脱贫句子",
    "乡村振兴战略全面实施成效显著【1】。",
    {"1": {"content": "全球气候变化导致极端天气事件频发，国际安全形势日趋复杂。"}},
    expect_drop=["1"],
)

# ── C 部分保留部分丢弃 ──────────────────────────────────
run_case(
    "C1 两引用：一相关一不相关",
    "乡村振兴战略全面实施【1】，城乡居民收入差距持续缩小【2】。",
    {
        "1": {"content": "乡村振兴战略深入推进，农业农村现代化建设全面加速。"},
        "2": {"content": "全球气候变化导致极端天气频发，国际安全形势复杂严峻。"},
    },
    expect_keep=["1"],
    expect_drop=["2"],
)
run_case(
    "C2 三引用：中间一个不相关",
    "生态文明建设成效显著【1】，经济总量居世界前列【2】，绿色低碳转型加快推进【3】。",
    {
        "1": {"content": "生态文明建设取得重大成就，绿水青山就是金山银山理念深入人心。"},
        "2": {"content": "某国军事预算持续增加，国防开支占GDP比重创历史新高。"},
        "3": {"content": "绿色低碳发展成为国家战略，清洁能源比重持续上升。"},
    },
    expect_keep=["1", "3"],
    expect_drop=["2"],
)

# ── D 标题不处理 ─────────────────────────────────────────
run_case(
    "D1 Markdown 标题原样保留",
    "## 第一章 脱贫攻坚成果\n相关工作持续推进【1】。",
    {"1": {"content": "相关工作持续推进，取得积极成效。"}},
    expect_keep=["1"],
)

# ── E 无引用句子 ─────────────────────────────────────────
run_case(
    "E1 无引用原样通过",
    "这是一段没有引用的背景介绍文字。",
    {"1": {"content": "任意内容。"}},
)

# ── F 数字分支 ───────────────────────────────────────────
run_case(
    "F1 数字匹配 走数字分支",
    "全国累计脱贫人口8000万【1】。",
    {"1": {"content": "全国累计实现脱贫人口8000万，如期完成脱贫攻坚目标任务。"}},
    expect_keep=["1"],
)
run_case(
    "F2 数字不匹配 应丢弃",
    "城镇登记失业率控制在5%以内【1】。",
    {"1": {"content": "经济增速保持在6%左右，实现高质量发展。"}},
    expect_drop=["1"],
)

# ── G 边界 ───────────────────────────────────────────────
run_case(
    "G1 引文极长但包含匹配片段",
    "坚持以人民为中心的发展思想【1】。",
    {"1": {"content": "党的十八大以来，坚持以人民为中心的发展思想，把人民对美好生活的向往作为奋斗目标，着力解决发展不平衡不充分问题，扎实推动共同富裕，不断增强人民群众获得感、幸福感、安全感，人民生活全方位改善。"}},
    expect_keep=["1"],
)
run_case(
    "G2 引文极短",
    "发展是硬道理【1】。",
    {"1": {"content": "发展是硬道理，是解决一切问题的基础。"}},
    expect_keep=["1"],
)
run_case(
    "G3 相似文字稍微变形",
    "坚决打赢脱贫攻坚战，确保如期完成目标任务【1】。",
    {"1": {"content": "全党全社会坚定信心，尽锐出战，确保如期完成脱贫攻坚目标任务。"}},
    expect_keep=["1"],
)

# ── 打印结果 ─────────────────────────────────────────────
total = len(results)
passed = sum(1 for r in results if r[1] == PASS)
print()
print("=" * 65)
print(f"  N-gram 非数字引用校验测试结果  ({passed}/{total} 通过)")
print("=" * 65)
for title, status, query, output, kept, fails in results:
    mark = "✅" if status == PASS else "❌"
    print(f"\n{mark} {title}")
    print(f"   输入: {query[:60]}{'...' if len(query)>60 else ''}")
    print(f"   输出: {output[:70]}{'...' if len(output)>70 else ''}")
    print(f"   保留引用: {kept if kept else '（无）'}")
    for f in fails:
        print(f"   >>> {f}")
print()
print("=" * 65)
print(f"  汇总: {passed} 通过 / {total-passed} 失败 / {total} 合计")
print("=" * 65)
