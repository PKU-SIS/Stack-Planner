import sys
sys.path.insert(0, "src")
from cite_verify.num_verify import (
    find_best_text_excerpt, extract_query_for_citation,
    NGRAM_DISCARD_THRESHOLD, NGRAM_LLM_THRESHOLD, NGRAM_HIGH_CONF_THRESHOLD
)

cases = [
    ("B1 粮食丰收 vs 城镇化",
     "我国粮食产量连续多年丰收【1】。",
     "城镇化率持续提升，截至2022年达到65.22%。"),
    ("B3 工业化 vs 农村土地",
     "工业化进程加快推进【1】。",
     "农村土地流转面积逐步扩大，农业规模化经营水平提高。"),
    ("B4 乡村振兴 vs 气候变化",
     "乡村振兴战略全面实施成效显著【1】。",
     "全球气候变化导致极端天气频发，国际安全形势复杂严峻。"),
    ("C1不相关 收入差距 vs 气候变化",
     "乡村振兴战略全面实施【1】，城乡居民收入差距持续缩小【2】。",
     "全球气候变化导致极端天气频发，国际安全形势复杂严峻。"),
    ("C2不相关 经济总量 vs 军事预算",
     "生态文明建设成效显著【1】，经济总量居世界前列【2】，绿色低碳转型加快推进【3】。",
     "某国军事预算持续增加，国防开支占GDP比重创历史新高。"),
    ("对照A1 直接匹配",
     "脱贫攻坚工作需要持续加强【1】。",
     "脱贫攻坚工作需要持续加强，绝不能有任何松懈。"),
    ("对照A4 高度相关",
     "习近平总书记指出，巩固脱贫成果难度很大【1】。",
     "脱贫攻坚战不是轻轻松松一冲锋就能打赢的，巩固脱贫成果难度很大。"),
    ("对照A3 近义词",
     "全面建成小康社会取得历史性成就【1】。",
     "全面建成小康社会，实现了历史性跨越，是重大历史成就。"),
]

print("\n分数分析（当前 keep 阈值 >= 0.18）")
print("-" * 70)
print("  %-32s %-18s %6s  结论" % ("场景", "专属query", "score"))
print("-" * 70)
for name, q, content in cases:
    ref = 1 if "【1】" in q else 2
    sq = extract_query_for_citation(q, ref)
    r = find_best_text_excerpt(source=content, query=sq)
    score = r["score"] if r else 0.0
    if score >= NGRAM_HIGH_CONF_THRESHOLD:
        verdict = "保留 (>=0.35)"
    elif score >= NGRAM_LLM_THRESHOLD:
        verdict = "保留 (>=0.18 ← 假阳性?)"
    elif score >= NGRAM_DISCARD_THRESHOLD:
        verdict = "丢弃 (0.08~0.18)"
    else:
        verdict = "丢弃 (<0.08)"
    sq_short = sq[:16]
    print("  %-32s %-18s %6.3f  %s" % (name, sq_short, score, verdict))
print()
print("结论：如果把 keep 阈值提升到 0.25 或 0.30，假阳性会减少多少？")
print("-" * 70)
thresholds = [0.20, 0.25, 0.30, 0.35]
for th in thresholds:
    keeps = []
    drops = []
    for name, q, content in cases:
        ref = 1 if "【1】" in q else 2
        sq = extract_query_for_citation(q, ref)
        r = find_best_text_excerpt(source=content, query=sq)
        score = r["score"] if r else 0.0
        if score >= th:
            keeps.append(name)
        else:
            drops.append(name)
    print("  阈值 %.2f: 保留 %d 个, 丢弃 %d 个" % (th, len(keeps), len(drops)))
    for k in keeps:
        print("    保留: %s" % k)
