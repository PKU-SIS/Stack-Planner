import sys, asyncio
sys.path.insert(0, "src")
from cite_verify.num_verify import cite_verify_report

query = """## 脱贫攻坚报告
习近平总书记指出，巩固脱贫成果难度很大【1】。
全国累计脱贫人口8000万【2】，贫困发生率降至0.6%【2】。
乡村振兴战略全面实施，成效显著【3】。
这句话完全无关却挂了错误引用【4】。
这是没有引用的纯描述句子。"""

docs = {
    "1": {
        "source": "脱贫攻坚政策文件.docx",
        "content": "脱贫攻坚战不是轻轻松松一冲锋就能打赢的。巩固脱贫成果难度很大，需要长期坚持。扶贫工作要坚持精准方略。",
    },
    "2": {
        "source": "国家统计局报告.docx",
        "content": "全国累计实现脱贫人口8000万，如期完成脱贫攻坚目标任务。贫困发生率降至0.6%，历史性解决了绝对贫困问题。",
    },
    "3": {
        "source": "乡村振兴战略规划.docx",
        "content": "乡村振兴战略深入推进，农业农村现代化建设全面加速。农村基础设施显著改善，农民生活水平持续提升。",
    },
    "4": {
        "source": "国际军事分析.docx",
        "content": "全球军事格局深刻演变，大国博弈持续加剧。",
    },
}

result = asyncio.run(cite_verify_report(query, docs))

print("=" * 60)
print("【1】正文句子列表（含 [doc_id-sent_id] 引用）")
print("=" * 60)
for i, s in enumerate(result["sentences"]):
    print("  [%d] %s" % (i, s.strip()))

print()
print("=" * 60)
print("【2】sentence_map（引用查找表）")
print("=" * 60)
for k, v in result["sentence_map"].items():
    print("  %s → %s" % (k, v["content"].strip()))

print()
print("=" * 60)
print("【3】docs_display（前端溯源展示）")
print("=" * 60)
for doc_id, display in result["docs_display"].items():
    print("\n  doc_id=%s:" % doc_id)
    for line in display.split("\n"):
        print("    %s" % line)

print()
print("=" * 60)
print("【4】join 后完整正文")
print("=" * 60)
print("".join(result["sentences"]))
