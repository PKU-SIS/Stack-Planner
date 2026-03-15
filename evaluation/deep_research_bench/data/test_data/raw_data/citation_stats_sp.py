#!/usr/bin/env python3
"""
引用完整性统计脚本 - 适配SP.jsonl格式
用于统计"一段一研最后生成"方案的引用保留情况

使用方法:
    python citation_stats_sp.py SP.jsonl

数据格式要求:
    SP.jsonl文件,每行一个JSON对象:
    {
        "id": 样本ID,
        "prompt": "用户查询",
        "article": "生成的完整内容(包含researcher检索、outline大纲、reporter报告)",
        "research": []
    }
"""

import json
import re
import sys
from collections import Counter


def extract_citations_from_article(article):
    """
    从article字段中提取引用统计

    Args:
        article: article字段内容

    Returns:
        (input_docs, output_citations, final_report)
    """
    # 提取最终报告
    reporter_match = re.search(r'委派reporter执行:.*?\n\n(.*)报告生成完成', article, re.DOTALL)

    if not reporter_match:
        return 0, 0, ""

    final_report = reporter_match.group(1)

    # 提取检索到的文档数
    # 方法1: 从【链接1】【链接2】等标记统计
    doc_links = re.findall(r'【链接(\d+)】', article)
    input_docs = len(set(doc_links))

    # 方法2: 如果方法1没有结果,从JSON数组统计
    if input_docs == 0:
        docs_match = re.search(r'\[\{"url".*?\}\]', article, re.DOTALL)
        if docs_match:
            try:
                docs_json = docs_match.group(0)
                docs = json.loads(docs_json)
                input_docs = len(docs)
            except:
                pass

    # 统计最终报告中的引用
    citations = re.findall(r'【(\d+)】', final_report)
    unique_citations = len(set(citations))

    return input_docs, unique_citations, final_report


def analyze_sp_data(data_file):
    """
    分析SP.jsonl文件

    Args:
        data_file: 文件路径

    Returns:
        统计结果列表
    """
    results = []

    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                query = data.get('prompt', '')
                article = data.get('article', '')

                # 提取引用统计
                input_docs, output_citations, final_report = extract_citations_from_article(article)

                # 计算保留率
                retention_rate = (output_citations / input_docs * 100) if input_docs > 0 else 0

                results.append({
                    'id': data.get('id', line_num),
                    'query': query[:60] + '...' if len(query) > 60 else query,
                    'input_docs': input_docs,
                    'output_citations': output_citations,
                    'retention_rate': retention_rate,
                    'lost_citations': max(0, input_docs - output_citations)
                })

                print(
                    f"✅ 处理样本 {data.get('id', line_num)}: 输入{input_docs}文档, 输出{output_citations}引用, 保留率{retention_rate:.1f}%")

            except Exception as e:
                print(f"❌ 第{line_num}行处理失败: {e}")
                continue

    return results


def print_summary(results):
    """打印汇总统计"""
    if not results:
        print("\n❌ 没有有效数据")
        return

    print("\n" + "=" * 90)
    print("引用完整性统计结果 - 方案B：一段一研最后生成")
    print("=" * 90)
    print()

    # 计算平均值
    avg_input = sum(r['input_docs'] for r in results) / len(results)
    avg_output = sum(r['output_citations'] for r in results) / len(results)
    avg_retention = sum(r['retention_rate'] for r in results) / len(results)
    total_lost = sum(r['lost_citations'] for r in results)

    # 打印详细表格
    print(f"{'ID':<5} {'Query':<45} {'输入文档':<10} {'输出引用':<10} {'保留率':<10} {'丢失数':<10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['id']:<5} {r['query']:<45} {r['input_docs']:<10} {r['output_citations']:<10} {r['retention_rate']:.1f}%{'':<5} {r['lost_citations']:<10}")

    print("-" * 90)
    print(f"{'平均':<5} {'':<45} {avg_input:<10.1f} {avg_output:<10.1f} {avg_retention:<10.1f}%")
    print()

    # 打印汇总统计
    print("=" * 90)
    print("汇总统计")
    print("=" * 90)
    print(f"总样本数: {len(results)}")
    print(f"平均输入文档数: {avg_input:.1f}")
    print(f"平均输出引用数: {avg_output:.1f}")
    print(f"引用保留率: {avg_retention:.1f}%")
    print(f"总丢失引用数: {total_lost}")
    print()

    # 引用保留率分布
    high_retention = sum(1 for r in results if r['retention_rate'] >= 80)
    medium_retention = sum(1 for r in results if 50 <= r['retention_rate'] < 80)
    low_retention = sum(1 for r in results if r['retention_rate'] < 50)

    print("引用保留率分布:")
    print(f"  高保留率 (≥80%): {high_retention} 个样本")
    print(f"  中保留率 (50-80%): {medium_retention} 个样本")
    print(f"  低保留率 (<50%): {low_retention} 个样本")
    print()

    # Case Study数据提取
    print("=" * 90)
    print("Case Study 数据")
    print("=" * 90)
    if results:
        # 找一个保留率高的样本
        good_sample = max(results, key=lambda x: x['retention_rate'])
        print(f"\n最佳样本 (ID: {good_sample['id']}):")
        print(f"  Query: {good_sample['query']}")
        print(f"  输入文档数: {good_sample['input_docs']}")
        print(f"  输出引用数: {good_sample['output_citations']}")
        print(f"  引用保留率: {good_sample['retention_rate']:.1f}%")
        print(f"  丢失引用数: {good_sample['lost_citations']}")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python citation_stats_sp.py SP.jsonl")
        print()
        print("数据格式要求: SP.jsonl文件，每行一个JSON对象，包含:")
        print("  - id: 样本ID")
        print("  - prompt: 用户查询")
        print("  - article: 生成的完整内容")
        return

    data_file = sys.argv[1]
    print(f"正在处理文件: {data_file}")
    print()

    results = analyze_sp_data(data_file)
    print_summary(results)


if __name__ == "__main__":
    main()
