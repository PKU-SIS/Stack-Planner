#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态大纲优化测试脚本

测试动态大纲优化功能的各个组件
"""

import sys
import os
import json
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.coverage_calculator import CoverageCalculator
from src.utils.outline_reviser import OutlineReviser
from src.tools.enhanced_search_docs import EnhancedSearchDocs
from src.graph.outline_optimizer_node import OutlineOptimizer


def test_coverage_calculator():
    """测试覆盖度计算器"""
    print("=" * 50)
    print("测试覆盖度计算器")
    print("=" * 50)

    # 创建测试数据
    outline_nodes = [
        {
            "title": "人工智能的发展历史",
            "description": "从早期符号主义到现代深度学习的演进过程",
        },
        {
            "title": "机器学习算法",
            "description": "监督学习、无监督学习和强化学习的主要算法",
        },
        {
            "title": "深度学习应用",
            "description": "在计算机视觉、自然语言处理等领域的应用",
        },
    ]

    retrieved_docs = [
        {
            "content": "人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试。",
            "source": "AI历史文献",
        },
        {
            "content": "机器学习包括监督学习、无监督学习和强化学习三大类。",
            "source": "ML教材",
        },
        {
            "content": "深度学习在图像识别和语音识别方面取得了突破性进展。",
            "source": "DL研究论文",
        },
    ]

    try:
        # 测试覆盖度计算
        calculator = CoverageCalculator()

        # 计算整体互信息
        overall_mi = calculator.calculate_mutual_information(
            outline_nodes, retrieved_docs
        )
        print(f"整体互信息: {overall_mi:.4f}")

        # 计算节点级覆盖度
        node_coverage = calculator.calculate_node_coverage(
            outline_nodes, retrieved_docs
        )
        print(f"节点覆盖度结果:")
        for node_info in node_coverage:
            print(
                f"  - {node_info['title']}: MI={node_info['local_mi']:.4f}, "
                f"相似度={node_info['max_similarity']:.4f}, "
                f"低覆盖={node_info['is_low_coverage']}"
            )

        print("✅ 覆盖度计算器测试通过")
        return True

    except Exception as e:
        print(f"❌ 覆盖度计算器测试失败: {e}")
        return False


def test_enhanced_search_docs():
    """测试增强检索模块"""
    print("=" * 50)
    print("测试增强检索模块")
    print("=" * 50)

    outline_nodes = [
        {
            "title": "量子计算原理",
            "description": "量子比特、量子叠加和量子纠缠的基本概念",
        },
        {"title": "量子算法", "description": "Shor算法、Grover算法等经典量子算法"},
    ]

    try:
        searcher = EnhancedSearchDocs(top_k=3)

        # 测试检索功能
        docs = searcher.search_for_outline_optimization(outline_nodes, iteration=0)
        print(f"检索到 {len(docs)} 个文档")

        if docs:
            print("检索结果示例:")
            for i, doc in enumerate(docs[:2]):  # 只显示前2个
                print(f"  文档 {i+1}:")
                print(f"    来源: {doc.get('source', '未知')}")
                print(f"    内容: {doc.get('content', '')[:100]}...")
                print(f"    目标节点: {doc.get('target_node_title', '')}")

        # 测试文档分组
        grouped_docs = searcher.group_docs_by_node(docs)
        print(f"按节点分组: {list(grouped_docs.keys())}")

        print("✅ 增强检索模块测试通过")
        return True

    except Exception as e:
        print(f"❌ 增强检索模块测试失败: {e}")
        return False


def test_outline_reviser():
    """测试大纲修纲模块"""
    print("=" * 50)
    print("测试大纲修纲模块")
    print("=" * 50)

    original_outline = """
    1. 人工智能概述
       - 定义和发展历程
       - 主要技术分支
    
    2. 机器学习基础
       - 监督学习算法
       - 无监督学习算法
    
    3. 深度学习应用
       - 计算机视觉
       - 自然语言处理
    """

    low_coverage_nodes = [
        {"title": "人工智能概述", "local_mi": 0.05, "max_similarity": 0.3}
    ]

    retrieved_docs = [
        {
            "content": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "source": "AI教科书",
        }
    ]

    try:
        reviser = OutlineReviser()

        # 测试修纲功能
        result = reviser.revise_outline(
            original_outline, low_coverage_nodes, retrieved_docs
        )

        print("修纲结果:")
        print(f"  优化后大纲长度: {len(result.get('optimized_outline', ''))}")
        print(f"  变更数量: {len(result.get('changes_made', []))}")

        if result.get("changes_made"):
            print("  变更详情:")
            for change in result["changes_made"]:
                print(
                    f"    - {change.get('node_title', '')}: {change.get('reason', '')}"
                )

        print("✅ 大纲修纲模块测试通过")
        return True

    except Exception as e:
        print(f"❌ 大纲修纲模块测试失败: {e}")
        return False


def test_outline_optimizer():
    """测试完整的大纲优化器"""
    print("=" * 50)
    print("测试完整的大纲优化器")
    print("=" * 50)

    outline = """
    1. 区块链技术原理
       - 分布式账本
       - 共识机制
    
    2. 加密货币发展
       - 比特币的诞生
       - 以太坊的创新
    
    3. 区块链应用场景
       - 金融领域
       - 供应链管理
    """

    outline_nodes = [
        {"title": "区块链技术原理", "description": "分布式账本和共识机制的基本概念"},
        {"title": "加密货币发展", "description": "从比特币到以太坊的发展历程"},
        {"title": "区块链应用场景", "description": "在金融和供应链等领域的应用"},
    ]

    # 模拟状态对象
    class MockState:
        def get(self, key, default=None):
            return default

    try:
        optimizer = OutlineOptimizer()
        mock_state = MockState()

        # 测试完整优化流程
        result = optimizer.optimize_outline(outline, outline_nodes, mock_state)

        print("优化结果:")
        print(f"  优化轮次: {result.get('optimization_iterations', 0)}")
        print(f"  最终覆盖度: {result.get('final_coverage', 0):.4f}")
        print(f"  检索文档数: {result.get('total_docs_retrieved', 0)}")

        optimization_history = result.get("optimization_history", [])
        if optimization_history:
            print("  优化历史:")
            for hist in optimization_history:
                print(
                    f"    第{hist['iteration']}轮: 覆盖度={hist['coverage']:.4f}, "
                    f"低覆盖节点={hist['low_coverage_count']}"
                )

        print("✅ 完整大纲优化器测试通过")
        return True

    except Exception as e:
        print(f"❌ 完整大纲优化器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试动态大纲优化功能")
    print("=" * 60)

    test_results = []

    # 运行各项测试
    test_results.append(test_coverage_calculator())
    test_results.append(test_enhanced_search_docs())
    test_results.append(test_outline_reviser())
    test_results.append(test_outline_optimizer())

    # 汇总测试结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(test_results)
    total = len(test_results)

    print(f"通过测试: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！动态大纲优化功能实现成功！")
        return True
    else:
        print("❌ 部分测试失败，请检查相关模块")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
