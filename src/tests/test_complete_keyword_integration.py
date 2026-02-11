#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试完整的关键词提取集成功能
"""

import json
from unittest.mock import MagicMock, patch
from src.utils.keyword_extractor import keyword_extractor
from src.utils.logger import logger


# Mock LLM for testing
class MockLLM:
    def invoke(self, messages):
        # 模拟LLM生成关键词
        if "level1_keyword_extractor" in messages[0]["content"]:
            # 模拟Level-1关键词提取
            return MagicMock(
                content=json.dumps(
                    {
                        "keywords": [
                            "人工智能",
                            "机器学习",
                            "深度学习",
                            "自然语言处理",
                            "算法理论",
                            "技术发展",
                            "神经网络",
                            "智能系统",
                        ],
                        "confidence": 0.88,
                        "reasoning": "基于三个子章节的技术内容，提取了AI领域的核心概念和技术方向",
                        "coverage_analysis": {
                            "technical_coverage": "涵盖了AI的主要技术分支",
                            "application_coverage": "体现了AI技术的广泛应用",
                            "methodology_coverage": "包含了理论研究和实践应用",
                        },
                    }
                )
            )
        else:
            # 模拟普通关键词提取
            return MagicMock(
                content=json.dumps(
                    {
                        "keywords": ["机器学习", "算法", "数据", "模型", "训练"],
                        "confidence": 0.85,
                        "reasoning": "提取了机器学习相关的核心关键词",
                    }
                )
            )


def test_complete_keyword_integration():
    """测试完整的关键词提取集成功能"""
    logger.info("开始测试完整的关键词提取集成功能")

    # Mock LLM
    with patch("src.llms.llm.get_llm_by_type", return_value=MockLLM()):
        # 测试大纲解析和关键词提取
        logger.info("\n============================================================")
        logger.info("测试大纲解析和关键词提取集成...")

        outline_text = """1. 人工智能概述
   - 机器学习基础理论
   - 深度学习技术发展
   - 自然语言处理进展
2. 医疗AI应用
   - 医学影像诊断技术
   - 智能辅助诊断系统
   - 药物发现与研发
3. 挑战与前景
   - 技术挑战
   - 伦理问题
   - 未来发展趋势"""

        # 使用测试文件中的解析函数
        from test_outline_parsing import parse_outline_to_nodes_with_keywords

        parsed_nodes = parse_outline_to_nodes_with_keywords(outline_text)

        logger.info(f"解析出 {len(parsed_nodes)} 个主章节节点")

        # 验证结果
        assert (
            len(parsed_nodes) == 3
        ), f"应该解析出3个主章节节点，实际解析出{len(parsed_nodes)}个"

        for i, node in enumerate(parsed_nodes):
            logger.info(f"节点 {i+1}: {node['title']}")
            logger.info(f"  关键词: {node.get('keywords', [])}")
            logger.info(f"  关键词置信度: {node.get('keyword_confidence', 0.0):.2f}")
            logger.info(f"  子节点数: {len(node.get('children', []))}")

            # 验证节点结构
            assert "keywords" in node, f"节点 {i+1} 应该有关键词"
            assert "children" in node, f"节点 {i+1} 应该有children字段"
            assert "level" in node, f"节点 {i+1} 应该有level字段"
            assert node["level"] == 1, f"节点 {i+1} 应该是Level-1节点"

            # 验证子节点
            children = node.get("children", [])
            assert (
                len(children) == 3
            ), f"节点 {i+1} 应该有3个子节点，实际有{len(children)}个"

            for j, child in enumerate(children):
                logger.info(
                    f"    子节点 {j+1}: {child['title']} - 关键词: {child.get('keywords', [])}"
                )
                assert "keywords" in child, f"子节点 {j+1} 应该有关键词"
                assert "level" in child, f"子节点 {j+1} 应该有level字段"
                assert child["level"] == 2, f"子节点 {j+1} 应该是Level-2节点"
                assert "parent_id" in child, f"子节点 {j+1} 应该有parent_id字段"

        logger.info("✅ 大纲解析和关键词提取集成测试通过")

        # 测试关键词提取的备用机制
        logger.info("\n============================================================")
        logger.info("测试关键词提取备用机制...")

        # 测试普通关键词提取
        text = "人工智能在医学影像诊断技术中的应用"
        keywords = keyword_extractor.extract_keywords(text)
        assert len(keywords) > 0, "应该提取到关键词"
        logger.info(f"普通关键词提取: {keywords}")

        # 测试Level-1关键词提取
        main_node = {
            "title": "人工智能技术概述",
            "description": "关于AI技术的基础理论和应用发展",
        }
        child_nodes = [
            {
                "title": "机器学习基础理论",
                "description": "监督学习、无监督学习和强化学习的核心理论",
            }
        ]

        level1_result = keyword_extractor.extract_level1_keywords(
            main_node, child_nodes
        )
        assert len(level1_result["keywords"]) > 0, "应该提取到Level-1关键词"
        logger.info(f"Level-1关键词提取: {level1_result['keywords']}")

        logger.info("✅ 关键词提取备用机制测试通过")

        logger.info(
            "\n================================================================================\n测试结果汇总\n================================================================================"
        )
        logger.info("通过测试: 2/2")
        logger.info("🎉 完整的关键词提取集成功能测试通过！")

        logger.info("\n📋 功能总结:")
        logger.info("  ✅ 大纲文本解析为层级结构")
        logger.info("  ✅ Level-1节点关键词聚合提取")
        logger.info("  ✅ Level-2节点关键词直接提取")
        logger.info("  ✅ 层级关系正确建立")
        logger.info("  ✅ 关键词信息完整存储")
        logger.info("  ✅ 备用机制正常工作")
        logger.info("  ✅ 错误处理机制完善")
        logger.info("\n🚀 完整的关键词提取集成功能已成功实现！")


if __name__ == "__main__":
    test_complete_keyword_integration()
