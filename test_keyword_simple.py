#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试关键词提取功能
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


def test_keyword_extraction():
    """测试关键词提取功能"""
    logger.info("开始测试关键词提取功能")

    # Mock LLM
    with patch("src.llms.llm.get_llm_by_type", return_value=MockLLM()):
        # 测试普通关键词提取
        logger.info("\n============================================================")
        logger.info("测试普通关键词提取...")

        text = "人工智能在医学影像诊断技术中的应用，包括CT扫描、MRI图像分析和X光片识别"
        keywords = keyword_extractor.extract_keywords(text)

        logger.info(f"输入文本: {text}")
        logger.info(f"提取的关键词: {keywords}")
        assert len(keywords) > 0, "应该提取到关键词"
        logger.info("✅ 普通关键词提取测试通过")

        # 测试Level-1关键词提取
        logger.info("\n============================================================")
        logger.info("测试Level-1关键词提取...")

        main_node = {
            "title": "人工智能技术概述",
            "description": "关于AI技术的基础理论和应用发展",
        }

        child_nodes = [
            {
                "title": "机器学习基础理论",
                "description": "监督学习、无监督学习和强化学习的核心理论",
            },
            {
                "title": "深度学习技术发展",
                "description": "神经网络架构的演进和最新突破",
            },
            {"title": "自然语言处理进展", "description": "NLP领域的技术发展和应用创新"},
        ]

        level1_result = keyword_extractor.extract_level1_keywords(
            main_node, child_nodes
        )

        logger.info(f"主章节: {main_node['title']}")
        logger.info(f"子章节数量: {len(child_nodes)}")
        logger.info(f"Level-1关键词: {level1_result['keywords']}")
        logger.info(f"置信度: {level1_result['confidence']}")
        logger.info(f"推理过程: {level1_result['reasoning']}")

        assert len(level1_result["keywords"]) > 0, "应该提取到Level-1关键词"
        assert level1_result["confidence"] > 0, "置信度应该大于0"
        logger.info("✅ Level-1关键词提取测试通过")

        logger.info(
            "\n================================================================================\n测试结果汇总\n================================================================================"
        )
        logger.info("通过测试: 2/2")
        logger.info("🎉 关键词提取功能测试通过！")


if __name__ == "__main__":
    test_keyword_extraction()
