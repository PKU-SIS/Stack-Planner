import json
import logging
from typing import Optional
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)


def generate_observation(self, document: FactStructDocument) -> Optional[dict]:
    """
    为单个文档生成 Observation 结构（单次 LLM 调用）

    参数:
        document: FactStructDocument 实例

    返回:
        observation dict，如果失败返回 None
    """

    if not document.text:
        logger.warning(f"Document {document.id} has empty text.")
        return None

    prompt = f"""
    请对以下文章进行结构化抽象重构，生成一个新的 Observation 树结构。

    目标：
    - 提炼文章的核心逻辑
    - 重新组织信息结构
    - 删除冗余内容
    - 强化逻辑关系

    重要规则：
    1. 输出必须是合法 JSON
    2. 只能输出 JSON，不要包含任何解释性文字
    3. 不能与原文目录结构一致
    4. 必须进行抽象，而不是改写原句
    5. 叶子节点必须是高度概括表达
    6. 层级不超过 4 层
    7. 每个节点不超过 20 字
    8. 所有重要“数字、比例、阶段、数量级、对比关系”必须抽象为关键观点节点
    9. 不能出现原文完整句子
    10. 不能增加除 title 和 children 之外的字段

    输出格式必须严格如下：

    {{
        "title": "文章核心主题抽象",
        "children": [
            {{
                "title": "关键逻辑结构",
                "children": [
                    {{
                        "title": "关键观点1",
                        "children": []
                    }},
                    {{
                        "title": "关键观点2",
                        "children": []
                    }}
                ]
            }}
        ]
    }}

文章内容如下：
--------------------
{document.text}
"""

    try:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        content = response.content.strip()

        # 尝试解析 JSON
        observation = json.loads(content)

        return observation

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON returned for document {document.id}")
        return None

    except Exception as e:
        logger.error(f"Failed to generate observation for {document.id}: {e}")
        return None
