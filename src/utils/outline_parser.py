"""
大纲解析工具模块
用于解析文档大纲，提取章节信息，管理研究进度
"""

import re
from typing import List, Dict, Optional
from src.utils.logger import logger


def parse_outline(outline_text: str) -> List[Dict[str, str]]:
    """
    解析大纲文本，提取章节列表

    Args:
        outline_text: 大纲文本，格式如：
            "1. 章节标题\n  内容...\n2. 章节标题\n  内容..."

    Returns:
        章节列表，每个元素包含：
        {
            'number': '1',           # 章节编号
            'title': '章节标题',      # 章节标题
            'content': '完整内容',    # 章节完整内容（可选）
        }

    示例：
        outline = "1. 全国脱贫攻坚战重大成就概述\n◦ 内容..."
        chapters = parse_outline(outline)
        # 返回: [{'number': '1', 'title': '全国脱贫攻坚战重大成就概述', 'content': '...'}]
    """
    if not outline_text:
        logger.warning("大纲文本为空")
        return []

    chapters = []

    # 匹配章节标题（支持 "1. xxx" 或 "一、xxx" 等格式）
    # 优先匹配数字编号格式：1. 标题 或 1、标题
    pattern = r'(?:^|\n)\s*(\d+)\s*[.、]\s*([^\n]+)'
    matches = list(re.finditer(pattern, outline_text))

    for i, match in enumerate(matches):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()

        # 提取章节内容（从当前标题到下一个标题之间的内容）
        start_pos = match.end()
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
            chapter_content = outline_text[start_pos:end_pos].strip()
        else:
            chapter_content = outline_text[start_pos:].strip()

        chapters.append({
            'number': chapter_num,
            'title': chapter_title,
            'content': chapter_content,
        })

    logger.info(f"大纲解析完成，共 {len(chapters)} 个章节")
    return chapters


def get_next_chapter_index(state: dict) -> int:
    """
    获取下一个需要研究的段落索引

    Args:
        state: 当前状态，包含 'current_chapter_index' 字段

    Returns:
        下一个段落的索引（从 0 开始）
    """
    return state.get('current_chapter_index', 0)


def get_chapter_task_description(chapter: Dict[str, str], index: int) -> str:
    """
    生成研究任务描述，用于传递给 Researcher Agent

    Args:
        chapter: 章节信息字典，包含 number, title, content
        index: 章节索引（从 0 开始）

    Returns:
        任务描述字符串，格式如：
            "研究第1章: 全国脱贫攻坚战重大成就概述\n\n章节内容：..."

    注意：
        - 任务描述中必须包含"第X章"字样，以便 Reporter 从 Memory Stack 中提取
    """
    chapter_num = chapter['number']
    chapter_title = chapter['title']
    chapter_content = chapter.get('content', '')

    # 构建任务描述
    task_description = f"研究第{chapter_num}章: {chapter_title}"

    if chapter_content:
        # 限制内容长度，避免过长
        max_content_length = 500
        if len(chapter_content) > max_content_length:
            chapter_content = chapter_content[:max_content_length] + "..."
        task_description += f"\n\n章节内容：\n{chapter_content}"

    return task_description


def is_all_chapters_researched(state: dict, outline_text: str) -> bool:
    """
    判断是否所有章节都已完成研究

    Args:
        state: 当前状态
        outline_text: 大纲文本

    Returns:
        True 如果所有章节都已研究完毕
    """
    chapters = parse_outline(outline_text)
    if not chapters:
        return True

    current_index = get_next_chapter_index(state)
    return current_index >= len(chapters)


# 导出所有公共函数
__all__ = [
    'parse_outline',
    'get_next_chapter_index',
    'is_all_chapters_researched',
    'get_chapter_task_description',
]
