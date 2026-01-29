#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_to_json.py
将无时间戳前缀的日志解析为 JSON（列表）。
修正点：
- 采用显式状态机：只有在不处于实体内时才识别实体开始，避免把属性误判为实体
- 支持属性块单行形式：=== ... ===END
- 无 agent 时自动填充 "central_agent"
- 属性为顶层键，字段名：action / agent
- 实体以 "ACTION END" 收束（action 名可不严格校验，尽量容错）
"""

import json
import re
import sys
from pathlib import Path
from typing import List


def parse_log(lines: List[str]):
    entities = []

    # 状态
    OUTSIDE = 0
    IN_ENTITY = 1
    IN_ATTR = 2
    state = OUTSIDE

    current = None  # 当前实体 dict
    current_attr = None  # 当前属性名
    buf = []  # 属性内容缓冲

    # 正则
    re_entity_start = re.compile(
        r"^([A-Z]+)(?:\s+(\S+))?$"
    )  # e.g. DECISION / DELEGATE reporter
    re_entity_end = re.compile(r"^([A-Z]+)\s+END$")  # e.g. DECISION END
    re_attr_start = re.compile(r"^([A-Z_]+)$")  # e.g. INPUT / OUTPUT / CONTEXT

    def flush_attr():
        """把当前属性缓冲写入实体并清空"""
        nonlocal current, current_attr, buf
        if current is not None and current_attr is not None:
            content = "\n".join(buf).strip()
            # 去除首尾 === 与 ===END（容错：若使用分段写入，这里多一道保险）
            content = content.strip()
            current[current_attr] = content
        buf = []
        current_attr = None

    def flush_entity():
        """结束当前实体写入列表"""
        nonlocal current
        if current is not None:
            entities.append(current)
            current = None

    for raw in lines:
        line = raw.rstrip("\n")

        # 统一处理空行
        if not line.strip():
            if state == IN_ATTR:
                buf.append("")  # 属性内容允许空行
            continue

        if state == OUTSIDE:
            # 只在 OUTSIDE 识别实体开始
            m = re_entity_start.match(line)
            if m:
                act, ag = m.groups()
                current = {"action": act, "agent": ag or "central_agent"}
                state = IN_ENTITY
                continue
            else:
                # 非法行（既不是实体开始，也不是空行），忽略或记录？
                # 这里选择忽略，必要时可改为 raise。
                continue

        if state == IN_ENTITY:
            # 先看是否实体结束
            if re_entity_end.match(line):
                # 如果正在一个属性块里，先收尾
                if current_attr is not None:
                    flush_attr()
                flush_entity()
                state = OUTSIDE
                continue

            # 属性开始？
            m = re_attr_start.match(line)
            if m:
                # 收尾上一个属性（若存在）
                if current_attr is not None:
                    flush_attr()

                current_attr = m.group(1)
                state = IN_ATTR
                continue

            # 既不是 END 也不是属性名：此时是非法结构（为了稳健，忽略）
            continue

        if state == IN_ATTR:
            s = line.strip()

            # 单行属性块：=== ... ===END
            if s.startswith("===") and s.endswith("===END"):
                content_inline = s[3:-6].strip()
                buf = [content_inline]
                flush_attr()
                state = IN_ENTITY
                continue

            # 属性块开始：=== ...
            if s.startswith("===") and not s.endswith("===END"):
                # 去掉前导 ===
                buf.append(s[3:].lstrip())
                continue

            # 属性块结束：... ===END
            if s.endswith("===END"):
                buf.append(s[:-6].rstrip())
                flush_attr()
                state = IN_ENTITY
                continue

            # 普通属性内容行
            buf.append(line)
            continue

    # 文件结束收尾
    if state == IN_ATTR:
        flush_attr()
        state = IN_ENTITY
    if state == IN_ENTITY:
        flush_entity()

    return entities


def main():
    import argparse

    parser = argparse.ArgumentParser(description="解析日志为 JSON 列表")
    parser.add_argument("logfile", help="输入日志文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径（可选）")
    parser.add_argument("--compact", action="store_true", help="紧凑输出（无缩进）")
    args = parser.parse_args()

    p = Path(args.logfile)
    if not p.exists():
        print(f"❌ 文件不存在: {p}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = parse_log(lines)
    if args.compact:
        result = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        result = json.dumps(data, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"✅ 已写入: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
