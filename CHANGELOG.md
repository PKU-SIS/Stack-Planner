# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `docs/state.md`: 文档化 State 字段（类型、含义、示例），来源于 `src/graph/types.py` 与全局使用点。
 - `test_hitl_v2.py`: 新增双中断联调测试脚本（问卷 → 大纲 → 成稿）。

### Changed
- `test_hitl_v2.py`: 当第二次进入 `perception` 节点时，抑制随后到下一次中断前的输出，避免在客户端显示内部提示拼接内容。
