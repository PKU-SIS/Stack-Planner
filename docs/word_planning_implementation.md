# 报告字数规划功能实现文档

## 1. 功能概述

字数规划功能允许用户在生成报告时指定总字数限制，系统会根据大纲结构智能地为每个节点分配字数配额，确保最终生成的报告符合用户的字数要求。

### 1.1 核心特性

- **智能分配**：使用 LLM 根据节点标题的重要性和内容复杂度进行字数分配
- **层级聚合**：叶子节点分配字数后，自动向上聚合计算父节点字数
- **容错机制**：LLM 分配失败时自动回退到平均分配策略
- **可视化输出**：大纲输出时可选择性地显示各节点的字数配额

### 1.2 工作流程

```
用户指定 total_word_limit
        ↓
execute_outline() 生成大纲
        ↓
检测到 total_word_limit > 0
        ↓
execute_word_planning() 调用 LLM 智能分配字数
        ↓
输出带字数标注的大纲（如 "- 引言 [500字]"）
        ↓
Reporter Agent 根据字数规划生成文档（遵循 ±10% 容差）
```

---

## 2. 数据结构设计

### 2.1 OutlineNode 扩展

**文件**: [src/factstruct/outline_node.py](../src/factstruct/outline_node.py)

在 `OutlineNode` 数据类中新增 `word_limit` 字段：

```python
@dataclass
class OutlineNode:
    id: str
    title: str
    parent: Optional["OutlineNode"] = None
    children: List["OutlineNode"] = field(default_factory=list)
    pull_count: int = 0
    reward_history: List[float] = field(default_factory=list)
    word_limit: int = 0  # 新增：该节点的字数配额
```

### 2.2 to_text_tree() 方法扩展

支持 `include_word_limit` 参数，控制是否在输出中包含字数信息：

```python
def to_text_tree(self, indent: int = 0, include_word_limit: bool = False) -> str:
    """
    将节点树转换为文本格式

    参数:
        indent: 缩进级别
        include_word_limit: 是否包含字数配额信息

    返回:
        文本格式的大纲树
    """
    prefix = "  " * indent
    if include_word_limit and self.word_limit > 0:
        result = f"{prefix}- {self.title} [{self.word_limit}字]\n"
    else:
        result = f"{prefix}- {self.title}\n"

    for child in self.children:
        result += child.to_text_tree(indent + 1, include_word_limit)

    return result
```

**输出示例**：
```
- 报告标题 [5000字]
  - 引言 [500字]
  - 核心分析 [3500字]
    - 分析点1 [1200字]
    - 分析点2 [2300字]
  - 结论 [1000字]
```

### 2.3 State 类型扩展

**文件**: [src/graph/types.py](../src/graph/types.py)

新增 `total_word_limit` 字段用于存储用户指定的总字数限制：

```python
class State(MessagesState):
    # ... 其他字段 ...

    # FactStruct Stage 1 相关字段
    factstruct_outline: Any = None
    factstruct_memory: Any = None
    total_word_limit: int = 0  # 新增：用户指定的总字数限制
```

---

## 3. 核心实现

### 3.1 字数规划 Prompt

**文件**: [src/prompts/word_planner.md](../src/prompts/word_planner.md)

```markdown
你是一位专业的报告字数规划专家。你的任务是根据报告大纲结构和用户指定的总字数限制，
为大纲中的每个节点智能分配字数配额。

## 输入信息
1. 报告大纲（树形结构，包含节点ID和标题）
2. 用户指定的总字数限制

## 分配原则
1. 所有叶子节点的字数之和应等于总字数限制
2. 根据节点标题判断内容的重要性和复杂度，重要/复杂的节点分配更多字数
3. 引言、概述类节点通常占比较小（5-10%）
4. 核心分析、详细论述类节点应占主要篇幅（60-70%）
5. 结论、总结类节点占比适中（10-15%）
6. 每个叶子节点最少分配100字，确保内容完整性

## 输出格式
请以JSON格式输出，包含每个叶子节点的ID和分配的字数：
{
  "allocations": [
    {"node_id": "节点ID", "word_limit": 字数},
    ...
  ],
  "total_allocated": 总分配字数
}

注意：只需要为叶子节点分配字数，非叶子节点的字数由其子节点字数之和决定。
```

### 3.2 execute_word_planning() 方法

**文件**: [src/agents/SubAgentManager.py](../src/agents/SubAgentManager.py)

```python
@timed_step("execute_word_planning")
def execute_word_planning(
    self, outline_root: OutlineNode, total_word_limit: int
) -> OutlineNode:
    """
    执行字数规划，为大纲中的每个叶子节点分配字数配额

    Args:
        outline_root: 大纲根节点
        total_word_limit: 用户指定的总字数限制

    Returns:
        更新了字数配额的大纲根节点
    """
    import json

    logger.info(f"开始字数规划，总字数限制: {total_word_limit}")

    # 1. 构建大纲结构信息供LLM分析
    def build_outline_info(node: OutlineNode, depth: int = 0) -> list:
        nodes_info = []
        nodes_info.append({
            "id": node.id,
            "title": node.title,
            "depth": depth,
            "is_leaf": node.is_leaf()
        })
        for child in node.children:
            nodes_info.extend(build_outline_info(child, depth + 1))
        return nodes_info

    outline_info = build_outline_info(outline_root)
    leaf_nodes = [n for n in outline_info if n["is_leaf"]]

    # 2. 构建LLM请求
    outline_text = outline_root.to_text_tree()
    prompt_content = f"""请为以下报告大纲分配字数。

## 大纲结构
{outline_text}

## 叶子节点列表
{json.dumps(leaf_nodes, ensure_ascii=False, indent=2)}

## 总字数限制
{total_word_limit} 字

请根据每个叶子节点的重要性和内容复杂度，智能分配字数配额。"""

    try:
        # 3. 调用LLM进行字数分配
        messages = apply_prompt_template("word_planner", {"messages": []}) + [
            HumanMessage(content=prompt_content)
        ]
        llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
        response = llm.invoke(messages)
        result = response.content

        # 4. 解析JSON结果
        result = result.replace("```json", "").replace("```", "").strip()
        allocations = json.loads(result)

        # 5. 将字数配额写入叶子节点
        for alloc in allocations.get("allocations", []):
            node_id = alloc.get("node_id")
            word_limit = alloc.get("word_limit", 0)
            node = outline_root.find_node_by_id(node_id)
            if node:
                node.word_limit = word_limit
                logger.debug(f"节点 {node_id} ({node.title}) 分配字数: {word_limit}")

        # 6. 自底向上计算非叶子节点的字数（子节点字数之和）
        def update_parent_word_limits(node: OutlineNode) -> int:
            if node.is_leaf():
                return node.word_limit
            total = sum(update_parent_word_limits(child) for child in node.children)
            node.word_limit = total
            return total

        update_parent_word_limits(outline_root)
        logger.info(f"字数规划完成，根节点总字数: {outline_root.word_limit}")

    except Exception as e:
        logger.error(f"字数规划失败: {str(e)}")
        # Fallback: 平均分配
        leaf_nodes_obj = outline_root.get_leaf_nodes()
        avg_words = total_word_limit // len(leaf_nodes_obj) if leaf_nodes_obj else 0
        for node in leaf_nodes_obj:
            node.word_limit = avg_words
        logger.warning(f"使用平均分配策略，每个叶子节点: {avg_words} 字")

    return outline_root
```

### 3.3 流程集成

**文件**: [src/agents/SubAgentManager.py](../src/agents/SubAgentManager.py) - `execute_outline()` 方法

在大纲生成完成后，检测是否有字数限制，如有则调用字数规划：

```python
# 转换为 Markdown 格式（完整大纲，不限制深度）
outline_response = outline_node_to_markdown(
    outline_root, max_depth=None, include_root=True
)

# 如果用户指定了字数限制，执行字数规划
total_word_limit = state.get("total_word_limit", 0)
if total_word_limit > 0:
    logger.info(f"检测到字数限制 {total_word_limit}，开始字数规划...")
    outline_root = self.execute_word_planning(outline_root, total_word_limit)
    # 更新大纲文本，包含字数信息
    outline_response = outline_root.to_text_tree(include_word_limit=True)
```

---

## 4. Reporter 集成

### 4.1 Prompt 更新

**文件**: [src/prompts/reporter_xxqg.md](../src/prompts/reporter_xxqg.md)

在 Writing Guidelines 中新增字数规划遵循指导：

```markdown
2. Word limit compliance:
   - If the outline contains word limits (e.g., "[500字]"), strictly follow them.
   - Each section should match its allocated word count (±10% tolerance).
   - Prioritize content quality while respecting word limits.
   - If word limits are specified, the total report length must match the sum of all section limits.
```

---

## 5. API 使用

### 5.1 请求参数

在调用工作流时，通过 State 传入 `total_word_limit` 参数：

```python
data = {
    "messages": [{"role": "user", "content": "..."}],
    "thread_id": "__default__",
    "auto_accepted_plan": True,
    # ... 其他参数 ...
    "total_word_limit": 5000,  # 指定总字数限制
}
```

### 5.2 测试脚本

**文件**: [test_hitl_v7.py](../test_hitl_v7.py)

```python
# 测试模式配置
TEST_MODE = "word_planning"  # 启用字数规划测试
TOTAL_WORD_LIMIT = 5000      # 总字数限制

# 请求数据
data = {
    # ... 其他参数 ...
    "total_word_limit": TOTAL_WORD_LIMIT if TEST_MODE == "word_planning" else 0,
}
```

运行测试：
```bash
python test_hitl_v7.py
```

---

## 6. 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| [src/factstruct/outline_node.py:31](../src/factstruct/outline_node.py#L31) | 新增 `word_limit: int = 0` 字段 |
| [src/factstruct/outline_node.py:92-112](../src/factstruct/outline_node.py#L92-L112) | 修改 `to_text_tree()` 支持 `include_word_limit` 参数 |
| [src/prompts/word_planner.md](../src/prompts/word_planner.md) | 新建字数规划 Prompt |
| [src/agents/SubAgentManager.py:643-649](../src/agents/SubAgentManager.py#L643-L649) | 在大纲生成后集成字数规划调用 |
| [src/agents/SubAgentManager.py:735-825](../src/agents/SubAgentManager.py#L735-L825) | 新增 `execute_word_planning()` 方法 |
| [src/graph/types.py:35](../src/graph/types.py#L35) | 新增 `total_word_limit: int = 0` 字段 |
| [src/prompts/reporter_xxqg.md:42-46](../src/prompts/reporter_xxqg.md#L42-L46) | 新增字数规划遵循指导 |
| [test_hitl_v7.py](../test_hitl_v7.py) | 新建测试脚本 |

---

## 7. 设计决策记录

### 7.1 字数分配策略

**选择**: LLM 智能评估的动态分配

**理由**:
1. 报告质量是核心目标，智能分配能确保重要内容获得足够篇幅
2. 大纲节点数量通常有限（10-30个），单次 LLM 调用成本可接受
3. 可以在一次 LLM 调用中完成所有节点的字数规划

**备选方案**:
- 基于层级深度的固定比例分配（实现简单但不够智能）
- 混合策略（复杂度中等）

### 7.2 数据结构扩展

**选择**: 扩展 OutlineNode 数据类，新增 `word_limit` 字段

**理由**:
1. `OutlineNode` 已有 `pull_count`、`reward_history` 等运行时字段，增加 `word_limit` 符合设计模式
2. 字数规划是节点的固有属性，应与节点绑定
3. 便于在 `to_text_tree()` 等方法中直接输出字数信息

**备选方案**:
- 使用独立的字数映射字典（解耦但访问不便）
- 在 Markdown 中内嵌字数标注（格式耦合）

### 7.3 执行时机

**选择**: 新建独立的 `execute_word_planning` 方法

**理由**:
1. 字数规划是独立的功能单元，应与大纲生成解耦
2. 便于后续扩展（如支持用户手动调整字数后重新规划）
3. 符合现有 SubAgentManager 的设计模式（每个功能一个 `execute_xxx` 方法）

**备选方案**:
- 在 `execute_outline` 方法末尾执行（逻辑集中但职责不单一）

---

## 8. 后续优化建议

1. **用户交互优化**: 允许用户在确认大纲时手动调整各节点的字数分配
2. **字数验证**: 在报告生成后验证实际字数是否符合规划，超出容差时提示用户
3. **分配策略可配置**: 支持用户选择不同的分配策略（智能/平均/自定义比例）
4. **历史学习**: 基于历史报告的实际字数分布，优化 LLM 的分配建议
