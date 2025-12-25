# State 字段说明

本文档汇总 `State`（工作流运行时状态）中可能出现的字段、类型、含义与示例，便于开发与调试。

数据来源：
- 类型定义：`src/graph/types.py` 中的 `class State(MessagesState)`
- 框架基类：`langgraph.graph.MessagesState`（提供 `messages` 等对话相关字段）
- 实际用例：代码中对 `state.get(...)` 的读取与写入

---

## 基础：MessagesState 提供的字段

- **messages**: list[BaseMessage]
  - **含义**：对话消息列表，包含用户消息、Agent回复、工具消息。
  - **示例**：
    ```json
    [
      {"type": "human", "content": "帮我研究比特币价格波动"},
      {"type": "ai", "content": "好的，我将进行研究。"}
    ]
    ```

---

## 运行时字段（src/graph/types.py）

- **locale**: str，默认 "zh-CN"
  - **含义**：当前任务的语言/地区偏好。
  - **示例**：`"en-US"`

- **observations**: list[str]
  - **含义**：执行过程中的观察/中间结论，用于上下文累积与报告生成。
  - **示例**：`["已收集到5篇相关文章", "模型A更适用本任务"]`

- **data_collections**: list[Any]
  - **含义**：原始数据收集结果，如网页抓取正文、检索到的文档片段等。
  - **示例**：`["<html>...", {"doc_id": "abc", "chunk": "..."}]`

- **resources**: list[Resource]
  - **含义**：可用资源清单（如检索器、文件、向量库等），用于工具链。
  - **示例**：`[{"type": "rag", "name": "knowledge-base-1"}]`

- **plan_iterations**: int，默认 0
  - **含义**：规划重试/迭代计数，用于控制多轮规划行为。
  - **示例**：`1`

- **current_plan**: Plan | str | None
  - **含义**：当前的任务规划（结构化 Plan 或字符串表示）。
  - **示例**：`{"steps": [{"title": "检索文献"}, {"title": "撰写报告"}]}` 或 `"先检索，后总结"`

- **user_query**: str
  - **含义**：用户原始问题/需求。
  - **示例**：`"请分析比特币过去一周价格波动原因"`

- **final_report**: str
  - **含义**：最终报告文本，由报告Agent生成。
  - **示例**：`"综述如下：..."`

- **replan_result**: str
  - **含义**：任务拆解/重规划结果的原始字符串（常为 JSON 字符串）。
  - **示例**：`"{\"DAG\": [[\"检索\", \"总结\"]]}"`

- **auto_accepted_plan**: bool，默认 False
  - **含义**：是否自动接受规划（跳过用户确认）。
  - **示例**：`true`

- **enable_background_investigation**: bool，默认 True
  - **含义**：是否开启后台调研流程（边执行边检索）。
  - **示例**：`true`

- **background_investigation_results**: str | None
  - **含义**：后台调研的汇总结果字符串（若开启且已产出）。
  - **示例**：`"已在后台补充收集到3条信息：..."`

- **user_dst**: str | None
  - **含义**：感知阶段（DST）归纳的用户约束/偏好总结。
  - **示例**：`"关注中文资料、偏向学术来源、期望有可视化图表"`

- **delegation_context**: dict | None
  - **含义**：委派上下文（子Agent的执行指令与参数），常含 `task_description`。
  - **示例**：`{"task_description": "收集近一周主流媒体报道"}`

- **current_node**: str | None
  - **含义**：当前工作流节点名称（用于图执行与跳转）。
  - **示例**：`"researcher"`、`"reporter"`

- **memory_stack**: str | None
  - **含义**：记忆栈的序列化表示（在进入提示时可反序列化为对象）。
  - **示例**：`"{\"entries\":[{\"action\":\"delegate\",...}]}"`

---

## 其他常见约定字段（来自使用场景）

- **original_query**: str（通常从 `user_query` 派生，用于报告或记录）
  - **示例**：`"请分析比特币过去一周价格波动原因"`

- （仅旧备份代码中出现）`task_completed`: bool
  - 说明：备份文件 `src/graph/delete_bak/sp_nodes_case.py` 中的历史字段，主线逻辑不依赖。

---

## 字段关系与典型流转

- `user_query` → 规划/感知（可能产出 `user_dst` 与 `current_plan`）
- 执行中产生 `observations` 与 `data_collections`，用于后续 `reporter` 汇总到 `final_report`
- `delegation_context` 为各子Agent输入上下文；`current_node` 用于图节点路由
- `resources` 参与工具配置（检索、RAG等）
- `memory_stack` 记录跨步骤记忆，可能以字符串在 Prompt 中传递

---

## 最小可用示例（片段）

```json
{
  "messages": [
    {"type": "human", "content": "请调研OpenAI Sora的典型应用"}
  ],
  "locale": "zh-CN",
  "user_query": "请调研OpenAI Sora的典型应用",
  "auto_accepted_plan": true,
  "delegation_context": {"task_description": "检索近6个月行业案例"},
  "resources": [{"type": "rag", "name": "industry-kb"}],
  "observations": [],
  "data_collections": []
}
```

---

## 兼容性与边界

- 未设置的字段应以 `get(key, default)` 形式读取，并提供合理默认值。
- `current_plan`、`final_report`、`background_investigation_results`、`user_dst` 等可能为 `None`。
- `memory_stack` 在进入提示模板前可能需要 JSON 反序列化。

---

## 来源引用

- `src/graph/types.py` — State 字段权威定义
- `src/agents/*`、`src/graph/nodes.py` — 字段的实际读写示例
- `src/prompts/template.py` — `memory_stack` 的反序列化逻辑
