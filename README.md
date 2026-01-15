# 🚀 StackPlanner

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 基于DeerFlow重构的多Agent智能规划系统，专注于高效的任务分解与执行

## 📑 目录

- [🌟 项目概述](#项目概述)
- [🏗️ 架构设计](#架构设计)
- [🚀 快速开始](#快速开始)
- [🔧 核心组件](#核心组件)
- [📝 配置说明](#配置说明)
- [💻 开发指南](#开发指南)
- [🐳 Docker部署](#docker部署)
- [📚 示例](#示例)
- [🔍 监控与调试](#监控与调试)
- [📜 许可证](#许可证)

## 项目概述

StackPlanner是一个基于多Agent架构的智能规划系统，由原DeerFlow项目重构而来。系统采用了全新的中枢Agent与子Agent设计，引入了MemoryStack记忆管理机制，并构建了名为"sp"的新算法，实现了更高效的任务分解与执行。

### 主要特性

- 🤖 **全新的中枢Agent设计**：智能决策与任务委派系统
- 📚 **MemoryStack记忆管理**：高效的执行历史存储与检索
- 🔄 **子Agent集成机制**：灵活的专项任务执行能力
- 🎯 **sp算法**：优化的任务规划与执行流程
- 🔧 **丰富的工具集成**：搜索、代码执行、信息检索等多种工具
- 🌐 **多格式输出**：支持研究报告、代码分析等多种输出形式

## 架构设计

StackPlanner采用分层架构设计，由感知层、决策层、执行层和输出层组成。系统基于LangGraph构建状态图，实现了灵活的工作流管理。

### 核心架构组件

1. **感知层（Perception）**
   - 处理用户输入与环境信息
   - 支持human-in-the-loop交互
   - 提供search before plan能力

2. **决策层（Central Agent）**
   - 中枢决策系统，负责任务分析与规划
   - 基于MemoryStack进行智能决策
   - 子Agent任务委派与管理

3. **执行层（Sub Agents）**
   - Researcher：信息检索与研究
   - Coder：代码生成与执行
   - Reporter：结果整理与报告生成
   - Planner：复杂任务分解与规划

4. **输出层（Output）**
   - 综合报告生成
   - 结果可视化
   - 多格式输出支持

### 状态图设计

系统提供多种图格式实现，适应不同场景需求：

- **base**：基础工作流
- **sp**：基于新算法的优化工作流
- **sp_xxqg**：增强版搜索任务处理流程

## 快速开始

### 环境要求

- **Python**：3.12+
- **Node.js**：22+（仅Web UI需要）

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd StackPlanner

# 安装Python依赖
uv sync

# 配置环境变量
cp .env.example .env

# 配置LLM模型与API密钥
cp conf.yaml.example conf.yaml

# 可选：安装Web UI依赖
cd web
pnpm install
```

### 启动方式

#### 1. 控制台模式

```bash
# 基本模式
uv run main.py "你的研究问题"

# 交互式模式
uv run main.py --interactive

# 查看所有选项
uv run main.py --help
```

#### 2. Web服务模式

```bash
# 启动后端API服务
uv run server.py

# 启动Web UI（在web目录下）
pnpm dev
```

#### 3. LangGraph Studio调试

```bash
# 启动LangGraph Studio
langgraph dev
```

## 核心组件

### 中枢Agent（Central Agent）

中枢Agent是系统的核心决策组件，负责分析任务、制定计划、委派子Agent执行任务，并最终生成结果。

#### 核心功能

- **智能决策**：基于当前状态分析生成决策
- **任务委派**：根据任务类型选择合适的子Agent
- **记忆管理**：通过MemoryStack追踪执行历史
- **结果整合**：汇总子Agent执行结果并生成最终报告

#### 决策流程

1. 分析用户输入与当前状态
2. 基于MemoryStack生成决策
3. 执行决策动作（思考、反思、委派、总结、完成）
4. 更新系统状态与记忆栈

### 子Agent系统

子Agent负责执行专项任务，由中枢Agent根据任务类型进行委派。

#### 内置子Agent

| Agent类型 | 职责 | 适用场景 |
|----------|------|----------|
| Researcher | 信息收集与研究 | 事实查询、资料收集、市场调研 |
| Coder | 代码生成与执行 | 数学问题、编程任务、数据分析 |
| Reporter | 结果整理与报告生成 | 综合报告、总结分析、文档撰写 |
| Planner | 任务分解与规划 | 复杂任务、多步骤流程、搜索策略 |

#### 子Agent注册机制

系统采用注册表模式管理子Agent，支持动态扩展：

```python
# 子Agent注册示例
sub_agents_sp = [
    {
        "name": "researcher",
        "description": "Information collection and research",
        "node": researcher_node,
    },
    # 其他子Agent...
]
```

### MemoryStack记忆管理

MemoryStack是系统的记忆管理组件，负责存储和管理执行历史，为决策提供上下文信息。

#### 核心功能

- **执行历史存储**：记录系统执行的每一步操作
- **记忆检索**：快速获取最近或相关的执行历史
- **记忆摘要**：生成记忆栈摘要，提供决策上下文
- **记忆管理**：支持记忆的推送、弹出、更新等操作

#### 使用方式

```python
# MemoryStack使用示例
memory_stack = MemoryStack()
memory_entry = MemoryStackEntry(
    timestamp=datetime.now().isoformat(),
    action="delegate",
    agent_type="researcher",
    content="委派任务: 收集AI趋势信息",
)
memory_stack.push(memory_entry)

# 获取记忆摘要
memory_summary = memory_stack.get_summary(include_full_history=True)
```

### 工具集成

系统集成了多种工具，为Agent提供执行能力：

| 工具类型 | 功能 | 适用场景 |
|----------|------|----------|
| 搜索工具 | Web搜索、学术搜索 | 信息收集、资料检索 |
| 代码执行 | Python代码执行 | 数据分析、算法验证 |
| 信息检索 | 文档解析、知识库查询 | 结构化信息获取 |
| TTS | 文本转语音 | 音频内容生成 |

## 配置说明

### 环境变量配置（.env）

```bash
# 搜索引擎配置
SEARCH_API=tavily
TAVILY_API_KEY=your_api_key

# LLM配置
OPENAI_API_KEY=your_api_key

# 其他服务配置
RAG_PROVIDER=ragflow
RAGFLOW_API_URL=http://localhost:9388
RAGFLOW_API_KEY=your_api_key
```

### 模型配置（conf.yaml）

```yaml
# LLM模型配置
llms:
  default:
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
  planner:
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
  researcher:
    model: gpt-3.5-turbo
    api_key: ${OPENAI_API_KEY}

# 搜索配置
search:
  provider: tavily
  max_results: 5
  timeout: 30
```

### 图格式配置

系统支持多种图格式，可在启动时指定：

- **base**：基础工作流
- **sp**：新算法优化工作流
- **sp_xxqg**：增强版搜索任务处理

```bash
# 使用sp算法启动
uv run main.py --graph_format sp "你的研究问题"
```

## 开发指南

### 代码结构

```
StackPlanner/
├── src/
│   ├── agents/           # Agent实现
│   │   ├── CentralAgent.py      # 中枢Agent
│   │   ├── SubAgentManager.py   # 子Agent管理
│   │   └── sub_agent_registry.py # 子Agent注册表
│   ├── graph/            # 状态图定义
│   │   ├── builder.py    # 图构建器
│   │   ├── nodes.py      # 基础节点
│   │   └── sp_nodes.py   # sp算法节点
│   ├── memory/           # 记忆管理
│   │   ├── memory_stack.py       # 记忆栈
│   │   └── memory_stack_entry.py # 记忆条目
│   ├── tools/            # 工具集成
│   ├── prompts/          # 提示词模板
│   └── config/           # 配置管理
├── web/                  # Web UI
├── main.py               # 主入口
└── server.py             # API服务
```

### 自定义开发教程

#### 1. 使用LangGraph的Graph Builder

LangGraph的Graph Builder是构建工作流的核心组件，您可以使用它创建自定义的状态图：

```python
from langgraph.graph import StateGraph, START, END
from src.graph.types import State

def build_custom_graph():
    """构建自定义状态图"""
    # 创建状态图构建器
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("custom_node", custom_node_function)
    builder.add_node("another_node", another_node_function)
    
    # 定义边
    builder.add_edge(START, "custom_node")
    builder.add_edge("custom_node", "another_node")
    builder.add_edge("another_node", END)
    
    # 编译并返回图
    return builder.compile()

# 节点函数示例
def custom_node_function(state: State):
    """自定义节点函数"""
    # 处理逻辑
    new_state = state.copy()
    new_state.observations.append("Custom node executed")
    return new_state
```

#### 2. 使用State管理系统状态

State是系统状态的核心数据结构，您可以使用它存储和传递信息：

```python
from src.graph.types import State

def process_state(state: State) -> State:
    """处理状态示例"""
    # 读取状态
    user_query = state.user_query
    observations = state.observations
    
    # 修改状态
    new_state = state.copy()
    new_state.observations.append(f"Processed query: {user_query}")
    new_state.current_node = "processing"
    
    # 添加自定义数据
    if not hasattr(new_state, "custom_data"):
        new_state.custom_data = {}
    new_state.custom_data["processed_at"] = "2024-01-01"
    
    return new_state
```

#### 3. 定义新的Agent

要定义新的Agent，您需要：

1. **创建Agent节点函数**：

```python
from src.graph.types import State
from langgraph.types import Command

def custom_agent_node(state: State) -> Command:
    """自定义Agent节点"""
    # 获取委派上下文
    task_description = state.delegation_context.get("task_description", "")
    
    # 执行Agent逻辑
    result = f"Custom agent processed: {task_description}"
    
    # 更新状态
    new_state = state.copy()
    new_state.observations.append(result)
    
    # 返回命令
    return Command(
        update=new_state,
        goto="central_agent"  # 返回中枢Agent
    )
```

2. **注册Agent**：

```python
# 在sub_agent_registry.py中添加
from src.graph.sp_nodes import custom_agent_node

sub_agents_sp.append({
    "name": "custom_agent",
    "description": "Custom agent for special tasks",
    "node": custom_agent_node,
})
```

3. **创建提示词模板**（可选）：

```python
# 在prompts目录下创建custom_agent.md

You are a custom agent specialized in special tasks.

Task: {task_description}

Context: {memory_context}

Please process this task and return the result.
```

#### 4. 定义新的工具

系统支持通过装饰器注册新工具：

```python
from src.tools.decorators import tool

@tool
def custom_search_tool(query: str, max_results: int = 3) -> str:
    """
    自定义搜索工具
    
    Args:
        query: 搜索查询
        max_results: 最大结果数
        
    Returns:
        搜索结果
    """
    # 实现搜索逻辑
    results = [f"Result {i}: {query} - {i}" for i in range(max_results)]
    return "\n".join(results)
```

**工具使用示例**：

```python
from src.tools import custom_search_tool

# 在Agent中使用工具
results = custom_search_tool("人工智能趋势", max_results=5)
print(results)
```

### 自定义子Agent

要添加自定义子Agent，需在`sub_agent_registry.py`中注册：

```python
# 添加自定义子Agent
sub_agents_sp.append({
    "name": "custom_agent",
    "description": "Custom agent description",
    "node": custom_agent_node,
})
```

### 扩展工具

系统支持通过装饰器注册新工具：

```python
from src.tools.decorators import tool

@tool
def custom_tool(query: str) -> str:
    """自定义工具描述"""
    # 工具实现
    return result
```

## Docker部署

### 构建镜像

```bash
docker build -t stackplanner .
```

### 运行容器

```bash
docker run -d -p 8000:8000 --env-file .env --name stackplanner-app stackplanner
```

### Docker Compose

```bash
# 构建并启动
 docker compose up --build

# 停止服务
 docker compose down
```

## 示例

### 基本用法

```bash
# 执行简单的研究任务
uv run main.py "人工智能在医疗领域的应用趋势"

# 使用sp算法
uv run main.py --graph_format sp "2024年编程语言趋势分析"

# 交互式模式
uv run main.py --interactive
```

### 高级用法

```bash
# 自定义最大规划迭代次数
uv run main.py --max_plan_iterations 3 "量子计算对密码学的影响"

# 自定义最大步骤数
uv run main.py --max_step_num 5 "如何构建一个智能推荐系统"

# 启用调试模式
uv run main.py --debug "机器学习模型部署最佳实践"
```

## 监控与调试

### LangGraph Studio

系统集成了LangGraph Studio，支持工作流可视化与调试：

```bash
# 启动LangGraph Studio
langgraph dev --allow-blocking
```

### 执行报告

系统会自动生成执行报告，保存到`./reports`目录：

```
reports/
execution_report_20240101_120000.json  # 执行报告
```

### 日志管理

系统使用结构化日志，可通过配置调整日志级别：

```yaml
# conf.yaml
logging:
  level: info  # 可选: debug, info, warning, error
  format: json
```

## 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 贡献

欢迎提交Issue和Pull Request，共同完善StackPlanner系统。

---

**StackPlanner** - 智能规划，高效执行 🚀