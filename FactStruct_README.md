# FactStruct 使用指南

## 🚀 快速开始

基本上沿用Stack-Planner的环境配置即可，只需要多安装几个包。

### 1. 安装依赖

FactStruct Stage 1 依赖以下 Python 包（已在原始的 `requirements.txt` 中追加这部分）：

```bash
# 核心依赖
sentence-transformers  # 用于文档嵌入生成
faiss-cpu             # 用于向量索引和相似度搜索
scikit-learn          # 用于向量计算和相似度度量
loguru                # 用于日志记录
```

如果使用 `uv` 管理依赖，运行：

```bash
uv sync
```

如果使用 `pip`，运行：

```bash
pip install -r requirements.txt
```


### 2. 验证测试

运行诊断脚本检查FactStruct部分是否可正常运行：

```bash
python examples/factstruct_diagnose.py
```

该脚本会检查：
- LLM 配置是否正确
- API key 是否设置
- LLM 连接是否正常


## 实现思路

目前只实现了Stage 1。


  

### **Stage 1: 动态大纲生成与优化 (Batch-MAB 框架)**

  

#### **1. 动机：为什么需要动态大纲？**

  

本模块旨在解决开放式深度研究（OEDR）中的“冷启动”和“结构构建”问题。传统的 RAG（检索增强生成）系统或简单的报告生成器，往往采用“一次性”大纲：即先根据用户查询生成一个静态大纲，然后才去检索文档。

  

这种方法的根本缺陷在于**它错误地假设我们在研究开始前就已经知道了答案的轮廓**。在真实的科研（Research）过程中，大纲并不是一成不变的，它是一个随着新信息的发现而不断演进、修正、生长的“活”结构。

  

因此，我们的核心动机是**必须将“大纲规划”与“信息检索”深度耦合、交错执行 (interleave)**。系统必须能够根据检索到的新证据，动态地、智能地调整其后续的研究方向（即大纲）。

  

#### **2. 现有工作的不足**

  

现有的先进动态大纲系统（如 WebWeaver）已经取得了巨大进展。它们不再是“策略盲目”的，而是采用了先进的“迭代规划”（iterative planning）或“ReAct 风格的代理循环”（ReAct-style agentic loop）。

  

在这些系统中，一个“规划器”代理会根据新检索到的证据，迭代地“优化大纲”（`outline_optimization`），这个过程是自适应的。

  

然而，这些 SOTA（State-of-the-Art）方案仍然存在一个根本性的局限：

  

1. **成本不可控 (Unpredictable Cost):** 现有的“迭代规划”依赖 LLM 代理的“判断”在一个循环中运行，直到代理“认为”大纲已经足够好为止。这个终止条件是模糊的，导致其 API 总成本（包括检索和 LLM 优化）仍然是**不可预测的**，并且与其最终大纲的复杂程度（$N_{\text{nodes}}$）高度相关。

2. **缺乏显式的预算分配 (Lack of Explicit Budgeting):** 现有的“迭代规划”是一个“黑盒”（依赖 LLM 的“直觉”）。它缺乏一个透明的、量化的经济模型来决定_为什么_要在某个分支上投入更多成本，或者_何时_应该停止探索。

  

#### **3. 我们的方案: Batch-IF-MAB 框架**

  

为了解决上述问题，我们设计了一个 **“批量-信息觅食多臂老虎机” (Batch-IF-MAB)** 框架。该框架旨在完美地平衡如下两点：

  

1. **API 成本 (Cost):** 必须严格控制 LLM API 的总调用次数。

2. **大纲质量 (Quality):** 必须是自适应的（Adaptive），能智能地将预算花在“信息增益”最高的分支上。

  

**Batch-IF-MAB 如何解决这个铁三角问题？**

  

- **(解决质量问题)** 我们使用 **MAB (UCB1)** 策略，将大纲的叶子节点视为“手臂”。系统会计算每个节点的 UCB 分数（结合“平均奖励”和“探索次数”），从而**智能地识别**出最值得探索的分支。

- **(解决成本问题)** 我们引入**批量处理 (Batching)**。我们不一次只选 1 个最优节点（这会导致 $O(T_{\text{max}})$ 次 LLM 调用），而是一次**选择 Top-K 个**（例如 K=5）最优节点。

  

**成本优势 (示例):**

  

- 总预算 $T_{max} = 20$ (即总共检索 20 次)

- 批量大小 $K = 5$

- 总轮数 = $T_{max} / K = 4$ 轮

- **总 LLM 调用 = 1 (初始大纲) + 4 轮** $\times$ **(1 批量查询 + 1 批量修订) = 9 次**

- 这（9 次）远低于标准 MAB（41 次）或“逐节点扩展”（$O(N_{\text{nodes}})$）的成本，同时保持了 MAB 的智能性。

  

#### **4. 核心算法流程**

  

Batch-IF-MAB 的执行流程分为**初始化**和**批量迭代循环**两个阶段。

  

**A：初始化 (LLM Call #1)**

  

1. **初始检索:** 接收用户的 `Initial_Query`，检索第一批文档 $D_0$。

2. **生成初始大纲:** 调用 `LLM.generate_initial_outline(Query, D_0)`，生成一个基础的大纲树 $O_{\text{root}}$（例如只包含 L1 层节点）。

3. **初始化:** 设置总迭代次数 `t = 0`。

  

**B：批量 MAB 迭代循环**

  

系统将执行固定的 `num_rounds = T_max / K` 轮（例如 4 轮）。在每一轮中，按顺序执行以下步骤：

  

1. **Step 1: 选择 Top-K“手臂” (UCB1 策略)**

- 系统遍历当前大纲树 $O_{\text{root}}$ 上的所有**叶子节点**（即“手臂”）。

- 为每个叶子节点 $n_i$ 计算其 **UCB 分数**，以平衡“利用”与“探索”：

$$ \text{UCB\_Score}(n_i) = \underbrace{\bar{x}_i}_{\text{Exploitation}} + \underbrace{\sqrt{\frac{2 \ln t}{N_i(t)}}}_{\text{Exploration}}$$

- $\bar{x}_i$ 是该节点已获得的**平均奖励** (`node.avg_reward()`)。

- $N_i(t)$ 是该节点已被检索的**次数** (`node.pull_count`)。

- 系统选择分数最高的 **Top-K** 个节点（例如 K=5）作为本轮要处理的目标 `SelectedNodes`。

2. **Step 2: 批量生成查询 (LLM Call #Round*2)**

- 系统将这 K 个 `SelectedNodes` 传递给 `LLM.batch_generate_queries`。

- 这是一个**单次 LLM 调用**，它接收 K 个节点信息，并一次性返回 K 个对应的搜索查询字符串 `Queries`。

3. **Step 3: 并行执行检索**

- 系统调用 `SearchEngine.parallel_search(Queries)`，并行执行 K 次搜索，返回 K 组新文档 `D_new_list`。

4. **Step 4: 批量计算奖励与更新状态 (本地计算)**

- 系统在本地循环 K 次（不涉及 API 调用）：

- 对于第 $i$ 个节点 $n_i$ 及其新文档 $D_i$：

- $t += 1$（全局迭代次数加 1）。

- 计算“信息增益”**奖励** $r_t$。这是 MAB 策略的核心，计算公式为：

$$ r_t = w_{\text{rel}} \cdot \text{Relevance} + w_{\text{nov}} \cdot \text{Novelty}$$

- 其中，`Relevance` 和 `Novelty` 的计算方式如下：

- **1. Relevance (相关性):**

- **定义：** 衡量本轮 new 检索到的文档 $D_i$ 与触发此次检索的查询节点 $n_i$ 的“主题一致性”。

- **计算：**

1. 获取查询节点 $n_i$ 的嵌入向量：$v_{\text{query}} = \text{Embedder.encode}(n_i.\text{title})$。

2. 获取 $D_i$ 中所有 $k$ 篇文档的嵌入向量：$V_{\text{docs}} = \{\text{Embedder.encode}(d_j) \text{ for } d_j \text{ in } D_i\}$。

3. 计算 $v_{\text{query}}$ 与 $V_{\text{docs}}$ 中每个向量的余弦相似度。

4. **Relevance = 这** $k$ **个相似度分数的平均值。**

- **2. Novelty (新颖性):**

- **定义：** 衡量 $D_i$ 提供了多少“新信息”，即这些信息在系统_已有的_全局文档池 $D_{all}$ 中是不存在的。

- **计算：**

1. 获取 $D_i$ 中 $k$ 篇文档的嵌入向量 $V_{\text{docs}}$。

2. 获取 $D_{all}$ 中所有 $m$ 篇（可能成千上万）文档的嵌入向量 $V_{\text{all}}$。

3. 对于 $D_i$ 中的**每一篇**文档 $d_j$，计算它与 $V_{\text{all}}$ 中所有向量的余弦相似度，并找到**最大值** $\text{max\_sim}(d_j)$。这个值代表 $d_j$ 与现有知识库的“最大重叠度”。

4. 该文档 $d_j$ 的“新颖度分”为 $1 - \text{max\_sim}(d_j)$。

5. **Novelty = 这** $k$ **个“新颖度分”的平均值。**

- **性能考量：**

- 步骤 2 和 3 的计算成本非常高 ($O(k \times m)$)。

- 在实际工程实现中，**必须**使用向量索引（如 FAISS 或 Annoy）来存储 $V_{\text{all}}$。

- 这样，步骤 3 就从“暴力计算” $O(m)$ 降维为“近似近邻搜索” $O(\log m)$，从而使 `Novelty` 的计算变得可行。

- 在计算出 $r_t$ 后，更新该节点 $n_i$ 的 MAB 状态：`n_i.reward_history.append(r_t)` 且 `n_i.pull_count += 1`。

5. **Step 5: 批量修订大纲 (LLM Call #Round*2 + 1)**

- 系统调用 `LLM.batch_refine_outline`，这是本轮的**第二次（也是最后一次）LLM 调用**。

- 它将当前大纲树 $O_{\text{root}}$ 和 K 组 `(node, D_new)` 数据对作为输入。

- LLM 被 Prompt 约束，在一次响应中对 K 个节点_分别_进行_独立的_局部优化，最后输出一个**完整的新大纲树** $O_{\text{root}}$。

6. **Step 6: MAB 状态继承 (关键机制)**

- `batch_refine_outline` 步骤会返回哪些节点被扩展了的信息（例如 $n$ 扩展出了 $n_a, n_b$）。

- 此时，系统**必须**执行“状态继承”：将父节点 $n$ 的 MAB 状态（`pull_count` 和 `reward_history` 副本）**复制**给所有新生成的子节点 ($n_a, n_b, ...$)。

- **(继承的动机)** 这是确保算法智能性的核心。如果不这样做，新节点 $n_a, n_b$ 的“利用项” $\bar{x}_i$ 将为 0，导致 MAB 策略在下一轮退化为“纯探索”。状态继承确保了父分支的“经验”得以保留和传递。

  

在所有轮次（例如 4 轮）完成后，Stage 1 结束，系统输出最终优化的大纲树 $O_{\text{Hroot}}$。