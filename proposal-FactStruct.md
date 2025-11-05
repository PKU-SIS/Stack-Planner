
## 1. 背景

### **研究任务**

本研究聚焦于 **开放式深度研究（Open-ended Deep Research, OEDR）** 任务，
即如何利用大型语言模型（Large Language Models, LLMs）在复杂、跨领域主题下自动生成
**结构化、内容充分且事实可验证的研究报告**。
这种任务不同于传统的摘要或问答生成，它要求系统能够：

1. 从海量异质文献中检索与筛选证据；
2. 构建合理的大纲结构；
3. 生成具有逻辑连贯性和事实支撑性的文本。

---

### **现有工作的不足**

尽管已有系统（如 WebWeaver、WriteHERE、DeepResearcher）在“动态大纲规划（dynamic outline planning）”与“递归写作（recursive writing）”上取得进展，但仍存在两个根本性局限：

1. **引用不等于支撑（Citation ≠ Entailment）**
   现有研究大多仅验证文本中“是否有引用”，
   却未验证“引用的内容是否语义上支撑文本陈述”。
   因此，生成报告中仍存在“被引用却不被支撑”的虚假事实（factual hallucination）。

2. **支撑文档选择粗糙**
   大多数检索增强系统（RAG, Retrieval-Augmented Generation）
   仅依据语义相似度（如 cosine similarity）选文，
   忽略了**来源质量、信息密度与时效性（freshness）**。
   结果导致模型可能引用过时或低可信度文献。


---

### **科学问题**

本研究拟解决的核心科学问题是：

> **如何在开放式科研报告生成中，实现生成内容与支撑文献之间的语义一致性与事实可验证性？**

---

### **研究意义与重要性**

在 LLM 被广泛用于科研写作、文献综述与技术报告的时代，
**事实一致性（factual consistency）** 已成为评价生成系统可靠性的关键指标。
AI 生成文本若缺乏事实支撑，不仅削弱其实用价值，更可能带来严重的知识污染。

本研究的重要性在于：

* 它使生成型系统不只是“能写”，而是“能写真”；
* 它推动生成式 AI 从“语言生成”向“知识验证（knowledge-grounded reasoning）”演化；
* 它为构建“可信赖科研智能体（Trustworthy Research Agent）”奠定理论与工程基础。

---

## 2. 解决方法与框架

### **核心洞见**

FactStruct 的核心洞见是：

> 科研报告生成不是一个单纯的文本生成问题，而是一个“**结构–事实耦合（structure–evidence coupling）**”过程。
> 生成的可信度不仅取决于语言模型的表达能力，更取决于它与外部事实源之间的逻辑一致性。

基于该洞见，我们将科学问题进一步拆分为两个子问题：
1. **如何在篇幅受限的条件下，从大量候选文献中选取最相关、最可靠、最新的支撑文档？**
2. **如何通过语义推理模型判断生成文本是否被引用文献真正支撑，并在不一致时进行自动修订？**

提出FactStruct方法框架以解决上述子问题。

---

### **方法框架概述**

FactStruct由四个核心阶段组成：

```
[输入主题]
   ↓
[Stage 3: 大纲规划]
   ↓
[Stage 4: 文档检索与多因素排序]
   ↓
[Stage 5: 层次化写作]
   ↓
[Stage 6: 一致性检测与自动修订]
   ↓
[输出：结构化、事实支撑的科研报告]
```


## 2. 解决方法与框架

#### **Stage 1: 动态大纲生成与优化 (Batch-MAB 框架)**

本模块解决“如何构建合理大纲结构”的问题。原有的 IF-MAB 方案（在 `proposal-FactStruct.md` 的先前版本中定义）虽然在理论上是自适应的，但其 API 成本过高（每次迭代都需要 2 次 LLM 调用，总计 $1 + 2 \times T_{max}$ 次）。

为解决此成本问题，我们引入**“批量-信息觅食多臂老虎机” (Batch-IF-MAB)** 框架。

核心思想：

我们保留 MAB (UCB1) 的自适应探索策略，但在执行层面进行“批量处理”。系统不再一次选择 1 个节点，而是一次选择 Top-K 个最值得探索的节点，然后对这 K 个节点进行批量查询生成和批量大纲修订。

**成本优势 (示例)：**

- 总预算 $T_{max} = 20$  
    
- 批量大小 $K = 5$  
    
- 总轮数 = $T_{max} / K = 4$ 轮
    
- **总 LLM 调用 = 1 (初始大纲) + 4 轮** $\times$ **(1 批量查询 + 1 批量修订) = 9 次**
    
- 这远低于原方案的 41 次调用，实现了成本与智能的平衡。
    

##### **A. MAB 组件定义**

1. “手臂” (Arms) $A_t$：
    
    在 $t$ 时刻，MAB 的“手臂”集合是当前大纲树 $O_t$ 中所有可被扩展的叶子节点 (Leaf Nodes)。
    
2. “拉动摇臂” (Action $a_t$)：
    
    在一个“轮次”（Round）中，系统根据一个策略（见 B 节）选择 Top-K 个叶子节点 $\{n^*_1, ..., n^*_K\}$。拉动这 K 个摇臂意味着：
    
    - **批量生成查询：** `Queries = LLM.Batch_QueryGen([n^*_1, ..., n^*_K])`。利用 LLM 在一次调用中，为 K 个节点并行生成 K 个高质量查询。
        
    - **并行执行检索：** `D_new_list = ParallelSearch(Queries)`。获取 K 组新文档。
        
3. “奖励” (Reward $r_t$)：
    
    奖励函数 $r_t$ 的定义与原 MAB 方案完全相同，但我们会并行计算 K 次。对于每个 $(n^*_i, D_{new\_i})$ 对，我们计算其“信息增益”：
    
    $$r_t(D_{new\_i}) = w_{\text{rel}} \cdot \text{Relevance} + w_{\text{nov}} \cdot \text{Novelty} + w_{\text{qual}} \cdot \text{Quality}$$
    - **Relevance (相关性)：** $D_{new\_i}$ 与查询节点 $n^*_i$ 的平均语义相关度。
        
    - **Novelty (新颖性)：** $D_{new\_i}$ 与全局文档池 $D_{all}$ 的差异度。
        
    - **Quality (质量)：** $D_{new\_i}$ 中文档的平均“内在质量”。
        

##### **B. 策略 (Policy): UCB1 (Top-K Selection)**

我们仍然采用 **UCB1 (Upper Confidence Bound)** 策略。但在每一轮（Round）开始时，我们为_所有_叶子节点 $n_i$ 计算其 UCB 分数，并**选择分数最高的 Top-K 个节点**进行批量处理。

$$\text{UCB\_Score}(n_i) = \underbrace{\bar{x}_i}_{\text{Exploitation}} + \underbrace{\sqrt{\frac{2 \ln t}{N_i(t)}}}_{\text{Exploration}}$$

- $\bar{x}_i$：节点 $n_i$ 迄今为止获得的**平均奖励** (Average Reward)。
    
- $t$：当前的总迭代次数（注意：`t` 仍然是从 1 累加到 $T_{max}$）。
    
- $N_i(t)$：节点 $n_i$ 迄今为止被“拉动”（检索）的**次数**。
    

##### **C. Stage 1 算法实现 (伪代码)**

**依赖的数据结构 (核心变更在 LLM 类):**

```
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from typing import List, Tuple, Dict

# Document 和 OutlineNode 类与之前版本相同
@dataclass
class Document:
    id: str
    text: str
    source_type: str
    timestamp: datetime
    embedding: np.ndarray = None

class OutlineNode:
    id: str
    title: str
    parent: 'OutlineNode'
    children: List['OutlineNode']
    pull_count: int = 0
    reward_history: List[float] = field(default_factory=list)
    
    def get_leaf_nodes(self) -> List['OutlineNode']: ...
    def get_parent_context(self) -> str: ...
    def avg_reward(self) -> float:
        if not self.reward_history: return 0.0
        return sum(self.reward_history) / len(self.reward_history)
    def to_text_tree(self) -> str: ...

class Memory:
    def store_docs(self, docs: List[Document]): ...
    def map_node_to_docs(self, node_id: str, docs: List[Document]): ...
    def get_all_doc_embeddings(self) -> np.ndarray: ... # (见 D 节的性能优化)
    def retrieve_by_citation_id(self, citation_id: str) -> List[Document]: ...

# --- 假设的依赖类 (LLM 类已更新) ---
class Embedder:
    def embed_docs(self, docs: List[Document]) -> List[Document]: ...

class LLM:
    def generate_initial_outline(self, query: str, docs: List[Document]) -> OutlineNode: ...
    
    # --- 新的批量方法 ---
    def batch_generate_queries(self, nodes: List[OutlineNode]) -> List[str]:
        # 单次 LLM 调用, 输入 K 个节点信息, 输出 K 个查询字符串
        # Prompt 示例:
        # "你是一个研究助手。请为以下 K 个大纲节点分别生成一个精确的搜索查询:
        # 1. 节点: '...', 上下文: '...' -> [查询 1]
        # 2. 节点: '...', 上下文: '...' -> [查询 2]
        # ...
        # K. 节点: '...', 上下文: '...' -> [查询 K]
        # 请严格按照 '查询 K: [内容]' 的格式输出。"
        ...

    def batch_refine_outline(self, 
                             current_outline: OutlineNode, 
                             node_doc_pairs: List[Tuple[OutlineNode, List[Document]]]
                             ) -> Tuple[OutlineNode, Dict[OutlineNode, List[OutlineNode]]]:
        # 单次 LLM 调用, 输入当前大纲和 K 组 (节点, 新文档)
        # 核心挑战: 确保 LLM 理解并 *分别* 对 K 个节点进行 *局部* 优化
        # Prompt 示例:
        # "这是当前的研究大纲: {current_outline.to_text_tree()}
        #
        # 我们刚刚检索了 K={len(node_doc_pairs)} 个节点, 获得了新信息。
        # 你的任务是根据这些新信息, 对大纲进行 K 次 *独立的局部优化*。
        #
        # 优化任务 1:
        #   - 节点: '{node_doc_pairs[0][0].title}'
        #   - 新信息: {Format(node_doc_pairs[0][1])}
        #   - 要求: 在该节点下增加子章节, 或修改其标题。
        #
        # 优化任务 2:
        #   - 节点: '{node_doc_pairs[1][0].title}'
        #   - 新信息: {Format(node_doc_pairs[1][1])}
        #   - 要求: ...
        #
        # ... (以此类推)
        #
        # 请在一次响应中, 输出经过这 K 次独立优化后 *最终的完整大纲树*。"
        
        # ---
        # (函数实现)
        # new_outline = ... # 解析 LLM 输出, 构建新的大纲树
        # expanded_nodes_map = ... # 记录 {parent_node: [new_child_1, new_child_2]}
        # return new_outline, expanded_nodes_map
        ...

class SearchEngine:
    # 假设 search 方法支持并行/批量
    def parallel_search(self, queries: List[str], k: int) -> List[List[Document]]: ...

class RewardCalculator:
    def calculate_reward(self, new_docs: List[Document], node_title: str, all_doc_embeddings: np.ndarray) -> (float, dict): ...

def Format(docs: List[Document]) -> str: ...
```

**主算法流程 (Batch-MAB):**

```
import math

# --- 输入 ---
Initial_Query: str
Max_Iterations (T_max): int # e.g., 20
Batch_Size (K): int         # e.g., 5

# --- 依赖 ---
Embedder: Embedder
LLM: LLM
SearchEngine: SearchEngine
RewardCalc: RewardCalculator
Memory: Memory

# --- 初始化 ---
# 1. 初始检索与大纲生成 (LLM Call #1)
D_initial = SearchEngine.parallel_search([Initial_Query], k=5)[0]
D_initial_with_embed = Embedder.embed_docs(D_initial)
Memory.store_docs(D_initial_with_embed)

O_root = LLM.generate_initial_outline(Initial_Query, D_initial_with_embed)
t = 0 # 总迭代次数计数器

# --- 迭代循环 (Batch-MAB 过程) ---
# 总共执行 T_max / K 轮
num_rounds = math.ceil(T_max / Batch_Size)

for round_num in range(num_rounds):
    
    # 1. 获取当前所有可行动的 "手臂"
    CurrentLeafNodes = O_root.get_leaf_nodes()
    if not CurrentLeafNodes:
        break # 没有叶子节点，提前终止

    # 2. UCB 策略选择 Top-K 手臂
    ucb_scores = []
    for node in CurrentLeafNodes:
        # UCB 公式中的 t 使用当前的全局迭代次数 (t+1)
        t_current = t + 1 
        if node.pull_count == 0:
            # 优先探索未被拉动过的节点 (或 t=0 时)
            ucb_score = float('inf')
        else:
            avg_reward = node.avg_reward()
            exploration_bonus = math.sqrt(2 * math.log(t_current) / node.pull_count)
            ucb_score = avg_reward + exploration_bonus
        ucb_scores.append((ucb_score, node))

    # 排序并选出 Top-K
    ucb_scores.sort(key=lambda x: x[0], reverse=True)
    SelectedNodes = [node for score, node in ucb_scores[:Batch_Size]]
    
    if not SelectedNodes:
        break

    # 3. "批量拉动摇臂" (执行检索)
    # (LLM Call #Round*2)
    Queries = LLM.batch_generate_queries(SelectedNodes)
    D_new_list = SearchEngine.parallel_search(Queries, k=3)
    
    # 预处理新文档 (嵌入)
    D_new_list_with_embed = [Embedder.embed_docs(docs) for docs in D_new_list]
    
    # 4. 批量计算并记录 "奖励"
    all_embeds = Memory.get_all_doc_embeddings()
    node_doc_pairs_for_refine = []
    
    for i, node in enumerate(SelectedNodes):
        t += 1 # 增加全局迭代计数器
        D_new = D_new_list_with_embed[i]
        
        r_t, breakdown = RewardCalc.calculate_reward(
            new_docs=D_new, 
            node_title=node.title, 
            all_doc_embeddings=all_embeds
        )
        
        # 更新 MAB 状态
        node.reward_history.append(r_t)
        node.pull_count += 1
        
        # 准备用于 LLM 修订的数据
        node_doc_pairs_for_refine.append((node, D_new))
        
        # 5. 更新记忆库
        Memory.store_docs(D_new)
        Memory.map_node_to_docs(node.id, D_new)

    # 6. (关键) LLM 批量更新大纲
    # (LLM Call #Round*2 + 1)
    if node_doc_pairs_for_refine:
        O_root, expanded_nodes_map = LLM.batch_refine_outline(
            O_root, node_doc_pairs_for_refine
        )

        # --- (已修正) 关键的状态继承步骤 ---
        # 遍历那些刚刚被扩展的节点 (从叶子节点变成了内部节点)
        # expanded_nodes_map 格式: {parent_node: [new_child_node_1, ...]}
        for parent_node, new_children in expanded_nodes_map.items():
            if not new_children:
                continue
            
            # 将父节点的 MAB 状态复制给所有新生成的子节点
            for child in new_children:
                child.pull_count = parent_node.pull_count
                # 必须创建 reward_history 的副本
                child.reward_history = list(parent_node.reward_history) 
        # --- 状态继承结束 ---

# --- 输出 ---
# return O_root, Memory
```

##### **D. 实现考量与性能优化**

1. **Novelty 计算瓶颈：**
    
    - (与前一版本相同) 这个问题依然存在。`Memory` 模块内部必须使用**向量索引（Vector Index）**（如 FAISS）来加速 `get_all_doc_embeddings()` 的近邻搜索，否则 $O(N \cdot M \cdot D)$ 的复杂度在 $t$ 增大时不可接受。
        
2. **LLM 批量修纲的稳定性 (新挑战)：**
    
    - **问题：** `LLM.batch_refine_outline` 是本方案最复杂的部分。它要求 LLM 在单次调用中理解并执行 K 个_独立_的局部修改。
        
    - **风险：** LLM 可能会“偷懒”，例如只执行 K 个任务中的一部分，或者在执行任务 2 时错误地“污染”了任务 1 附近的节点。
        
    - **解决方案：**
        
        1. **强力 Prompt 约束：** 如伪代码中所示，Prompt 必须极度清晰，使用数字列表、分隔符等方式，强制 LLM 按顺序、独立地处理 K 个任务。
            
        2. **输出解析与验证：** 在收到 `batch_refine_outline` 的输出后，系统可能需要一个验证步骤，检查返回的新大纲树是否真的包含了对 K 个目标节点（或其父节点）的修改。
            
        3. 状态继承 (Mandatory)：
            
            这是算法的核心要求，而非可选优化。
            
            如C节（主算法流程）的第 6 步所示，当一个叶子节点** $n$ 被扩展并生成新的子节点 $n_a, n_b, ...$ ****时，系统必须**将 $n$ 的 pull_count 和 reward_history (副本) 复制到所有新的子节点 ($n_a, n_b, ...$)。
            
            - **效果：** 这确保了新“手臂” ($n_a, n_b$) 继承了其父分支的“利用”值 ($\bar{x}_i$)，使得 MAB 策略能够在新节点上继续进行“利用”和“探索”的权衡，**从而避免了算法退化为“纯探索”**。
                



            

#### **Stage 2: 文档选择与排序**

本模块对应“如何在篇幅受限的条件下，从大量候选文献中选取最相关、最可靠、最新的支撑文档？”这一子问题。

**与 Stage 1 的关系：** Stage 1 (MAB) 的目标是_探索_并_构建_一个好的大纲，其 `Memory` 中存储了节点与“相关文档池”的_粗粒度_映射。Stage 2 的目标是在_写作_前，从这个池子中进行_精细化_筛选。

现有长文写作系统往往只依赖“语义相似度”来选择支撑文档。

这会导致以下三个问题：

- 信息噪声：相关但不可靠的文档被选中；
    
- 过时信息：旧文献被误认为最相关；
    
- 事实片面：信息密度低、覆盖不足。
    

为此，我们引入一个能够综合考虑**语义相关性、可信度、信息密度与时效性**的排序机制，确保引用文献既“相关”又“可靠”。

##### 实现方案：**Multi-factor Reranker**

对于大纲（大纲可以视为一棵树）中的每个节点 $n_i$，

系统根据Stage 1中存储的映射关系从记忆库中取出相应的候选文档集 $D_i = \{d_1, d_2, …, d_k\}$。

然后使用启发式的多因素评分模型进行加权排序：

$$\text{Score}(d_j) = w_1 \text{Sim}(d_j) + w_2 \text{Cred}(d_j) + w_3 \text{Density}(d_j) + w_4 \text{Fresh}(d_j)$$

其中：

- **Sim**：主题–文档语义相似度（通过向量嵌入计算）；
    
- **Cred**：来源可信度（例如：学术论文 > 博客 > 社交媒体）；
    
- **Density (密度)**：文档中与主题相关句子的比例。（**注意：** 这是一个_高成本_计算，不用于 Stage 1 的 MAB 奖励，但用于此处的最终精排）。
    
- **Fresh**：基于时间衰减函数的时效性因子，
    
    $$ \text{Fresh}(d_j) = e^{-\lambda (t_{\text{now}} - t_{d_j})}$$
    
    其中 $\lambda$ 为时间衰减系数。
    

每个大纲节点最终获得一组**经过排序的支撑文档列表**。排名越靠前的文档在写作时引用优先级越高。这一阶段确保系统生成的内容“有根据”，并优先吸收最新、可信的信息来源。


---

#### **Stage 3: 层次化写作系统**

本模块虽然有创新点，但和研究的整体目标似乎不太搭（正在考虑如何把本模块解决的问题融入到我们开篇定义的科学问题中）。

在长篇科研报告生成中，让语言模型一次性根据所有检索到的支撑文档和大纲生成整篇文本会导致显著的质量下降。这是因为语言模型的上下文窗口有限，当输入包含大量支撑文档时，
模型的注意力会被过度分散，出现 上下文污染（context contamination）——即模型在生成某章节时受到其他部分语义的干扰。

这种污染带来一系列问题：
- 主题漂移（topic drift）：模型偏离当前章节主题，混入无关内容；
- 引用错乱（citation misalignment）：不同章节的支撑文档被错误引用；
- 逻辑不连贯（coherence break）：章节间衔接混乱，论述跳跃。

因此，引入层次化写作的方式，逐一生成大纲中的各章节（大纲中的一个节点）再整合而非一次性生成整篇文章。写作每一章节前从记忆库根据映射关系取出相关支撑文档，写完一节即清除上下文，保持专注，避免“上下文污染”。

本模块实现有以下两种选项。

##### (A) naive版 — **Hierarchical Sequential Writer**

* 自上而下逐章写作；
* 每章使用 reranker 提供的文档；
* 子章节继承父章节摘要；
* 自底向上合并结果。

```python
def sequential_writer(outline, reranker, agent):
    report = {}
    for ch in outline:
        docs = reranker.get_top_docs(ch)
        text = agent.generate(ch.title, docs)
        for sub in ch.subsections:
            sub_text = agent.generate(sub.title, docs, parent_context=text)
            text += sub_text
        report[ch.id] = text
    return assemble(report)
```

---

##### (B) Full 版 — **Hierarchical Dependency Graph Parallel Writer**

  - 当动态大纲确定后，我们构建一个层次依赖图（Hierarchical Dependency Graph, HDG），其中每个节点对应大纲中的一个部分，边表示语义或逻辑依赖关系。若两个节点不存在直接依赖，则可并行生成。
  - 在写作阶段，系统首先对 HDG 执行拓扑排序，确定写作顺序与并行分组；随后，对每个可并行节点集启动并行写作，从记忆库检索相关支撑文档并生成对应部分。
  -  当所有子节点完成后，其输出文本与引用被作为上下文注入到父节点，用于自底向上的章节生成。最终，顶层节点汇总形成完整报告。

```python
def hdg_parallel_writer(hdg, agent_pool):
    layers = topological_layers(hdg)
    for L in layers:
        results = parallel_generate(L, agent_pool)
        merge_context(results)
    return aggregate(results)
```



---

#### **Stage 4: 内容一致性校验与自修订**

本模块对应子问题“如何通过语义推理模型判断生成文本是否被引用文献真正支撑，并在不一致时进行自动修订？”。

LLM 写作的最大风险是“语言正确但事实错误”。句子可能看似合理，却与引用文档的语义相矛盾。人工检查成本极高，因此我们希望让系统自动识别并修正事实不一致。

##### 实现方案：**NLI-based Consistency Checker**

对于每一处引用所在的句子，检查支撑文档是否能够支撑（语义蕴含）当前句子，如果不能支撑，则根据支撑文档对当前句子进行rewrite，在保持原句语义的前提下，使修改后的句子与根据支撑文档尽可能保持一致。具体来说，
* 使用 NLI 模型判断支撑文档对句子的语义蕴含关系；
* 若结果为 “neutral” 或 “contradiction” → 触发重写。

**流程：**

```python
for sentence in report.sentences_with_citations():
    evidence = memory.retrieve(sentence.citation_id)
    result = NLI_model(premise=evidence, hypothesis=sentence)
    if result != "entailment":
        sentence = rewrite(sentence, evidence)
```


---

## 3. 实验计划

### 数据与任务环境

采用DeepResearch Bench这个benchmark


* **评测指标：**

  * RACE（结构与完整性）
  * FACT（事实支撑率）

### 实验设计

* **对比实验：**

  * WebWeaver (2025) — 双代理动态规划系统
  * WriteHERE (2025) — 异质递归规划框架
  * DeepResearcher (2024) — 检索增强型写作
  * RAG-only Baseline — 最小检索增强系统
* **消融实验：**

  * 与本方法的“无动态修纲 / 无 reranker / 无并行 / 无一致性校验”四个变体进行对比


---

## 4. 预期贡献总结

本研究提出的 **FactStruct** 系统在开放式科研写作中实现了从“流畅生成”到“事实一致”的范式转变。主要贡献如下：

1. **提出多因素文档选择机制**
   将多因素纳入排序指标，使引用更符合写作需求。

2. **构建基于语义蕴含的事实一致性检测与修订模块**
   通过 NLI 模型逐句检测事实支撑性，并在矛盾时自动重写，显著提升报告的事实可信度。



