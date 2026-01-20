import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents.sub_agent_registry import get_sub_agents_by_global_type
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.memory import MemoryStack, MemoryStackEntry
from src.prompts.template import apply_prompt_template, get_prompt_template
from src.utils.json_utils import repair_json_output
from src.utils.logger import logger
from src.utils.statistics import global_statistics
# from src.prompts.central_decision import Decision, DelegateParams
from src.utils.reference_utils import global_reference_map
from ..graph.types import State
# from .SubAgentConfig import get_sub_agents_by_global_type
# from src.factstruct.llm_wrapper import FactStructLLMWrapper
# from src.factstruct.batch_mab import BatchMAB
# from src.factstruct.outline_node import OutlineNode
from src.factstruct import FactStructLLMWrapper, BatchMAB, OutlineNode, create_search_engine_adapter,Embedder



from src.utils.statistics import global_statistics, timed_step

# -------------------------
# 核心枚举定义
# -------------------------
class OutlineTool(Enum):
    """Outline Agent 可执行的结构性工具"""

    INITIALIZATION = "initialization"
    EXPANDATION = "expandation"
    REDUCTION = "reduction"
    REFLECT = "reflect"
    FINISH = "finish"



# -------------------------
# 中枢Agent核心模块--中枢Agent的action
# exp: 与prompt/Outline_decision.py中的Decision类不同的是，那里是字符串类型，这里是枚举类型，所以要定义两次
# -------------------------
@dataclass
class OutlineToolDecision:
    """Outline Agent 的决策结果"""

    tool: OutlineTool                 # 使用的工具
    reasoning: str                    # 为什么这么做
    params: Optional[Dict[str, Any]]  # 工具参数（各 tool 自己解释）



#不知道为啥要两个，那就实现成两个吧
from pydantic import BaseModel

class OutlineToolDecision_Base(BaseModel):
    tool: Literal[
        "initialization",
        "expandation",
        "reduction",
        "reflect",
        "finish",
    ]
    reasoning: str
    params: Optional[Dict[str, Any]] = None





class OutlineAgent:
    """
    大纲Agent核心类，负责大纲决策与任务编排
    """
    
    def __init__(
        self,
        initial_query: str,
        central_guidance: str | None = None,
        state: State | None = None,
        llm=None,#这三个就是 None
        search_engine=None,
        embedder=None,
        max_trys: int = 5,#这两个是默认参数
    ):
        #处理上游输入信息
        # --- Core task signal ---
        self.initial_query = initial_query
        # --- High-level planning signals ---
        self.central_guidance = central_guidance
        self.replan_result = state.get("replan_result")
        self.total_word_limit = state.get("total_word_limit")
        
        
        #处理当前状态，这玩意是不是应该放到 decision 里面也要啊
        # --- Current outline state ---
        self.factstruct_outline = state.get("factstruct_outline")
        self.factstruct_memory = state.get("factstruct_memory")
        self.outline_feedback = state.get("outline_feedback")
        self.max_trys = max_trys

        
        # === Search Engine ===
        self.search_engine = (search_engine or create_search_engine_adapter())

        # === Embedder（重资源，只初始化一次）===
        self.embedder = (embedder or Embedder(model_name="../../Model/MiniLM/all-MiniLM-L6-v2"))


        # === LLM ===
        if llm is None:
            llm_type = AGENT_LLM_MAP.get("outline", "basic")
            self.llm = get_llm_by_type(llm_type)
        else:
            self.llm = llm
            
        self.llm_wrapper = FactStructLLMWrapper(self.llm)
        
        
        # --- Batch MAB（核心）---
        self.batch_mab = BatchMAB(
            llm_wrapper=self.llm_wrapper,
            embedder=self.embedder,
            search_engine=self.search_engine,
            max_iterations=4,
            batch_size=2,
        )

        # --- Tool handlers ---
        self.tool_handlers = {
            "initialization": self._tool_initialization,
            "expandation": self._tool_expansion,
            # "reduction": self._tool_reduction,#未实现，暂时注释
            # "reflect": self._tool_reflect,#未实现，暂时注释
            "finish": self._tool_finish,
        }


    async def execute(self, state: State, config: RunnableConfig) -> Command:
        """
        OutlineAgent 的主循环：
        decision → tool → state update → until finish
        """
        logger.info("OutlineAgent 执行开始")
        #万一有报错信息
        last_decision = None
        last_error = None

        for step in range(self.max_trys):
            logger.info(f"OutlineAgent Step {step + 1}/{self.max_trys}")#一共迭代了多少步

            try:
                # === 1. 决策 ===
                decision = self.make_decision(state, config)
                last_decision = decision

                logger.info(f"OutlineAgent Decision: {decision.tool} | {decision.reasoning}")

                # === 2. 执行工具 ===
                command = await self.execute_tool(decision, state, config)
            
                # === 3. 合并 state ===
                if command and command.update:
                    state.update(command.update)
                
                # === 4. 是否完成 ===
                if decision.tool == "finish":
                    logger.info("OutlineAgent 收到 finish 指令，退出循环")
                    break
                
            except Exception as e:
                import traceback

                logger.error("OutlineAgent 执行异常")
                logger.error(traceback.format_exc())
                last_error = str(e)
                break

        # =========================
        # === 统一结果整理阶段 ===
        # =========================
        factstruct_outline = state.get("factstruct_outline")
        factstruct_memory = state.get("factstruct_memory")
        report_outline = state.get("report_outline")
        if not report_outline: #没有大纲的情况下要反馈
            report_outline = "大纲未完成生成（OutlineAgent 未正常 finish）"
            state["report_outline"] = report_outline
  
            
        return Command(
            update={
                "factstruct_outline": state.get("factstruct_outline"),
                "factstruct_memory": state.get("factstruct_memory"),
                "report_outline": state.get("report_outline"),
            }
        )





    def make_decision(
        self, state: State, config: RunnableConfig, retry_count: int = 0
    ) -> OutlineToolDecision:
        """
        中枢Agent决策核心逻辑，分析当前状态生成决策结果

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            决策结果对象
        """
        max_retries = 3
        logger.info("Outline Agent进行决策...")
        start_time = datetime.now()

        # 构建决策prompt
        messages = self._build_decision_prompt(state, config)
        logger.debug(f"outline 决策prompt: {messages}")


        try:
            llm = get_llm_by_type(
                AGENT_LLM_MAP.get("outline", "default")
            ).with_structured_output(
                OutlineToolDecision_Base,   # ✅ 给 LLM 用的 schema
                method="json_mode",
            )

            response: OutlineToolDecision_Base = llm.invoke(messages)

            logger.info(f"Outline 决策结果(raw): {response}")

            # ✅ 从 LLM 协议对象 → 系统内部对象
            decision = OutlineToolDecision(
                tool=response.tool,
                reasoning=response.reasoning,
                params=response.params,
            )

            end_time = datetime.now()
            global_statistics.add_time_entry(
                {
                    "step_name": "outline_decision",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
            )

            return decision

        except Exception as e:
            import traceback

            logger.error(
                f"Outline 决策解析失败 (尝试 {retry_count + 1}/{max_retries}): {e}"
            )
            logger.error(traceback.format_exc())

            if retry_count < max_retries - 1:
                return self.make_decision(state, config, retry_count + 1)

            # 🚨 兜底：强制 finish，防止系统卡死
            return OutlineToolDecision(
                tool="finish",
                reasoning="Outline decision parsing failed repeatedly, forcing termination.",
                params=None,
            )


    def _build_decision_prompt(
        self,
        state: State,
        config: RunnableConfig,
    ) -> List[Union[AIMessage, HumanMessage]]:
        """
        构建 Outline Agent 的决策 prompt
        """

        context = {
            # 必须项
            "user_query": state.get("initial_query"),

            # 可选项（prompt 里有 if 判断）
            "central_guidance": state.get("central_guidance"),
            "factstruct_outline": state.get("factstruct_outline"),
            "total_word_limit": state.get("total_word_limit"),
            "outline_feedback": state.get("outline_feedback"),
        }

        # 合并 config（如需要）
        context = {**context, **config}

        return apply_prompt_template(
            "outline_decision",   # ✅ 对应 src/prompts/outline_decision.md
            state,
            extra_context=context,
        )


    def execute_tool(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        根据 Outline Agent 的 tool 决策，调用对应的 outline 工具函数
        """

        tool_name = decision.tool
        handler = self.tool_handlers.get(tool_name)

        if not handler:
            error_msg = f"未知 outline tool: {tool_name}"
            logger.error(error_msg)
            #这个地方要不要这么跳，还不一定，还需要再看一看
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"错误：未知 outline tool: {tool_name}",
                            name="outline_error",
                        )
                    ],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )


        logger.info(
            f"Outline Agent 执行工具: {tool_name}, params={decision.params}"
        )

        return handler(
            decision=decision,
            state=state,
            config=config,
        )


    def _tool_initialization(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        初始化大纲结构（initialization tool）
        """
        logger.info("Outline Tool: initialization")

        initial_query = state["initial_query"]
        central_guidance = state.get("central_guidance")
        replan_result = state.get("replan_result")
        factstruct_outline = state.get("factstruct_outline")
        initial_docs = state.get("initial_docs")
        
        #提取decision当中的参数
        params = decision.params or {}
        instruction=params.get("instruction",None)

        # 已有 outline，不应再次初始化
        if factstruct_outline is not None:
            logger.warning("Outline already exists, skip initialization.")
            return Command(
                update={
                    "outline_feedback": "Initialization skipped: outline already exists."
                },
            )

        # --- Step 1: 初始检索 ---
        if initial_docs is None:
            logger.info("开始预检索")
            initial_docs = self.search_engine(initial_query, k=5, config=config)

        # --- Step 2: 向量化 ---
        initial_docs_with_embed = self.embedder.embed_docs(initial_docs)

        # --- Step 3: 存入 FactStruct Memory ---
        self.factstruct_memory.store_docs(initial_docs_with_embed)

        # --- Step 4: 生成初始大纲 ---
        logger.info("Generating initial outline...")
        self.outline_root = self.llm_wrapper.generate_initial_outline(
            query=initial_query,
            docs=initial_docs_with_embed,
            central_guidance=central_guidance,
            replan_result=replan_result,
            instruction=instruction,
        )

        logger.info(f"Generated outline root id: {self.outline_root.id}")

        # --- Step 5: 文档绑定 ---
        self.factstruct_memory.map_node_to_docs(self.outline_root.id, initial_docs_with_embed)

        # --- Step 6: 日志 ---
        try:
            from .integration import outline_node_to_markdown
            logger.info("Initial Outline:\n"+ outline_node_to_markdown(self.outline_root, include_root=True))
        except Exception:
            logger.info(self.outline_root.to_text_tree())

        # --- Step 7: 更新状态并回到 outline agent ---
        return Command(
            update={
                "factstruct_outline": self.outline_root,
                "factstruct_memory":self.factstruct_memory,
                "initial_docs": initial_docs,
                "outline_tool_feedback": "Outline initialized successfully.",
            },
        )

    def _tool_expansion(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        扩展现有大纲（Batch-MAB 驱动）
        """
        logger.info(f"Outline Tool: expansion | reasoning={decision.reasoning}")


        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")
        #错误排查，防止没初始化
        if outline_root is None or memory is None:
            logger.warning("Expansion skipped: outline or memory missing")
            return Command(
                update={
                    "outline_feedback": "Expansion skipped: outline or memory missing."
                }
            )



        #提取 decision 参数
        params = decision.params or {}
        max_iterations = params.get("max_iterations",state.get("factstruct_max_iterations", 4),)
        batch_size = params.get("batch_size",state.get("factstruct_batch_size", 2),)
        logger.info(f"Expansion params resolved: max_iterations={max_iterations}, batch_size={batch_size}")
        
        

        try:
            # === 调用算法层 ===
            outline_root, memory = self.batch_mab.run_expansion(
                outline_root=outline_root,
                memory=memory,
                max_iterations=max_iterations,
                batch_size=batch_size,
                config=config,
            )

            logger.info(
                f"Outline expanded: {len(outline_root.get_all_nodes())} nodes total"
            )
            
            #暂时在这里放一个大纲字数匹配吧
            total_word_limit = state.get("total_word_limit", 5000)
            if total_word_limit > 0:
                logger.info(f"检测到字数限制 {total_word_limit}，执行字数规划...")
                outline_root = self.execute_word_planning(
                    outline_root, total_word_limit
                )
                outline_response = outline_root.to_text_tree(
                    include_word_limit=True
                )
            else:
                outline_response = outline_node_to_markdown(
                    outline_root, max_depth=None, include_root=True
                )

            factstruct_outline_dict = outline_node_to_dict(outline_root)
            factstruct_memory_dict = memory_to_dict(memory)

            logger.info(f"FactStruct Stage 1 完成: "f"{len(outline_root.get_all_nodes())} 个节点")

            return Command(
                update={
                    "factstruct_outline": outline_root,
                    "factstruct_memory": memory,
                    "outline_feedback": "Outline expansion completed successfully.",
                }
            )

        except Exception as e:
            import traceback
            logger.error("Outline expansion failed")
            logger.error(traceback.format_exc())

            return Command(
                update={
                    "outline_feedback": f"Outline expansion failed: {str(e)}"
                }
            )



    def _tool_finish(
        self,
        decision: OutlineToolDecision,
        state: State,
        config: RunnableConfig,
    ) -> Command:
        """
        完成 OutlineAgent 执行，生成总结性反馈返回给 CentralAgent
        """
        logger.info(f"Outline Tool: finish | reasoning={decision.reasoning}")

        outline_root = state.get("factstruct_outline")
        memory = state.get("factstruct_memory")
        total_word_limit = state.get("total_word_limit")

        # === 兜底处理 ===
        if outline_root is None:
            logger.warning("Finish called but outline is missing")

            return Command(
                update={
                    "report_outline": "大纲未成功生成（OutlineAgent 在 finish 前缺失 outline）",
                    "outline_feedback": "Finish reached without valid outline.",
                }
            )

        # === 基本结构信息 ===
        node_count = len(outline_root.get_all_nodes())
        leaf_count = len(outline_root.get_leaf_nodes())

        # === 可读大纲输出（给 Central / Reporter 用）===
        try:
            outline_text = outline_root.to_text_tree(
                include_word_limit=bool(total_word_limit)
            )
        except Exception:
            outline_text = outline_root.to_text_tree()

        # === 构建给 LLM 的总结 prompt（可选，但很有价值）===
        try:
            llm = get_llm_by_type(
                AGENT_LLM_MAP.get("outline", "default")
            )

            context_lines = []

            if state.get("initial_query"):
                context_lines.append(f"用户原始问题：{state.get('initial_query')}")

            if state.get("central_guidance"):
                context_lines.append(f"中枢策略指导：{state.get('central_guidance')}")

            if total_word_limit:
                context_lines.append(f"目标总字数限制：{total_word_limit}")

            context_block = "\n".join(context_lines) if context_lines else "（无额外上下文）"

            summary_prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是一个研究大纲评估助手，"
                        "你的任务不是生成内容，而是判断当前大纲是否已经达到"
                        "可以进入正式内容生成阶段的结构成熟度。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"=== 当前大纲结构指标 ===\n"
                        f"- 节点总数: {node_count}\n"
                        f"- 叶子节点数: {leaf_count}\n\n"
                        f"=== 当前大纲 ===\n"
                        f"{outline_text}\n\n"
                        "请基于【用户问题】和【中枢策略指导】，判断该大纲：\n"
                        "1. 是否覆盖了用户问题的主要方面\n"
                        "2. 结构层级是否清晰、粒度是否合适\n"
                        "3. 是否适合在当前字数限制下展开正文\n\n"
                        "请给出一句简洁的总结性判断（不超过 2 句话），"
                        "用于上游 Agent 的决策参考。"
                    ),
                },
            ]


            llm_response = llm.invoke(summary_prompt)
            finish_summary = llm_response.content.strip()

        except Exception as e:
            logger.warning(f"Finish summary LLM failed: {e}")
            finish_summary = (
                "大纲结构已生成，节点层级完整，可进入内容生成阶段。"
            )

        # === 写回 state（这是 CentralAgent 最关心的部分）===
        return Command(
            update={
                "report_outline": outline_text,              # ✅ 给 reporter / central
                "outline_feedback": finish_summary,           # ✅ 决策级总结
            }
        )




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

        # 构建大纲结构信息供LLM分析
        def build_outline_info(node: OutlineNode, depth: int = 0) -> list:
            nodes_info = []
            nodes_info.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "depth": depth,
                    "is_leaf": node.is_leaf(),
                }
            )
            for child in node.children:
                nodes_info.extend(build_outline_info(child, depth + 1))
            return nodes_info

        outline_info = build_outline_info(outline_root)
        leaf_nodes = [n for n in outline_info if n["is_leaf"]]

        # 构建LLM请求
        outline_text = outline_root.to_text_tree()
        prompt_content = f"""请为以下报告大纲分配字数。

        ## 大纲结构
        {outline_text}

        ## 叶子节点列表
        {json.dumps(leaf_nodes, ensure_ascii=False, indent=2)}

        ## 总字数限制
        {total_word_limit} 字

        请根据每个叶子节点的重要性和内容复杂度，智能分配字数配额。
        你必须只输出一个合法的 JSON 对象。禁止输出任何解释、说明、注释、标题或额外文本。如果输出包含非 JSON 内容，将被视为错误。
        """

        try:
            messages = apply_prompt_template("word_planner", {"messages": []}) + [
                HumanMessage(content=prompt_content)
            ]
            llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
            response = llm.invoke(messages)
            result = response.content

            # 解析JSON结果
            logger.info(f"result:{result}")
            # result = result.replace("```json", "").replace("```", "").strip()

            match = re.search(r"\{[\s\S]*\}", result)
            if not match:
                raise ValueError("No JSON object found in LLM output")

            allocations = json.loads(match.group(0))

            # 将字数配额写入节点
            for alloc in allocations.get("allocations", []):
                node_id = alloc.get("node_id")
                word_limit = alloc.get("word_limit", 0)
                node = outline_root.find_node_by_id(node_id)
                if node:
                    node.word_limit = word_limit
                    logger.debug(
                        f"节点 {node_id} ({node.title}) 分配字数: {word_limit}"
                    )

            # 自底向上计算非叶子节点的字数
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

        logger.info(f"outline_root:{outline_root}")
        # exit()
        return outline_root
