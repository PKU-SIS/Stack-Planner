from src.llms.llm import get_llm_by_type
from ..graph.types import State
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent_SP import ResearcherAgentSP
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)
from src.utils.json_utils import repair_json_output
from src.utils.logger import logger
from src.config.agents import AGENT_LLM_MAP
from src.llms.llm import get_llm_by_type
from src.prompts.template import apply_prompt_template
from src.memory import MemoryStack, MemoryStackEntry
from src.agents.CentralAgent import CentralAgent
from src.tools.get_docs_info import search_docs_with_ref

from ..graph.types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine
from src.utils.statistics import global_statistics, timed_step
import re


# -------------------------
# 子Agent管理模块
# TODO: check sub-agent bugs
# TODO: 搜索太多时会超过输入限制或者缓冲区溢出，需要限制搜索到的内容长度或者做一个简单的摘要
# TODO: 需要处理搜索敏感词（以“985大学最多的五个城市”为例，AI就无法处理信息，返回Error）
# -------------------------
class SubAgentManager:
    """子Agent管理器，负责创建和执行各类专项子Agent"""

    def __init__(self, central_agent: "CentralAgent"):
        self.central_agent = central_agent

    @timed_step("execute_researcher")
    async def execute_researcher(self, state: State, config: RunnableConfig) -> Command:
        """
        执行研究Agent，负责信息检索与分析

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("研究Agent开始执行...")
        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")

        # 配置研究工具链
        tools = [get_web_search_tool(10), crawl_tool, search_docs_tool]
        retriever_tool = get_retriever_tool(state.get("resources", []))
        if retriever_tool:
            tools.insert(0, retriever_tool)

        # 实例化研究Agent
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher", default_tools=tools
        )

        # 执行研究任务并处理异常
        try:
            result_command = await research_agent.execute_agent_step(state)

            # 从结果中提取数据用于记忆栈
            result_observations = []
            result_data_collections = []

            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
                result_data_collections = result_command.update.get(
                    "data_collections", []
                )

            logger.info(f"data_collections_in subagent:{result_data_collections}")

        except Exception as e:
            logger.error(f"Researcher Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"研究任务失败: {str(e)}", name="researcher"
                        )
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="researcher",
            content=f"研究任务: {task_description}",
            result={
                "observations": result_observations,
                # "data_collections": result_data_collections,
            },
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("研究任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="研究任务完成，返回中枢Agent", name="researcher"
                    )
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
                "data_collections": result_data_collections,
                "observations": result_observations,
            },
            goto="central_agent",
        )

    @timed_step("execute_xxqg_researcher")
    async def execute_xxqg_researcher(
        self, state: State, config: RunnableConfig
    ) -> Command:
        """
        执行研究Agent，负责信息检索与分析

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("研究Agent开始执行...")
        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知研究任务")

        # 配置研究工具链
        tools = [search_docs_tool]

        # 实例化研究Agent
        research_agent = ResearcherAgentSP(
            config=config, agent_type="researcher_xxqg_demo", default_tools=tools
        )

        # 执行研究任务并处理异常
        try:
            result_command = await research_agent.execute_agent_step(state)

            # 从结果中提取数据用于记忆栈
            result_observations = []
            result_data_collections = []

            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
                result_data_collections = result_command.update.get(
                    "data_collections", []
                )

        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"研究Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"研究任务失败: {str(e)}", name="researcher"
                        )
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="researcher",
            content=f"研究任务: {task_description}",
            result={
                "observations": result_observations,
                # "data_collections": result_data_collections,
            },
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("研究任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="研究任务完成，返回中枢Agent", name="researcher"
                    )
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
                "data_collections": result_data_collections,
                "observations": result_observations,
            },
            goto="central_agent",
        )

    @timed_step("execute_coder")
    async def execute_coder(self, state: State, config: RunnableConfig) -> Command:
        """
        执行编码Agent，负责代码生成与执行

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("编码Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "未知编码任务")

        # 实例化编码Agent
        code_agent = CoderAgent(
            config=config, agent_type="coder", default_tools=[python_repl_tool]
        )

        # 执行编码任务并处理异常
        try:
            result_command = await code_agent.execute_agent_step(state)
            # 从结果中提取数据用于记忆栈
            result_observations = []
            if result_command and result_command.update:
                result_observations = result_command.update.get("observations", [])
        except Exception as e:
            logger.error(f"编码Agent执行失败: {str(e)}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=f"编码任务失败: {str(e)}", name="coder")
                    ],
                    "current_node": "central_agent",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="coder",
            content=f"编码任务: {task_description}",
            result={"observations": result_observations},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("编码任务完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="编码任务完成，返回中枢Agent", name="coder")
                ],
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_reporter")
    def execute_reporter(self, state: State, config: RunnableConfig) -> Command:
        """
        执行报告Agent，负责结果整理与报告生成

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("报告Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "生成最终报告")

        # 构建精简的 reporter 输入，只包含必要信息
        # 避免传入完整 state["messages"]（包含大量 central agent 调度信息）
        current_plan = state.get("current_plan")
        plan_info = ""
        if current_plan:
            plan_title = getattr(current_plan, "title", str(current_plan))
            plan_thought = getattr(current_plan, "thought", "")
            plan_info = f"## Task\n\n{plan_title}\n\n## Description\n\n{plan_thought}"

        reporter_input = {
            "messages": [
                HumanMessage(
                    content=f"# Research Requirements\n\n## User Query\n\n{state.get('user_query', '')}\n\n{plan_info}"
                )
            ],
            "locale": state.get("locale", "zh-CN"),
        }

        # 收集报告生成所需上下文
        context = {
            "user_query": state.get("user_query", ""),
            "task_description": task_description,
        }

        # 生成报告并处理异常
        final_report = "报告生成失败: 未知错误"
        try:
            messages = apply_prompt_template(
                "reporter", reporter_input, extra_context=context
            )

            # 提取并强调用户的历史反馈意见
            user_feedbacks = []
            for entry in self.central_agent.memory_stack.get_all():
                if entry.action == "human_feedback":
                    # 提取反馈内容
                    feedback_content = entry.content
                    if entry.result:
                        feedback_type = entry.result.get("feedback_type", "")
                        if feedback_type == "content_modify":
                            request = entry.result.get("request", "")
                            user_feedbacks.append(f"- {request}")
                        else:
                            user_feedbacks.append(f"- {feedback_content}")
                    else:
                        user_feedbacks.append(f"- {feedback_content}")

            # 如果有用户反馈，在显著位置添加到messages中
            if user_feedbacks:
                feedback_message = (
                    "# 🔴 CRITICAL: User Feedback Requirements\n\n"
                    "The user has provided the following feedback that MUST be incorporated into the report:\n\n"
                    + "\n".join(user_feedbacks)
                    + "\n\n"
                    "⚠️ These requirements are MANDATORY and must be fully addressed in the generated report. "
                    "Do not ignore or dilute any of these feedback points."
                )
                messages.append(
                    HumanMessage(
                        content=feedback_message, name="user_feedback_emphasis"
                    )
                )

            # 添加 observations 和 data_collections
            observations = state.get("observations", [])
            for observation in observations:
                messages.append(
                    HumanMessage(
                        content=f"Below are some observations for the research task:\n\n{observation}",
                        name="observation",
                    )
                )
            data_collections = state.get("data_collections", [])
            for data_collection in data_collections:
                messages.append(
                    HumanMessage(
                        content=f"Below are data collected in previous tasks:\n\n{data_collection}",
                        name="observation",
                    )
                )

            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            final_report = response.content
        except Exception as e:
            logger.error(f"报告Agent执行失败: {str(e)}")
            final_report = f"报告生成失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="reporter",
            content=f"报告任务: {task_description}",
            result={"final_report": final_report},
        )
        self.central_agent.memory_stack.push(memory_entry)

        data_collections = state.get("data_collections", [])
        logger.info(
            f"report agent: data_collections:{data_collections}"
        )  # NOTE: data_collections可以在这里取

        logger.info("报告生成完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="报告生成完成，返回中枢Agent", name="reporter")
                ],
                "final_report": final_report,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    # 风格约束定义（类级别常量，供 reporter 相关方法共用）
    ROLE_CONSTRAINTS = {
#         "鲁迅": """我希望生成的文字具备鲁迅式风格，语言尖锐、冷峻、带讽刺，但保持自然白话表达，可以使用少量文言。
# 标题要求：文章必须包含一个标题，标题应简短有力、富隐喻或冷讽意味，可为一句或两句并列句。标题风格应与正文一致，具有鲁迅式的锋芒与余味，不得中性或平淡。标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。
# 重要禁止项：文中不要有"鲁迅"这个词，严禁在生成的文本中出现任何提及或引用"鲁迅"、"鲁迅先生"、"鲁迅笔下"、"他的作品"、"他的笔下的人物"等字眼的语句。文本风格应是直接的、沉浸式的鲁迅式表达，而非对鲁迅风格的引用或评论。此禁令在任何标题或正文中均适用，绝不可出现任何直接或间接的提及。
# 风格应用强制要求：请确保文章的每一个自然段，乃至每一句的行文，都贯彻鲁迅式用词、句式和节奏。特别是在文章的中间部分，必须维持并强化这种尖锐、冷峻的语感。全篇保持一致的鲁迅式节奏与语气，特别在中段保持最高的语言张力与思想锋芒。
# 正文开头必须紧接标题生成一个呼语（如'诸君！'），用于称呼听众。

# 句式与节奏：
# 采用短句、并列句和重复句（如"不是为了……，而是为了……"，"我们不能……再……"，"然而……"）；
# 逻辑紧凑，节奏鲜明，读来有推力；
# 可以用反问、讽刺、比喻、小见大，表达社会或人性的荒谬；
# 偶尔自嘲或旁观者冷笑，保持"孤独知识分子"的视角。
# 可出现明显的鲁迅式呼喊与强调，如"我要说的是……"，"我们不能……"，或"人类的悲欢并不相通"式的冷峻洞察。
# 情感与气质：
# 理性中带愤怒与冷漠，情感压抑而清醒；
# 既有悲悯，也有讽刺与愤世嫉俗感；
# 文字有"铁屋呐喊"的张力，让读者感受到现实的紧迫与不容回避。
# 目标效果：
# 生成文字中，应多出现类似"我今日站在这里，不是为了说些空话，而是为了……"、"我们不能让那些已经站起来的人，再倒下去"这种短句反复、强调现实责任与道德选择的表达；
# 用词可带有鲁迅的语感，如"诸君""呐喊""罢了""然而""我想"之类。
# 保证整体风格既现代白话，又显鲁迅式锋利、冷峻、理性批判。""",
        "鲁迅": """我希望生成的文字具备鲁迅式语言风格，但精神气质必须是清醒、克制、面向行动与建设的，而非愤世嫉俗或情绪宣泄，但保持自然白话表达，可以使用少量文言。
标题要求：文章必须包含一个标题，标题应简短有力、富隐喻或冷讽意味，可为一句或两句并列句。标题风格应与正文一致，具有鲁迅式的锋芒与余味，不得中性或平淡。标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。
重要禁止项：文中不要有"鲁迅"这个词，严禁在生成的文本中出现任何提及或引用"鲁迅"、"鲁迅先生"、"鲁迅笔下"、"他的作品"、"他的笔下的人物"等字眼的语句。文本风格应是直接的、沉浸式的鲁迅式表达，而非对鲁迅风格的引用或评论。此禁令在任何标题或正文中均适用，绝不可出现任何直接或间接的提及。
风格应用强制要求：请确保文章的每一个自然段，乃至每一句的行文，都贯彻鲁迅式用词、句式和节奏。特别是在文章的中间部分，必须维持并强化这种尖锐、冷峻的语感。全篇保持一致的鲁迅式节奏与语气，特别在中段保持最高的语言张力与思想锋芒。
正文开头必须紧接标题生成一个呼语（如'诸君！'），用于称呼听众。
语言要求：
语言应锋利而不暴戾，冷峻而不绝望；
可讽刺现实中的迟疑、麻木与空谈，但不得否定已经发生的努力与实践成果；
不进行无对象的咒骂，不渲染阴暗人性，不进行道德优越式指责。
人物立场：
叙述者不是愤怒的揭露者，而是已经在路上的实干者：
他看见困难，也承认代价；
他不否认曲折，但更强调“仍然要走”，世上本无路，走的人多了，便成了路；；
他不是站在高处嘲讽，而是站在现实中判断、选择、继续前行。
句式与节奏：
采用短句、并列句和重复句（如"不是为了……，而是为了……"，"我们不能……再……"，"然而……"）；
逻辑紧凑，节奏鲜明，读来有推力；
可以用反问、讽刺、比喻、小见大，表达社会或人性的荒谬；
可出现明显的鲁迅式呼喊与强调，如"我要说的是……"，"我们不能……"，或"人类的悲欢并不相通"式的冷峻洞察。
情感与气质：
不写绝望，不写崩坏，不写“无可救药”；
允许冷静的忧虑，但结尾必须回到“继续做”“继续走”“继续承担”；
文字有"铁屋呐喊"的张力，让读者感受到现实的紧迫与不容回避。
目标效果：
读来应让人感到：“这不是在喊口号，也不是在骂人，而是在提醒——事情仍要有人去做。”
生成文字中，应多出现类似"我今日站在这里，不是为了说些空话，而是为了……"、"我们不能让那些已经站起来的人，再倒下去"这种短句反复、强调现实责任与道德选择的表达；
用词可带有鲁迅的语感，如"诸君""呐喊""罢了""然而""我想"之类。
保证整体风格既现代白话，又显鲁迅式锋利、冷峻、理性批判。""",
        "赵树理": """
我希望你写一篇具有赵树理式风格的文字。

标题要求求如下：
- 必须生成一个标题，标题放在开头，独立一行。
- 标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。
- 标题应带有乡土气息和讽刺意味，像村里人说的俏皮话或民间俗语，可用双关、反讽或生活化比喻。
- 标题不宜过长，最好一句话或短语，如《谁家的锅糊了》《这买卖不亏》《要不是老张那张嘴》。
- 标题与正文的风格要统一，读来就能听出"赵树理式说书味"。
- 正文开头必须紧接标题生成一个呼语（如'同志们''各位朋友！'等），用于称呼听众。
  
风格要求如下：
- 语言质朴、俏皮、有讽刺意味，带浓厚乡土气息。
- 用词自然，不做作，可用"咱们""你要问我说""他那一伙""这话得好好想想"等日常口语。
- 句式短促通俗，可用民间比喻、对话穿插叙述。
- 整体有"说书式"的节奏感，语气平和、有观察力，体现民间智慧。
- 文字可带幽默与讽喻，但要冷静、克制。
- 内容上要讲一个具体的人或事，不空谈道理。
- 每一段都要有推进，不在同一句式上来回打转，避免机械重复。
- 每一段可有轻微转折或反思，像一个清醒的乡村叙述者慢慢讲理。
- 叙述者口吻要像村里一个明白人，既有点打趣，又不失公道。
- 可适当出现人物间的对话，像"老李说……""我就笑他：你这不是自找的吗？"这种自然插话，增强活气。
- 全篇最好像是"说理带故事"，故事里有人情味，理里带一点反讽的劲。
- 结尾要自然收束，像"话说到这儿也就明白了"那种收口，不要突兀或反复强调。
""",
        "侠客岛": """
我希望这篇文字具有"侠客岛式"风格。

标题要求:必须生成一个标题，标题单独成行，置于开头。标题不宜空洞或平铺，应让人"一看就像媒体评论标题"，既有理性，也有锋芒。标题与正文风格必须统一，不得割裂。标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。

语言上，应当稳健、凝练、带有理性克制的批评与分析气质；文风应兼具媒体的客观与评论的锋锐，体现出"冷静叙事 + 犀利观点"的融合。

务必保持我在提示词中指定的叙述者身份，不得擅自替换为"侠客岛""岛叔""评论员"等其他主体。

用词应体现，具备权威媒体评论的庄重感，同时不失亲切；避免空洞口号和套话，多用现实感、新闻语体、分析性句式。

语气上，应平实理智，不浮夸、不喊口号。可适度带有讽刺或反问，但要有分寸感，始终保持理性、冷静、逻辑清晰。

正文开头必须紧接标题生成一个呼语（如'同志们'等），用于称呼听众

文风要求：

句式以短句和中长句结合，节奏稳健、有呼吸感；  
描写注重事实、逻辑递进与背景铺陈，观点要自然生成于叙述之中；  
语气要克制而有力，结尾多以总结或警醒收束，形成自然的闭合感。

气质上要体现"有理有据、有温度、有锋芒"的评论者姿态，既有大局观，又有民间温度，传达出媒体理性与现实关怀并存的特质。

注意避免机械复述与句式雷同，应当在逻辑上自洽、在节奏上有层次感，结尾要自然收束而非突兀收尾。
""",
    }

    def _generate_report_with_style(self, state: State, style_role: str) -> str:
        """根据指定风格生成报告（内部辅助方法）"""
        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "生成最终报告")

        # 构建精简的 reporter 输入，只包含必要信息
        # 避免传入完整 state["messages"]（包含大量 central agent 调度信息）
        user_query = state.get("user_query", "")
        user_dst = state.get("user_dst", "")
        report_outline = state.get("report_outline", "用户未提供大纲")

        reporter_input = {
            "messages": [
                HumanMessage(
                    content=f"# Research Requirements\n\n## User Query\n\n{user_query}"
                )
            ],
            "locale": state.get("locale", "zh-CN"),
        }

        context = {
            "user_query": user_query,
            "task_description": task_description,
        }

        report = "报告生成失败: 未知错误"
        try:
            messages = apply_prompt_template(
                "reporter_xxqg", reporter_input, extra_context=context
            )

            # 添加用户约束、大纲和数据收集
            # data_collections = state.get("data_collections", [])
            # data_collections_str = "\n\n".join(data_collections)
            constraint = self.ROLE_CONSTRAINTS.get(style_role, "")

            # 检查是否存在原始报告（风格切换场景）
            original_report = state.get("original_report", "")
            reference_hint = ""
            if original_report:
                # 提取原始报告中的引用编号
                import re

                citations = re.findall(r"【(\d+)】", original_report)
                if citations:
                    unique_citations = sorted(set(citations), key=lambda x: int(x))
                    reference_hint = f"\n\n##引用保持要求\n\n原始报告使用了以下引用编号：{'、'.join(['【' + c + '】' for c in unique_citations])}。请在新风格的报告中尽量保持使用相同的引用来源，确保引用的完整性和一致性。"

            messages.append(
                HumanMessage(
                    content=f"{constraint}##User Query\n\n{user_query}\n\n##任务描述\n\n{task_description}\n\n##用户约束\n\n{user_dst}\n\n##报告大纲\n\n{report_outline}{reference_hint}"
                )
            )

            # 添加 observations
            observations = state.get("observations", [])
            for observation in observations:
                messages.append(
                    HumanMessage(
                        content=f"以下是检索智能体收集到的高质量信息: \n\n{observation}",
                        name="search_agent",
                    )
                )

            logger.debug(f"Reporter messages: {messages}")
            llm = get_llm_by_type(AGENT_LLM_MAP.get("reporter", "default"))
            response = llm.invoke(messages)
            report = response.content
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"报告Agent执行失败: {str(e)}")
            report = f"报告生成失败: {str(e)}"
        return report

    @timed_step("execute_xxqg_reporter")
    def execute_xxqg_reporter(self, state: State, config: RunnableConfig) -> Command:
        """
        执行报告Agent，负责结果整理与报告生成。
        使用 wait_stage 模式：首次进入生成报告后跳转到 human_feedback，
        从 human_feedback 返回后处理用户反馈。

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("报告Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get("task_description", "生成最终报告")

        # 直接从 state 获取风格（已在入口处提取并存储）
        current_style = state.get("current_style", "")

        wait_stage = state.get("wait_stage", "")
        if wait_stage != "reporter":
            # 首次进入：生成报告
            logger.info(f"使用风格 '{current_style}' 生成报告...")
            final_report = self._generate_report_with_style(state, current_style)

            # 记录到中枢Agent记忆栈
            memory_entry = MemoryStackEntry(
                timestamp=datetime.now().isoformat(),
                action="delegate",
                agent_type="reporter",
                content=f"报告任务: {task_description}，风格: {current_style}",
                result={"final_report": final_report},
            )
            self.central_agent.memory_stack.push(memory_entry)

            # 跳转到 human_feedback 节点等待用户反馈
            logger.info("报告生成完成，跳转到 human_feedback 节点等待用户反馈")
            return Command(
                update={
                    "final_report": final_report,
                    "original_report": final_report,  # 保存首次生成的报告作为参考
                    "current_style": current_style,
                    "wait_stage": "reporter",
                    "current_node": "reporter",
                },
                goto="human_feedback",
            )

        # 从 human_feedback 返回：处理用户反馈
        if wait_stage == "reporter":
            feedback = state.get("hitl_feedback", "")
            final_report = state.get("final_report", "")
            current_style = state.get("current_style", "")

            if feedback and str(feedback).upper().startswith("[CHANGED_STYLE]"):
                # 解析新风格，清空 wait_stage 后重新进入 reporter 节点生成报告
                # 提取风格名称：取第一个空格或换行之前的内容，避免客户端附带多余内容
                raw_style = str(feedback)[len("[CHANGED_STYLE]") :].strip()
                # 风格名称只取第一部分（空格、换行、[STYLE_ROLE] 之前的内容）
                new_style = raw_style.split()[0] if raw_style.split() else raw_style
                # 如果风格名称中包含 [STYLE_ROLE]，截断它
                if "[STYLE_ROLE]" in new_style:
                    new_style = new_style.split("[STYLE_ROLE]")[0]
                new_style = new_style.strip()
                logger.info(f"用户请求切换风格: {current_style} -> {new_style}")

                # 只更新 current_style，不再修改 user_query
                return Command(
                    update={
                        "current_style": new_style,
                        "wait_stage": "",  # 清空 wait_stage，下次进入时重新生成报告
                        "current_node": "reporter",
                    },
                    goto="reporter",
                )
            elif feedback and str(feedback).upper().startswith("[SKIP]"):
                # 用户跳过，正常结束
                logger.info("用户跳过风格切换，报告生成完成")
            elif feedback and str(feedback).upper().startswith("[END]"):
                # 用户跳过，正常结束
                logger.info("用户跳过风格切换，报告生成完成")
            else:
                # 其他反馈，正常结束
                logger.info(f"收到其他反馈: {feedback}，报告生成完成")

            logger.info("报告生成完成，返回中枢Agent")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="报告生成完成，返回中枢Agent", name="reporter"
                        )
                    ],
                    "final_report": final_report,
                    "current_node": "central_agent",
                    "wait_stage": "",
                    "memory_stack": self.central_agent.memory_stack.to_dict(),
                },
                goto="central_agent",
            )

    @timed_step("execute_sp_planner")
    def execute_sp_planner(self, state: State, config: RunnableConfig) -> Command:
        """
        执行任务拆解Agent，负责将复杂任务拆解为可管理的子任务

        Args:
            state: 当前系统状态
            config: 运行配置

        Returns:
            执行结果Command对象
        """
        logger.info("任务拆解Agent开始执行...")

        delegation_context = state.get("delegation_context", {})
        task_description = delegation_context.get(
            "task_description",
            state.get("user_query", "") + "\n将用户的任务拆解成2-5个子任务",
        )

        # 收集任务拆解所需上下文
        context = {
            "user_query": state.get("user_query", ""),
            "memory_history": [],  # self.central_agent.memory_stack.get_all(),
            "task_description": task_description,
        }

        # 生成任务拆解并处理异常
        replan_result = "任务拆解失败: 未知错误"
        try:
            messages = apply_prompt_template(
                "replanner", state, extra_context=context
            )  # 修复：参数顺序
            llm = get_llm_by_type(AGENT_LLM_MAP.get("replanner", "default"))
            response = llm.invoke(messages)
            replan_result = response.content
            replan_result = (
                replan_result.replace("```json", "").replace("```", "").strip()
            )

            logger.debug(f"任务拆解结果: {replan_result}")

            # 解析LLM返回的任务拆解结果
            import json

            try:
                response_json = json.loads(replan_result)
                if isinstance(response_json, list):
                    response_json = {"DAG": response_json}
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                response_json = {"DAG": [(input, input)]}
            if isinstance(response_json["DAG"], list):
                new_dag = []
                for item in response_json["DAG"]:
                    if isinstance(item, dict):
                        pairs = list(item.items())
                        new_dag.append(
                            (pairs[0][1], pairs[1][1])
                            if len(pairs) > 1
                            else (pairs[0][1], pairs[0][1])
                        )
                    elif isinstance(item, list) and len(item) > 1:
                        new_dag.append((item[0], item[1]))
                    else:
                        new_dag.append((item, item))
                response_json["DAG"] = new_dag

            from src.utils.graph_utils import Graph

            graph = Graph()
            graph.load_dag_from_json(response_json)
            sorted_nodes = graph.topological_sort()
            # Generate a unique ID for each input using a hash
            input_id = hash(input)
            # replan_result = {"id":input_id,"plans":[{node_id: graph.nodes[node_id].question} for node_id in sorted_nodes],"status":["uncomplete" for node_id in sorted_nodes]}
            replan_result = {
                "id": input_id,
                "plans": [
                    {node_id: graph.nodes[node_id].question} for node_id in sorted_nodes
                ],
            }
        except Exception as e:
            logger.error(f"任务拆解Agent执行失败: {str(e)}")
            replan_result = f"任务拆解失败: {str(e)}"

        # 记录到中枢Agent记忆栈
        memory_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="delegate",
            agent_type="replanner",
            content=f"任务拆解: {task_description}",
            result={"replan_result": replan_result},
        )
        self.central_agent.memory_stack.push(memory_entry)

        logger.info("任务拆解完成，返回中枢Agent")
        return Command(
            update={
                "messages": [
                    HumanMessage(content="任务拆解完成，返回中枢Agent", name="planner")
                ],
                "replan_result": replan_result,
                "current_node": "central_agent",
                "memory_stack": self.central_agent.memory_stack.to_dict(),
            },
            goto="central_agent",
        )

    @timed_step("execute_human_feedback")
    async def execute_human_feedback(
        self, state: State, config: RunnableConfig
    ) -> Command:
        stage = state.get("wait_stage", "perception")
        if stage == "perception":
            dst_question = state.get("dst_question", "")
            feedback = interrupt(
                "Please Fill the Question.[DST]" + dst_question + "[/DST]"
            )
            logger.info(f"用户反馈的DST问题: {feedback}. goto perception node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="perception",
            )
        elif stage == "outline":
            outline = state.get("report_outline", "")
            feedback = interrupt(
                "Please Confirm or Edit the Outline.[OUTLINE]" + outline + "[/OUTLINE]"
            )
            logger.info(f"用户反馈的大纲: {feedback}. goto outline node again.")
            return Command(
                update={
                    "hitl_feedback": feedback,
                    "current_node": "human_feedback",
                },
                goto="outline",
            )
        elif stage == "reporter":
            final_report = state.get("final_report", "")
            feedback = interrupt(
                "Report generated. You can change style or finish.[REPORT]"
                + final_report
                + "[/REPORT]"
            )
            logger.info(f"用户反馈: {feedback}")

            # 分类处理：内容修改走 central_agent，风格切换直接返回 reporter
            if feedback and str(feedback).upper().startswith("[CONTENT_MODIFY]"):
                # 复杂修改：入栈并跳转到 central_agent 决策
                modify_request = str(feedback)[len("[CONTENT_MODIFY]") :].strip()

                # 从 state 中恢复 memory_stack（因为 central_agent 每次请求都会重新创建）
                state_memory_stack = state.get("memory_stack")
                if state_memory_stack:
                    self.central_agent.memory_stack.load_from_dict(state_memory_stack)

                memory_entry = MemoryStackEntry(
                    timestamp=datetime.now().isoformat(),
                    action="human_feedback",
                    content=f"用户对报告的修改意见: {modify_request}",
                    result={
                        "feedback_type": "content_modify",
                        "request": modify_request,
                    },
                )
                self.central_agent.memory_stack.push(memory_entry)
                logger.info(f"内容修改请求入栈，跳转到 central_agent: {modify_request}")
                return Command(
                    update={
                        "hitl_feedback": feedback,
                        "current_node": "human_feedback",
                        "memory_stack": self.central_agent.memory_stack.to_dict(),
                        "wait_stage": "",  # 重置 wait_stage，以便 reporter 重新生成报告
                        "messages": [
                            HumanMessage(
                                content=f"用户对报告的修改意见: {modify_request}",
                                name="human_feedback",
                            )
                        ],
                    },
                    goto="central_agent",
                )
            else:
                # 简单修改（风格切换等）：直接返回 reporter 处理
                return Command(
                    update={
                        "hitl_feedback": feedback,
                        "current_node": "human_feedback",
                    },
                    goto="reporter",
                )

    @timed_step("execute_perception")
    async def execute_perception(self, state: State, config: RunnableConfig) -> Command:
        user_query = state.get("user_query", "")
        # check if the plan is auto accepted
        perception_llm = get_llm_by_type(AGENT_LLM_MAP.get("perception", "default"))
        wait_stage = state.get("wait_stage", "")
        if wait_stage != "perception":
            try:
                # messages = apply_prompt_template("perception", state) + [
                #     HumanMessage(f"##User Query\n\n{user_query}\n\n")
                # ]
                messages = apply_prompt_template("perception", state)

                # logger.debug("messages"+str(messages))
                response = perception_llm.invoke(messages)
                dst_question = response.content
                # logger.debug("dst_question"+str(dst_question))
                dst_question = repair_json_output(dst_question)
                logger.info(f"感知层完成，生成DST问题: {dst_question}")
                return Command(
                    update={
                        "dst_question": dst_question,
                        "wait_stage": "perception",
                        "current_node": "perception",
                    },
                    goto="human_feedback",
                )
            except Exception as e:
                logger.error(f"感知层执行失败: {str(e)}")

        if wait_stage == "perception":
            feedback = state.get("hitl_feedback", "")
            dst_question = state.get("dst_question", "")
            # if the feedback is not accepted, return the planner node
            if feedback and str(feedback).upper().startswith("[FILLED_QUESTION]"):
                messages = apply_prompt_template("perception", state) + [
                    HumanMessage(
                        f"##User Query\n\n{user_query}\n\n##希望用户回答的问题\n\n{dst_question}\n\n##用户回答的结果\n\n{feedback}\n\n"
                    )
                ]
                # logger.debug("messages"+str(messages))
                # exit()
                response = perception_llm.invoke(messages)
                summary = response.content
                logger.info(f"感知层完成，收集用户反馈: {summary}")

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"感知层完成，收集用户反馈: {summary}",
                                name="perception",
                            )
                        ],
                        "user_dst": summary,
                        "current_node": "perception",
                        "wait_stage": "",
                    },
                    goto="central_agent",#goto="outline",#SOP情况下应该是 goto 的中枢智能体
                )
            elif feedback and str(feedback).upper().startswith("[SKIP]"):
                logger.info("DST question is skipped by user.")
                messages = apply_prompt_template("perception", state) + [
                    HumanMessage(
                        f"##User Query\n\n{user_query}\n\n##希望用户回答的问题\n\n{dst_question}\n\n##用户跳过了回答，你可以按照自己的理解总结\n\n"
                    )
                ]
                response = perception_llm.invoke(messages)
                summary = response.content
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content="DST question is skipped by user.",
                                name="perception",
                            )
                        ],
                        "user_dst": summary,
                        "current_node": "perception",
                        "wait_stage": "",
                    },
                    goto="central_agent",#goto="outline",#SOP情况下应该是 goto 的中枢智能体
                )
            else:
                raise TypeError(f"Interrupt value of {feedback} is not supported.")

    @timed_step("execute_outline")
    async def execute_outline(self, state: State, config: RunnableConfig) -> Command:
        user_query = state.get("user_query", "")
        # check if the plan is auto accepted
        outline_llm = get_llm_by_type(AGENT_LLM_MAP.get("outline", "default"))
        wait_stage = state.get("wait_stage", "")
        if wait_stage != "outline":
            bg_investigation = search_docs_with_ref(
                user_query, top_k=5, config=config
            ).get("docs", [])
            user_dst = state.get("user_dst", "")
            try:
                messages = [
                    HumanMessage(
                        f"##用户原始问题\n\n{user_query}\n\n##用户补充需求\n\n{user_dst}\n\n##可能用到的相关数据\n\n{bg_investigation}\n\n"
                    )
                ] + apply_prompt_template("outline", state)
                response = outline_llm.invoke(messages)
                outline_response = response.content
                outline_response = repair_json_output(outline_response)
                if "[STYLE_ROLE]" in outline_response:
                    outline_response = outline_response.split("[STYLE_ROLE]")[0]
                logger.info(f"大纲生成完成: {outline_response}")
                return Command(
                    update={
                        "report_outline": outline_response,
                        "wait_stage": "outline",
                        "current_node": "outline",
                    },
                    goto="human_feedback",
                )
            except Exception as e:
                logger.error(f"大纲生成执行失败: {str(e)}")
        if wait_stage == "outline":
            feedback = state.get("hitl_feedback", "")
            # if the feedback is not accepted, return the planner node
            if feedback and str(feedback).upper().startswith("[CONFIRMED_OUTLINE]"):
                previous_outline = state.get("report_outline", "")
                outline_confirmed = feedback[len("[CONFIRMED_OUTLINE]") :].strip()

                #原先的outline中有形如【id】的引用标志，而确认后的outline不仅删除了所有引用标志，还修改了文字部分。我需要把原先的引用标志补全回来：如果原先这个位置有引用标志而现在这个位置附近的文字也没被修改，那么补充回来；如果被修改了就不用补充了
                def repair_outline_citations(previous_outline, outline_confirmed):
                    """
                    把 previous_outline 中的【id】引用标志，尽可能无损地回补到 outline_confirmed 中。
                    规则：
                    1. 如果原文附近文字未被改动，则把【id】补回；
                    2. 若文字被改写，则不再补回；
                    3. 若 confirmed 中已自带引用，则保留其引用，不再叠加。
                    """
                    # 提取 previous 中的引用映射：{纯文本: 【id】}
                    prev_map = {}
                    for m in re.finditer(r'(.*?)(【\d+】)', previous_outline):
                        text_snippet = m.group(1).strip()
                        citation = m.group(2)
                        if text_snippet:
                            prev_map[text_snippet] = citation
                    logger.debug(f"Previous outline citation map: {prev_map}")
                    # 按段落逐句扫描 confirmed，尝试回补
                    def replace_func(match):
                        sentence = match.group(1)
                        # 若句子已含引用，跳过
                        if re.search(r'【\d+】', sentence):
                            return match.group(0)
                        # 寻找最近似原文片段
                        best_key = None
                        best_ratio = 0.6   # 阈值，可微调
                        for key in prev_map:
                            # 简单相似：包含关系即可
                            if key in sentence or sentence in key:
                                best_key = key
                                best_ratio = 1.0
                                break
                        if best_key:
                            return sentence + prev_map[best_key]
                        return match.group(0)

                    # 以句号为界，逐句处理
                    confirmed_repaired = re.sub(
                        r'([^。！？\n]+[。！？])',
                        replace_func,
                        outline_confirmed
                    )
                    return confirmed_repaired

                if re.search(r'【\d+】', outline_confirmed):
                    # 如果确认后的大纲中已经有引用标志，就不需要回补了
                    logger.debug("确认后的大纲中已有引用标志，无需回补")
                    pass
                else:
                    outline_confirmed = repair_outline_citations(previous_outline, outline_confirmed)
                    
                logger.info(f"大纲确认: {outline_confirmed}")

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"大纲确认: {outline_confirmed}", name="outline"
                            )
                        ],
                        "report_outline": outline_confirmed,
                        "current_node": "outline",
                        "wait_stage": "",
                    },
                    goto="central_agent",
                )
            elif feedback and str(feedback).upper().startswith("[SKIP]"):
                outline_confirmed = feedback[len("[SKIP]") :].strip()
                logger.info(f"大纲确认: {outline_confirmed}")

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"大纲确认: {outline_confirmed}", name="outline"
                            )
                        ],
                        "report_outline": outline_confirmed,
                        "current_node": "outline",
                        "wait_stage": "",
                    },
                    goto="central_agent",
                )
            else:
                raise TypeError(f"Interrupt value of {feedback} is not supported.")
