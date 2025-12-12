# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
from src.utils.logger import logger
import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.CommonReactAgent import CommonReactAgent
from src.agents.CoderAgent import CoderAgent
from src.agents.ResearcherAgent_SP import ResearcherAgentSP
from src.agents.ResearcherAgent import ResearcherAgent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
    search_docs_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine
from datetime import datetime
from src.utils.statistics import global_statistics


@tool
def handoff_to_planner(
    task_title: Annotated[str, "The title of the task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner"]]:
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    query = state["messages"][-1].content

    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        background_investigation_results = None
        if isinstance(searched_content, list):
            background_investigation_results = [
                {"title": elem["title"], "content": elem["content"]}
                for elem in searched_content
            ]
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)

    # background_investigation_results = search_docs_tool.invoke(query)
    # background_investigation_results = []
    return Command(
        update={
            "background_investigation_results": json.dumps(
                background_investigation_results, ensure_ascii=False
            )
        },
        goto="planner",
    )


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    if (
        plan_iterations == 0
        and state.get("enable_background_investigation")
        and state.get("background_investigation_results")
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    if AGENT_LLM_MAP["planner"] == "basic":
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
            Plan,
            method="json_mode",
        )
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if AGENT_LLM_MAP["planner"] == "basic":
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        logger.info(f"Planner: {full_response}")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
            },
            goto="reporter",
        )
    logger.info(f"Planner: {full_response}")
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
        },
        goto="human_feedback",
    )


def sp_planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    if (
        plan_iterations == 0
        and state.get("enable_background_investigation")
        and state.get("background_investigation_results")
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    if AGENT_LLM_MAP["planner"] == "basic":
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
            Plan,
            method="json_mode",
        )
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if AGENT_LLM_MAP["planner"] == "basic":
        response = llm.invoke(messages)
        full_response = response.model_dump_json(indent=4, exclude_none=True)
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
        plan_iterations += 1
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        logger.info(f"Planner: {full_response}")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
            },
            goto="reporter",
        )
    return Command(
        update={
            "current_plan": Plan.model_validate(curr_plan),
            "plan_iterations": plan_iterations,
            "locale": curr_plan["locale"],
        },
        goto="research_team",
    )


def human_feedback_node(
    state,
) -> Command[Literal["planner", "research_team", "reporter", "__end__"]]:
    current_plan = state.get("current_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        feedback = interrupt("Please Review the Plan.")

        # if the feedback is not accepted, return the planner node
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                },
                goto="planner",
            )
        elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        current_plan = repair_json_output(current_plan)
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(current_plan)
        if new_plan["has_enough_context"]:
            goto = "reporter"
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    return Command(
        update={
            "current_plan": Plan.model_validate(new_plan),
            "plan_iterations": plan_iterations,
            "locale": new_plan["locale"],
        },
        goto=goto,
    )


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified
    logger.debug(f"Coordinator response: {response}")
    if len(response.tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_locale := tool_call.get("args", {}).get("locale"):
                    locale = tool_locale
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )
        logger.debug(f"Coordinator response: {response}")

    return Command(
        update={"locale": locale, "resources": configurable.resources},
        goto=goto,
    )


def coordinator_xxqg_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified
    logger.debug(f"Coordinator response: {response}")
    if len(response.tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_locale := tool_call.get("args", {}).get("locale"):
                    locale = tool_locale
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )
        logger.debug(f"Coordinator response: {response}")

    return Command(
        update={"locale": locale, "resources": configurable.resources},
        goto=goto,
    )


def reporter_node(state: State):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("reporter", input_)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )
    data_collections = state.get("data_collections", [])
    for data_collection in data_collections:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are data collected in previous tasks:\n\n{data_collection}",
                name="observation",
            )
        )
    data_collections = state.get("data_collections", [])
    for data_collection in data_collections:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are data collected in previous tasks:\n\n{data_collection}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}


def reporter_xxqg_node(state: State):
    """
    Reporter node that write a final report.
    生成文档后进入循环，等待用户风格切换请求 [CHANGED_STYLE]xxx，
    收到后用新风格重新生成文档；收到 [SKIP] 或其他反馈则结束。
    """
    logger.info("Reporter write final report")
    current_plan = state.get("current_plan")
    user_query = state.get("user_query")

    # 读取所有二级文体类别
    with open("src/prompts/xxqg_rule_demo_lib.json", "r") as f:
        rule_demo_lib = json.load(f)
    # 通过字符串匹配的方式简单检索文体类别
    rule, demos = None, None
    for type1, type2_dict in rule_demo_lib.items():
        for type2, type2_info_dict in type2_dict.items():
            type3_dict = type2_info_dict.get("tags", {})
            rule = type2_info_dict.get("ori_text", None)
            for type3, type3_info_dict in type3_dict.items():
                if type3 in user_query:
                    demos = type3_info_dict["few_shot"]
                    break
            if demos:
                break
        if demos:
            break
    demos_str = "\n\n".join(
        [f"### Demonstration {i+1}\n\n{d}" for i, d in enumerate(demos)]
    )

    # 风格约束定义
    role_constraints = {
        "鲁迅": """我希望生成的文字具备鲁迅式风格，语言尖锐、冷峻、带讽刺，但保持自然白话表达，可以使用少量文言。
标题要求：文章必须包含一个标题，标题应简短有力、富隐喻或冷讽意味，可为一句或两句并列句。标题风格应与正文一致，具有鲁迅式的锋芒与余味，不得中性或平淡。标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。
重要禁止项：文中不要有“鲁迅”这个词，严禁在生成的文本中出现任何提及或引用“鲁迅”、“鲁迅先生”、“鲁迅笔下”、“他的作品”、“他的笔下的人物”等字眼的语句。文本风格应是直接的、沉浸式的鲁迅式表达，而非对鲁迅风格的引用或评论。此禁令在任何标题或正文中均适用，绝不可出现任何直接或间接的提及。
风格应用强制要求：请确保文章的每一个自然段，乃至每一句的行文，都贯彻鲁迅式用词、句式和节奏。特别是在文章的中间部分，必须维持并强化这种尖锐、冷峻的语感。全篇保持一致的鲁迅式节奏与语气，特别在中段保持最高的语言张力与思想锋芒。
正文开头必须紧接标题生成一个呼语（如‘诸君！’），用于称呼听众。

句式与节奏：
采用短句、并列句和重复句（如“不是为了……，而是为了……”，“我们不能……再……”，“然而……”）；
逻辑紧凑，节奏鲜明，读来有推力；
可以用反问、讽刺、比喻、小见大，表达社会或人性的荒谬；
偶尔自嘲或旁观者冷笑，保持“孤独知识分子”的视角。
可出现明显的鲁迅式呼喊与强调，如“我要说的是……”，“我们不能……”，或“人类的悲欢并不相通”式的冷峻洞察。
情感与气质：
理性中带愤怒与冷漠，情感压抑而清醒；
既有悲悯，也有讽刺与愤世嫉俗感；
文字有“铁屋呐喊”的张力，让读者感受到现实的紧迫与不容回避。
目标效果：
生成文字中，应多出现类似“我今日站在这里，不是为了说些空话，而是为了……”、“我们不能让那些已经站起来的人，再倒下去”这种短句反复、强调现实责任与道德选择的表达；
用词可带有鲁迅的语感，如“诸君”“呐喊”“罢了”“然而”“我想”之类。
保证整体风格既现代白话，又显鲁迅式锋利、冷峻、理性批判。""",
        "赵树理": """
我希望你写一篇具有赵树理式风格的文字。

标题要求求如下：
- 必须生成一个标题，标题放在开头，独立一行。
- 标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。
- 标题应带有乡土气息和讽刺意味，像村里人说的俏皮话或民间俗语，可用双关、反讽或生活化比喻。
- 标题不宜过长，最好一句话或短语，如《谁家的锅糊了》《这买卖不亏》《要不是老张那张嘴》。
- 标题与正文的风格要统一，读来就能听出“赵树理式说书味”。
- 正文开头必须紧接标题生成一个呼语（如‘同志们’‘各位朋友！’等），用于称呼听众。
  
风格要求如下：
- 语言质朴、俏皮、有讽刺意味，带浓厚乡土气息。
- 用词自然，不做作，可用“咱们”“你要问我说”“他那一伙”“这话得好好想想”等日常口语。
- 句式短促通俗，可用民间比喻、对话穿插叙述。
- 整体有“说书式”的节奏感，语气平和、有观察力，体现民间智慧。
- 文字可带幽默与讽喻，但要冷静、克制。
- 内容上要讲一个具体的人或事，不空谈道理。
- 每一段都要有推进，不在同一句式上来回打转，避免机械重复。
- 每一段可有轻微转折或反思，像一个清醒的乡村叙述者慢慢讲理。
- 叙述者口吻要像村里一个明白人，既有点打趣，又不失公道。
- 可适当出现人物间的对话，像“老李说……”“我就笑他：你这不是自找的吗？”这种自然插话，增强活气。
- 全篇最好像是“说理带故事”，故事里有人情味，理里带一点反讽的劲。
- 结尾要自然收束，像“话说到这儿也就明白了”那种收口，不要突兀或反复强调。
""",
        "侠客岛": """
我希望这篇文字具有“侠客岛式”风格。

标题要求:必须生成一个标题，标题单独成行，置于开头。标题不宜空洞或平铺，应让人“一看就像媒体评论标题”，既有理性，也有锋芒。标题与正文风格必须统一，不得割裂。标题必须使用 Markdown 一级标题格式呈现（即 # 标题），不得使用书名号、引号、括号等符号。

语言上，应当稳健、凝练、带有理性克制的批评与分析气质；文风应兼具媒体的客观与评论的锋锐，体现出“冷静叙事 + 犀利观点”的融合。

务必保持我在提示词中指定的叙述者身份，不得擅自替换为“侠客岛”“岛叔”“评论员”等其他主体。

用词应体现，具备权威媒体评论的庄重感，同时不失亲切；避免空洞口号和套话，多用现实感、新闻语体、分析性句式。

语气上，应平实理智，不浮夸、不喊口号。可适度带有讽刺或反问，但要有分寸感，始终保持理性、冷静、逻辑清晰。

正文开头必须紧接标题生成一个呼语（如‘同志们’等），用于称呼听众

文风要求：

句式以短句和中长句结合，节奏稳健、有呼吸感；  
描写注重事实、逻辑递进与背景铺陈，观点要自然生成于叙述之中；  
语气要克制而有力，结尾多以总结或警醒收束，形成自然的闭合感。

气质上要体现“有理有据、有温度、有锋芒”的评论者姿态，既有大局观，又有民间温度，传达出媒体理性与现实关怀并存的特质。

注意避免机械复述与句式雷同，应当在逻辑上自洽、在节奏上有层次感，结尾要自然收束而非突兀收尾。
""",
    }

    # 从 user_query 中解析初始风格
    if "[STYLE_ROLE]" in user_query:
        current_style = user_query.split("[STYLE_ROLE]")[-1]
    else:
        current_style = ""

    def _generate_report_with_style(style_role: str) -> str:
        """根据指定风格生成报告"""
        input_ = {
            "messages": [
                HumanMessage(
                    f"# Research Requirements\n\n##User Query\n\n{demos_str}\n\n请严格仿照以上示例。{user_query}\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
                )
            ],
            "locale": state.get("locale", "en-US"),
        }
        input_["demonstrations"] = demos
        input_["rule"] = rule

        # 应用对应文体的prompt模板
        invoke_messages = apply_prompt_template(f"reporter_xxqg_rule_demo", input_)
        observations = state.get("observations", [])

        # Add a reminder about the new report format, citation style, and table usage
        invoke_messages.append(
            HumanMessage(
                content="IMPORTANT: Structure your report according to the format in the prompt.",
                name="system",
            )
        )

        for observation in observations:
            invoke_messages.append(
                HumanMessage(
                    content=f"Below are some observations for the research task:\n\n{observation}",
                    name="observation",
                )
            )
        data_collections = state.get("data_collections", [])
        for data_collection in data_collections:
            invoke_messages.append(
                HumanMessage(
                    content=f"Below are data collected in previous tasks:\n\n{data_collection}",
                    name="observation",
                )
            )

        # 添加风格约束
        constraint = role_constraints.get(style_role, "")
        if constraint:
            invoke_messages.append(
                HumanMessage(
                    content=f"风格要求：\n{constraint}",
                    name="style_constraint",
                )
            )

        logger.debug(f"Current invoke messages: {invoke_messages}")
        response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
        return response.content

    # 风格切换循环
    while True:
        logger.info(f"使用风格 '{current_style}' 生成报告...")
        response_content = _generate_report_with_style(current_style)
        logger.info(f"reporter response: {response_content}")

        # 中断等待用户反馈（风格切换或结束）
        feedback = interrupt(
            "Report generated. You can change style or finish.[REPORT]"
            + response_content
            + "[/REPORT]"
        )

        if feedback and str(feedback).upper().startswith("[CHANGED_STYLE]"):
            # 解析新风格，继续循环
            new_style = str(feedback)[len("[CHANGED_STYLE]") :].strip()
            logger.info(f"用户请求切换风格: {current_style} -> {new_style}")
            current_style = new_style
            continue
        elif feedback and str(feedback).upper().startswith("[SKIP]"):
            # 用户跳过，正常结束
            logger.info("用户跳过风格切换，报告生成完成")
            break
        else:
            # 其他反馈，正常结束
            logger.info(f"收到其他反馈: {feedback}，报告生成完成")
            break

    return {"final_report": response_content}


def research_team_node(
    state: State,
) -> Command[Literal["planner", "researcher", "coder"]]:
    """Research team node that collaborates on tasks."""
    logger.info("Research team is collaborating on tasks.")
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return Command(goto="planner")
    if all(step.execution_res for step in current_plan.steps):
        return Command(goto="planner")
    for step in current_plan.steps:
        if not step.execution_res:
            break
    if step.step_type and step.step_type == StepType.RESEARCH:
        return Command(goto="researcher")
    if step.step_type and step.step_type == StepType.PROCESSING:
        return Command(goto="coder")
    return Command(goto="planner")


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    current_plan = state.get("current_plan")
    observations = state.get("observations", [])

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
        else:
            completed_steps.append(step)

    if not current_step:
        logger.warning("No unexecuted step found")
        return Command(goto="research_team")

    logger.info(f"Executing step: {current_step.title}, agent: {agent_name}")

    # Format completed steps information
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Existing Research Findings\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Existing Finding {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"{completed_steps_info}# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        # agent_input["messages"].append(
        #     HumanMessage(
        #         content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
        #         name="system",
        #     )
        # )
        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title]\n\n- [Another Source]",
                name="system",
            )
        )

    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.info(f"Agent input: {agent_input}")
    result = await agent.ainvoke(
        input=agent_input, config={"recursion_limit": recursion_limit}
    )

    # Process the result
    response_content = result["messages"][-1].content
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Update the step with the execution result
    current_step.execution_res = response_content
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": observations + [response_content],
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

    # Create and execute agent with MCP tools if available
    if mcp_servers:
        async with MultiServerMCPClient(mcp_servers) as client:
            loaded_tools = default_tools[:]
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    tool.description = (
                        f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                    )
                    loaded_tools.append(tool)
            agent = CommonReactAgent(
                agent_name=agent_type, tools=loaded_tools, system_prompt=agent_type
            )
            return await _execute_agent_step(state, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        agent = CommonReactAgent(
            agent_name=agent_type, tools=default_tools, system_prompt=agent_type
        )
        return await _execute_agent_step(state, agent, agent_type)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")

    configurable = Configuration.from_runnable_config(config)
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.insert(0, retriever_tool)

    # tools = [search_docs_tool]
    logger.info(f"Researcher tools: {tools}")
    research_agent = ResearcherAgent(
        config=config, agent_type="researcher", default_tools=tools
    )
    return await research_agent.execute_agent_step(state)


async def researcher_sp_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")

    configurable = Configuration.from_runnable_config(config)
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.insert(0, retriever_tool)

    # tools = [search_docs_tool]
    logger.info(f"Researcher tools: {tools}")
    research_agent = ResearcherAgentSP(
        config=config, agent_type="researcher", default_tools=tools
    )
    return await research_agent.execute_agent_step(state)


async def researcher_xxqg_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")

    tools = [search_docs_tool]
    logger.info(f"Researcher tools: {tools}")
    research_agent = ResearcherAgent(
        config=config, agent_type="researcher_xxqg", default_tools=tools
    )
    return await research_agent.execute_agent_step(state)


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    code_agent = CoderAgent(
        config=config, agent_type="coder", default_tools=[python_repl_tool]
    )
    return await code_agent.execute_agent_step(state)


def speech_node(state: State):
    """Speech node that write a governor speech."""
    logger.info("Speech write a speech")
    current_plan = state.get("current_plan")
    user_query = state.get("user_query")
    input_ = {
        "messages": [
            HumanMessage(
                f"# 讲话稿生成要求\n\n##用户完整需求\n\n{user_query}\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("speech_zh", input_)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="注意: 根据prompt中的约束生成讲话稿",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"以下是之前的检索信息和分析后的结果:\n\n{observation}",
                name="observation",
            )
        )
    data_collections = state.get("data_collections", [])
    for data_collection in data_collections:
        invoke_messages.append(
            HumanMessage(
                content=f"以下是之前的完整检索信息:\n\n{data_collection}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}


def zip_data(state: State):
    final_report = state.get("final_report")
    user_query = state.get("user_query")
    plan = state.get("current_plan")
    # Ensure the reports directory exists
    os.makedirs("./reports", exist_ok=True)

    # Prepare data to save
    data = {
        "user_query": user_query,
        "plan": plan.model_dump() if hasattr(plan, "model_dump") else str(plan),
        "final_report": final_report,
        "statistics": global_statistics.get_statistics(),
    }

    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./reports/report_{timestamp}.json"

    # Save data as JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
