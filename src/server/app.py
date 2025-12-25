# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import base64
import json
from math import log

from sympy import im
from src.utils.logger import logger
import os

from typing import Annotated, List, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage
from langgraph.types import Command

from src.config.tools import SELECTED_RAG_PROVIDER
from src.graph.builder import build_graph_with_memory
from src.graph.builder import get_graph_by_format
from src.podcast.graph.builder import build_graph as build_podcast_graph
from src.ppt.graph.builder import build_graph as build_ppt_graph
from src.prose.graph.builder import build_graph as build_prose_graph
from src.rag.builder import build_retriever
from src.rag.retriever import Resource
from src.server.chat_request import (
    ChatMessage,
    ChatRequest,
    GeneratePodcastRequest,
    GeneratePPTRequest,
    GenerateProseRequest,
    TTSRequest,
)
from src.server.mcp_request import MCPServerMetadataRequest, MCPServerMetadataResponse
from src.server.mcp_utils import load_mcp_tools
from src.server.rag_request import (
    RAGConfigResponse,
    RAGResourceRequest,
    RAGResourcesResponse,
)
from src.tools import VolcengineTTS
from src.utils.reference_utils import global_reference_map

app = FastAPI(
    title="DeerFlow API",
    description="API for Deer",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
from langgraph.store.memory import InMemoryStore

in_memory_store = InMemoryStore()
graph = build_graph_with_memory()


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    if thread_id == "__default__":
        thread_id = str(uuid4())
    return StreamingResponse(
        _astream_workflow_generator(
            request.model_dump()["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings,
            request.enable_background_investigation,
            request.knowledge_base_name,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator(
    messages: List[ChatMessage],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation,
    knowledge_base_name,
):
    input_ = {
        "messages": messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
        "user_query": messages[-1]["content"] if messages else "",
    }
    if not auto_accepted_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}]"
        # add the last message to the resume message
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        input_ = Command(resume=resume_msg)
    async for agent, _, event_data in graph.astream(
        input_,
        config={
            "thread_id": thread_id,
            "resources": resources,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "max_search_results": max_search_results,
            "mcp_settings": mcp_settings,
            "knowledge_base_name": knowledge_base_name,
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        if isinstance(event_data, dict):
            if "__interrupt__" in event_data:
                yield _make_event(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "id": event_data["__interrupt__"][0].ns[0],
                        "role": "assistant",
                        "content": event_data["__interrupt__"][0].value,
                        "finish_reason": "interrupt",
                        "options": [
                            {"text": "Edit plan", "value": "edit_plan"},
                            {"text": "Start research", "value": "accepted"},
                        ],
                    },
                )
            continue
        message_chunk, message_metadata = cast(
            tuple[BaseMessage, dict[str, any]], event_data
        )
        event_stream_message: dict[str, any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0],
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(message_chunk, ToolMessage):
            # Tool Message - Return the result of the tool call
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif isinstance(message_chunk, AIMessageChunk):
            # AI Message - Raw message tokens
            if message_chunk.tool_calls:
                # AI Message - Tool Call
                event_stream_message["tool_calls"] = message_chunk.tool_calls
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_calls", event_stream_message)
            elif message_chunk.tool_call_chunks:
                # AI Message - Tool Call Chunks
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_call_chunks", event_stream_message)
            else:
                # AI Message - Raw message tokens
                yield _make_event("message_chunk", event_stream_message)


@app.post("/api/chat/xxqg_stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    if thread_id == "__default__":
        thread_id = str(uuid4())
    return StreamingResponse(
        _astream_workflow_generator_xxqg(
            request.model_dump()["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings,
            request.enable_background_investigation,
            request.knowledge_base_name,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator_xxqg(
    messages: List[ChatMessage],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation,
    knowledge_base_name,
):
    input_ = {
        "messages": messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
        "user_query": messages[-1]["content"] if messages else "",
        "data_collections": [],
    }
    if not auto_accepted_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}]"
        # add the last message to the resume message
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        input_ = Command(resume=resume_msg)
    async for agent, _, event_data in get_graph_by_format("xxqg").astream(
        input_,
        config={
            "thread_id": thread_id,
            "resources": resources,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "max_search_results": max_search_results,
            "mcp_settings": mcp_settings,
            "knowledge_base_name": knowledge_base_name,
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        if isinstance(event_data, dict):
            if "__interrupt__" in event_data:
                yield _make_event(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "id": event_data["__interrupt__"][0].ns[0],
                        "role": "assistant",
                        "content": event_data["__interrupt__"][0].value,
                        "finish_reason": "interrupt",
                        "options": [
                            {"text": "Edit plan", "value": "edit_plan"},
                            {"text": "Start research", "value": "accepted"},
                        ],
                    },
                )
            continue
        message_chunk, message_metadata = cast(
            tuple[BaseMessage, dict[str, any]], event_data
        )
        event_stream_message: dict[str, any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0],
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(message_chunk, ToolMessage):
            # Tool Message - Return the result of the tool call
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif isinstance(message_chunk, AIMessageChunk):
            # AI Message - Raw message tokens
            if message_chunk.tool_calls:
                # AI Message - Tool Call
                event_stream_message["tool_calls"] = message_chunk.tool_calls
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_calls", event_stream_message)
            elif message_chunk.tool_call_chunks:
                # AI Message - Tool Call Chunks
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_call_chunks", event_stream_message)
            else:
                # AI Message - Raw message tokens
                yield _make_event("message_chunk", event_stream_message)


@app.post("/api/chat/sp_stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    if thread_id == "__default__":
        thread_id = str(uuid4())
    logger.info(f"request param details: {request}")
    return StreamingResponse(
        _astream_workflow_generator_sp(
            request.model_dump()["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings,
            request.enable_background_investigation,
            request.graph_format,
            request.knowledge_base_name,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator_sp(
    messages: List[ChatMessage],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation,
    graph_format: str = "sp",
    knowledge_base_name="学习强国",
):
    # 从 user_query 中提取 style_role 并清理
    raw_user_query = messages[-1]["content"] if messages else ""
    style_role = ""
    clean_user_query = raw_user_query
    if "[STYLE_ROLE]" in raw_user_query:
        parts = raw_user_query.split("[STYLE_ROLE]")
        clean_user_query = parts[0].strip()
        style_role = parts[-1].strip()

    # 同时清理 messages 中的 [STYLE_ROLE] 后缀
    clean_messages = []
    for msg in messages:
        clean_msg = msg.copy()
        if "[STYLE_ROLE]" in clean_msg.get("content", ""):
            clean_msg["content"] = clean_msg["content"].split("[STYLE_ROLE]")[0].strip()
        clean_messages.append(clean_msg)

    input_ = {
        "messages": clean_messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
        "user_query": clean_user_query,
        "current_style": style_role,
        "data_collections": [],
    }
    if not auto_accepted_plan and interrupt_feedback:
        if interrupt_feedback.startswith("["):
            resume_msg = interrupt_feedback
        else:
            resume_msg = f"[{interrupt_feedback}]"
        # add the last message to the resume message (使用清理后的内容)
        if clean_messages:
            resume_msg += f" {clean_messages[-1]['content']}"
        input_ = Command(resume=resume_msg)

    from src.graph.sp_nodes import init_agents

    init_agents(graph_format)
    if graph_format == "sp_xxqg":
        graph = get_graph_by_format(graph_format, with_memory=True)
    else:
        graph = get_graph_by_format(graph_format, with_memory=False)

    last_known_agent = None
    async for agent, _, event_data in graph.astream(
        input_,
        config={
            "thread_id": thread_id,
            "resources": resources,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "max_search_results": max_search_results,
            "mcp_settings": mcp_settings,
            "knowledge_base_name": knowledge_base_name,
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        # 返回当前节点状态
        if agent and agent != last_known_agent:
            last_known_agent = agent
            current_node_state = {
                "thread_id": thread_id,
                "current_node": agent,  # 当前节点名称
                "status": "processing",  # 节点状态
            }
            yield _make_event("node_status", current_node_state)

        if isinstance(event_data, dict):
            logger.debug(f"Event data: {event_data}")
        else:
            logger.debug(f"Event data type: {type(event_data)}")
        if isinstance(event_data, dict):
            if "__ref_map__" in event_data:
                ref_map = event_data["__ref_map__"][0].value
                yield _make_event(
                    "ref_map",
                    {
                        "thread_id": thread_id,
                        "ref_map": ref_map,
                    },
                )
            if "__interrupt__" in event_data:
                data_value = event_data["__interrupt__"][0].value
                if "[DST]" in data_value:
                    dst_question = data_value.split("[DST]")[-1].split("[/DST]")[0]
                    yield _make_event(
                        "interrupt",
                        {
                            "thread_id": thread_id,
                            "id": event_data["__interrupt__"][0].ns[0],
                            "role": "assistant",
                            "content": "Please Fill the Question",
                            "finish_reason": "interrupt",
                            "question": dst_question,
                        },
                    )
                elif "[OUTLINE]" in data_value:
                    outline = data_value.split("[OUTLINE]")[-1].split("[/OUTLINE]")[0]
                    yield _make_event(
                        "interrupt",
                        {
                            "thread_id": thread_id,
                            "id": event_data["__interrupt__"][0].ns[0],
                            "role": "assistant",
                            "content": "Please Confirm or Edit the Outline",
                            "finish_reason": "interrupt",
                            "outline": outline,
                        },
                    )
                elif "[REPORT]" in data_value:
                    # 报告生成完成，支持风格切换
                    report = data_value.split("[REPORT]")[-1].split("[/REPORT]")[0]
                    yield _make_event(
                        "interrupt",
                        {
                            "thread_id": thread_id,
                            "id": event_data["__interrupt__"][0].ns[0],
                            "role": "assistant",
                            "content": f"[REPORT]{report}[/REPORT]",
                            "finish_reason": "interrupt",
                        },
                    )
            elif "tools" in event_data:
                toolMessage = event_data["tools"]["messages"][0]
                yield _make_event(
                    "tool_call_result",
                    {
                        "thread_id": thread_id,
                        "role": "assistant",
                        "agent": agent[0].split(":")[0],
                        "content": toolMessage.content,
                        "id": toolMessage.id,
                        "tool_name": toolMessage.name,
                        "tool_call_id": toolMessage.tool_call_id,
                    },
                )
            continue
        message_chunk, message_metadata = cast(
            tuple[BaseMessage, dict[str, any]], event_data
        )
        
        event_stream_message: dict[str, any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0],
            "action_name": getattr(message_chunk, "name", None),
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(message_chunk, ToolMessage):
            # Tool Message - Return the result of the tool call
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif isinstance(message_chunk, AIMessageChunk):
            # AI Message - Raw message tokens
            if message_chunk.tool_calls:
                # AI Message - Tool Call
                event_stream_message["tool_calls"] = message_chunk.tool_calls
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_calls", event_stream_message)
            elif message_chunk.tool_call_chunks:
                # AI Message - Tool Call Chunks
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_call_chunks", event_stream_message)
            else:
                # AI Message - Raw message tokens
                yield _make_event("message_chunk", event_stream_message)
        else:
            #把 outline 给跳过了，紧急处理，后续需要修改
            logger.info(f"action的数据结构打印{event_stream_message}")
            if event_stream_message["agent"]=="outline" or event_stream_message["agent"]=="researcher":
                logger.info(f"前端内容筛选，不要 outline 或 researcher{event_stream_message}")
                continue
            yield _make_event("agent_action", event_stream_message)


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/references/{thread_id}")
async def get_references(thread_id: str):
    """Get the references for a given thread ID."""
    logger.info(f"Received request for references with thread_id: {thread_id}")
    try:
        references = global_reference_map.get_session_ref_map(thread_id)
        return {"thread_id": thread_id, "references": references}
    except Exception as e:
        logger.exception(f"Error in get_references endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using volcengine TTS API."""
    try:
        app_id = os.getenv("VOLCENGINE_TTS_APPID", "")
        if not app_id:
            raise HTTPException(
                status_code=400, detail="VOLCENGINE_TTS_APPID is not set"
            )
        access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN", "")
        if not access_token:
            raise HTTPException(
                status_code=400, detail="VOLCENGINE_TTS_ACCESS_TOKEN is not set"
            )
        cluster = os.getenv("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        voice_type = os.getenv("VOLCENGINE_TTS_VOICE_TYPE", "BV700_V2_streaming")

        tts_client = VolcengineTTS(
            appid=app_id,
            access_token=access_token,
            cluster=cluster,
            voice_type=voice_type,
        )
        # Call the TTS API
        result = tts_client.text_to_speech(
            text=request.text[:1024],
            encoding=request.encoding,
            speed_ratio=request.speed_ratio,
            volume_ratio=request.volume_ratio,
            pitch_ratio=request.pitch_ratio,
            text_type=request.text_type,
            with_frontend=request.with_frontend,
            frontend_type=request.frontend_type,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=str(result["error"]))

        # Decode the base64 audio data
        audio_data = base64.b64decode(result["audio_data"])

        # Return the audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{request.encoding}",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=tts_output.{request.encoding}"
                )
            },
        )
    except Exception as e:
        logger.exception(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/podcast/generate")
async def generate_podcast(request: GeneratePodcastRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_podcast_graph()
        final_state = workflow.invoke({"input": report_content})
        audio_bytes = final_state["output"]
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        logger.exception(f"Error occurred during podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ppt/generate")
async def generate_ppt(request: GeneratePPTRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_ppt_graph()
        final_state = workflow.invoke({"input": report_content})
        generated_file_path = final_state["generated_file_path"]
        with open(generated_file_path, "rb") as f:
            ppt_bytes = f.read()
        return Response(
            content=ppt_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logger.exception(f"Error occurred during ppt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prose/generate")
async def generate_prose(request: GenerateProseRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Generating prose for prompt: {sanitized_prompt}")
        workflow = build_prose_graph()
        events = workflow.astream(
            {
                "content": request.prompt,
                "option": request.option,
                "command": request.command,
            },
            stream_mode="messages",
            subgraphs=True,
        )
        return StreamingResponse(
            (f"data: {event[0].content}\n\n" async for _, event in events),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error occurred during prose generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/server/metadata", response_model=MCPServerMetadataResponse)
async def mcp_server_metadata(request: MCPServerMetadataRequest):
    """Get information about an MCP server."""
    try:
        # Set default timeout with a longer value for this endpoint
        timeout = 300  # Default to 300 seconds for this endpoint

        # Use custom timeout from request if provided
        if request.timeout_seconds is not None:
            timeout = request.timeout_seconds

        # Load tools from the MCP server using the utility function
        tools = await load_mcp_tools(
            server_type=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            timeout_seconds=timeout,
        )

        # Create the response with tools
        response = MCPServerMetadataResponse(
            transport=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            tools=tools,
        )

        return response
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception(f"Error in MCP server metadata endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        raise


@app.get("/api/rag/config", response_model=RAGConfigResponse)
async def rag_config():
    """Get the config of the RAG."""
    return RAGConfigResponse(provider=SELECTED_RAG_PROVIDER)


@app.get("/api/rag/resources", response_model=RAGResourcesResponse)
async def rag_resources(request: Annotated[RAGResourceRequest, Query()]):
    """Get the resources of the RAG."""
    retriever = build_retriever()
    if retriever:
        return RAGResourcesResponse(resources=retriever.list_resources(request.query))
    return RAGResourcesResponse(resources=[])
