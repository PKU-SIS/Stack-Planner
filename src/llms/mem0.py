import json
import asyncio
from mem0 import AsyncMemory, Memory
from src.utils.logger import logger
from src.config import load_yaml_config
from typing import Any, Dict
from pathlib import Path

def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())

def get_mem0_conf() -> Dict[str, Any]:
    conf = load_yaml_config(_get_config_file_path())
    return conf.get("MEM_MODEL", {})

def mem0_add(response):
    m = Memory.from_config(config_dict=get_mem0_conf())

    content = response.content
    if isinstance(content, str):
        data = json.loads(content)
    elif isinstance(content, dict):
        data = content
    else:
        logger.error(f"Unsupported response.content type: {type(content)}")
        return []

    results = []

    for item in data.get("user_profiles", []):
        messages = item
        try:
            res = m.add(
                messages=messages,
                user_id="dev",
                metadata={"catagory": "user_profile"},
                infer=False,
            )
            results.append(res)
            # logger.info(f"user_profile add result: {res}")
        except Exception as e:
            logger.exception(f"user_profile add failed: {e}")
            results.append(e)

    for item in data.get("semantic_memory", []):
        messages = item
        try:
            res = m.add(
                messages=messages,
                user_id="dev",
                metadata={"catagory": "semantic_memory"},
                infer=False,
            )
            results.append(res)
            # logger.info(f"semantic_memory add result: {res}")
        except Exception as e:
            logger.exception(f"semantic_memory add failed: {e}")
            results.append(e)

    for sop in data.get("SOP", []):
        sop_text = json.dumps(sop, ensure_ascii=False)
        messages = sop_text
        try:
            res = m.add(
                messages=messages,
                user_id="dev",
                metadata={"catagory": "SOP"},
                infer=False,
            )
            results.append(res)
            # logger.info(f"SOP add result: {res}")
        except Exception as e:
            logger.exception(f"SOP add failed: {e}")
            results.append(e)

    return results

async def mem0_search(querys, user_id = "dev"):
    m = await AsyncMemory.from_config(config_dict=get_mem0_conf())
    tasks = []
    for query in querys:
        tasks.append(
            m.search(
                query,
                user_id=user_id,
                # filters={"category": "user_profile"}
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    paired_results = []
    for query, result in zip(querys, results):
        if not isinstance(result, Exception):
            paired_results.append({
                "query": query,
                "result": result,
            })

    logger.info(f"search results: {paired_results}")
    return paired_results