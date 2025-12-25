"""
Patch for langchain_openai to handle non-standard API responses
Fixes: ChatMessage role validation error when API returns role=None
"""

import functools
from typing import Any, Dict
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
    FunctionMessage,
)
from src.utils.logger import logger


def patch_convert_dict_to_message():
    """
    Monkey patch langchain_openai's _convert_dict_to_message function
    to handle cases where role is None or missing
    """
    try:
        from langchain_openai.chat_models import base as openai_base

        # Store original function
        original_convert = openai_base._convert_dict_to_message

        @functools.wraps(original_convert)
        def _convert_dict_to_message_patched(_dict: Dict[str, Any]) -> Any:
            """
            Patched version that handles None role values
            """
            # Extract role with fallback
            role = _dict.get("role")

            # If role is None or empty, default to 'assistant'
            if not role:
                logger.warning(
                    f"API returned message with role=None, defaulting to 'assistant'. Message: {_dict}"
                )
                role = "assistant"
                _dict["role"] = role

            # Ensure content exists
            if "content" not in _dict or _dict["content"] is None:
                logger.warning(
                    f"API returned message with no content, using empty string"
                )
                _dict["content"] = ""

            # Now call original function with corrected dict
            try:
                return original_convert(_dict)
            except Exception as e:
                # Last resort fallback
                logger.error(
                    f"Failed to convert message even after patching: {e}. Dict: {_dict}"
                )
                # Return a basic AIMessage
                return AIMessage(
                    content=_dict.get("content", ""), additional_kwargs=_dict
                )

        # Apply the patch
        openai_base._convert_dict_to_message = _convert_dict_to_message_patched
        logger.info("✅ Successfully patched langchain_openai._convert_dict_to_message")
        return True

    except ImportError as e:
        logger.warning(f"Could not patch langchain_openai: {e}")
        return False
    except Exception as e:
        logger.error(f"Error applying langchain_openai patch: {e}")
        return False


def apply_all_patches():
    """
    Apply all necessary patches for langchain compatibility
    """
    logger.info("Applying langchain compatibility patches...")
    success = patch_convert_dict_to_message()

    if success:
        logger.info("✅ All patches applied successfully")
    else:
        logger.warning("⚠️ Some patches failed to apply")

    return success
