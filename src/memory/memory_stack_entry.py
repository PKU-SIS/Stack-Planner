from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from langchain_core.messages import AIMessage, HumanMessage
from typing import Annotated, Literal, Dict, List, Optional, Any
from src.agents.sub_agent_registry import SubAgentType

# -------------------------
# 数据模型定义
# -------------------------
@dataclass
class MemoryStackEntry(DataClassJsonMixin):
    """记忆栈条目数据模型，记录系统执行历史"""

    timestamp: str  # 时间戳
    action: str  # 执行动作
    agent_type: Optional[str] = None  # 代理类型(central/sub-agent)
    content: str = ""  # 动作内容
    result: Optional[Dict[str, Any]] = None  # Sub-Agent的执行结果
    def __post_init__(self):
        allowed_agent_types = [agent.value for agent in SubAgentType] + ["central_agent"]
        if self.agent_type is not None:
            assert self.agent_type in allowed_agent_types, f"agent_type must be one of {allowed_agent_types}, got '{self.agent_type}'"

    # FIXME check 这里有没有bugs
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，确保所有消息对象都被正确序列化"""

        def convert_messages(obj):
            if isinstance(obj, (HumanMessage, AIMessage)):
                return {
                    "type": obj.__class__.__name__,
                    "content": obj.content,
                    "additional_kwargs": obj.additional_kwargs,
                }
            elif isinstance(obj, dict):
                return {k: convert_messages(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_messages(item) for item in obj]
            return obj

        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_type": self.agent_type,
            "content": convert_messages(self.content),
            "result": convert_messages(self.result),
        }
