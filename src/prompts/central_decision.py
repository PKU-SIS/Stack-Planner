from typing import Optional, Dict, Any, Union
from pydantic import BaseModel


class DelegateParams(BaseModel):
    agent_type: str
    task_description: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class Decision(BaseModel):
    action: str  # "think" | "reflect" | "summarize" | "delegate" | "finish"
    reasoning: str
    params: Optional[Union[DelegateParams, Dict[str, Any], None]] = None
    instruction: Optional[str] = None
    locale: Optional[str] = None
