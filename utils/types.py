from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any, Union
from enum import Enum
import json


class BaseFunction(BaseModel):
    name: Optional[str] = Field(None, description="The name of the function.")


class BaseTool(BaseModel):
    type: str = Field(
        "function",
        Literal="function",
        description="The type of the tool. Currently only 'function' is supported.",
    )


class ToolsFunctionDefinition(BaseFunction):
    description: str = Field(description="A description of what the function does.")
    parameters: Dict[str, Any] = Field(
        ...,
        description="The parameters the function accepts, described as a JSON Schema object.",
    )


class ToolsCallFunction(BaseFunction):
    arguments: Union[str, Dict[str, Any], None] = Field(
        description="The arguments to call the function with, described as a JSON object."
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string for arguments: {e}")
        return value


class Tools(BaseTool):
    function: ToolsFunctionDefinition = Field(
        description="The definition of the function this tool provides."
    )


class ToolsCall(BaseTool):
    id: Optional[str] = Field(None, description="An optional ID for the tool call.")
    function: ToolsCallFunction = Field(description="The details of the function call.")


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"
    developer = "developer"
    codereviewer = "codereviewer"


class ChatMessage(BaseModel):
    role: str = Field(
        "user",
        description="The role of the message sender (e.g., user, assistant, system, tool).",
    )
    content: str = Field(description="The main content of the message.")


class ChatUsage(BaseModel):
    prompt_tokens: Optional[int] = Field(
        None, description="The number of input tokens used."
    )
    completion_tokens: Optional[int] = Field(
        None, description="The number of output tokens generated."
    )
    total_tokens: Optional[int] = Field(
        None, description="The total number of tokens (prompt + completion)."
    )

    @field_validator(
        "prompt_tokens", "completion_tokens", "total_tokens", mode="before"
    )
    @classmethod
    def convert_tokens_to_int(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Token count must be an integer, got {v} ({type(v)})"
            ) from e


class FinishReason(str, Enum):
    stop = "stop"
    length = "length"
    error = "error"
    tool_calls = "tool_calls"
    end_turn = "end_turn"


class ChatResponse(BaseModel):
    id: Optional[str] = Field(
        None, description="A unique identifier for the chat completion."
    )
    model: Optional[str] = Field(
        None, description="A unique identifier for the chat completion."
    )
    message: Optional[ChatMessage] = Field(
        None,
        description="The content of the message, its role, and any associated tool calls.",
    )
    tool_calls: Optional[List[ToolsCall]] = Field(
        None, description="A list of tool calls made by the assistant in this message."
    )
    usage: Optional[ChatUsage] = Field(
        None, description="Token usage statistics for the completion."
    )
    finish_reason: Optional[str] = Field(
        None,
        description="An alternative or additional reason for completion (e.g., from Ollama).",
    )
