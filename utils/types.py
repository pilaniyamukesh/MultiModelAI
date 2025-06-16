from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional, Dict, List, Any
from enum import Enum


class BaseFunction(BaseModel):
    name: str = Field(description="The name of the function.")


class BaseTool(BaseModel):
    type: str = Field("function", Literal="function",
                      description="The type of the tool. Currently only 'function' is supported.")


class ToolsFunctionDefinition(BaseFunction):
    description: str = Field(description="A description of what the function does.")
    parameters: Dict[str, Any] = Field(
        ..., description="The parameters the function accepts, described as a JSON Schema object.")


class ToolsCallFunction(BaseFunction):
    arguments: Dict[str, Any] = Field(
        description="The arguments to call the function with, described as a JSON object.")


class Tools(BaseTool):
    function: ToolsFunctionDefinition = Field(
        description="The definition of the function this tool provides.")


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
    role: str = Field("user",
        description="The role of the message sender (e.g., user, assistant, system, tool).")
    content: str = Field(description="The main content of the message.")

    #@field_validator('content')
    #@classmethod
    #def content_must_not_be_empty_for_user_system(cls, v, info):
    #    if info.data and info.data.get('role') in [ChatRole.user, ChatRole.system] and (v is None or v.strip() == ""):
    #        raise ValueError(
    #            f"Content cannot be empty for '{info.data.get('role')}' role messages.")
    #    return ValueError


class ChatResponseContent(BaseModel):
    ChatMessage
    tool_calls: Optional[List[ToolsCall]] = Field(
        None, description="A list of tool calls made by the assistant in this message.")


class ChatUsage(BaseModel):
    input_tokens: Optional[int] = Field(None, description="The number of input tokens used.")
    output_tokens: Optional[int] = Field(None, description="The number of output tokens generated.")
    total_tokens: Optional[int] = Field(
        None, description="The total number of tokens (input + output).")

    @field_validator('input_tokens', 'output_tokens', 'total_tokens', mode='before')
    @classmethod
    def convert_tokens_to_int(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Token count must be an integer, got {v} ({type(v)})") from e


class ChatResponse(BaseModel):
    id: Optional[str] = Field(None, description="A unique identifier for the chat completion.")
    message: Optional[ChatResponseContent] = Field(
        None, description="The content of the message, its role, and any associated tool calls.")
    usage: Optional[ChatUsage] = Field(
        None, description="Token usage statistics for the completion.")
    finish_reason: Optional[str] = Field(
        None, description="The reason the model stopped generating tokens (e.g., 'stop', 'tool_calls').")
    done_reason: Optional[str] = Field(
        None, description="An alternative or additional reason for completion (e.g., from Ollama).")

