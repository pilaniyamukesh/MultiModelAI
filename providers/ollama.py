from typing import Any, Dict, Generator, List, Optional, Type, Union

import httpx
from pydantic import BaseModel, ValidationError

from providers.base import BaseProvider
from utils.types import (ChatMessage, ChatResponse, ChatRole, Tools, ToolsCall,
                         ToolsCallFunction)


class OllamaClient(BaseProvider):
    _OLLAMA_BASE_URL = "http://localhost:11434"
    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _OLLAMA_KNOWN_OPTIONS_KEYS = {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "tfs_z",
        "typical_p",
        "repeat_last_n",
        "repeat_penalty",
        "seed",
        "num_predict",
        "stop",
        "num_ctx",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "num_thread",
        "low_vram",
        "f16_kv",
        "vocab_only",
        "use_mmap",
        "use_mlock",
        "num_keep",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "num_gqa",
    }
    _TIMEOUT = 300

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_url = endpoint if endpoint else self._OLLAMA_BASE_URL
        self.client = None  # if ollama sdk is required, empty for now
        self.timeout = timeout if timeout is not None else self._TIMEOUT

    def chat(
        self,
        messages: List[Union[str, dict, ChatMessage]],
        model: Optional[str] = None,
        tools: Optional[List[Tools]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        stream: Optional[bool] = False,
        **kargs: Any,
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        other_kargs = {}
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages_payload(messages),
            "tools": tools,
            "options": self._options_payload(other_kargs, **kargs),
            "format": self._format_payload(response_format),
            "stream": stream,
        }
        # payload.update(other_kargs)
        # Remove keys with None values
        payload = {k: v for k, v in payload.items() if v is not None}
        self.logger.debug(f"Sending a request to Ollama model: {self.model_name}")

        try:
            response = httpx.post(
                self.base_url + self._CHAT_COMPLETION_ENDPOINT,
                json=payload,
                timeout=self.timeout,
                verify=False,
            )
            response.raise_for_status()
            data = response.json()

            # Convert this to ollama response to chatResponse format
            return OllamaChatResponse(data)

        except httpx.HTTPError as e:
            self.logger.error(f"Chat request failed for Ollama model {self.model_name}: {e}")
            raise RuntimeError(f"Chat request failed: {e}") from e

    def _messages_payload(
        self, messages_input: List[Union[str, dict, ChatMessage]]
    ) -> List[Dict[str, str]]:
        """
        Normalizes the incoming 'messages' list into the format expected by the Ollama API:
        List[{"role": "...", "content": "..."}]
        """
        normalized_messages = []
        for msg in messages_input:
            if isinstance(msg, str):
                # If it's a string, assume it's a user message
                normalized_messages.append(
                    ChatMessage(role=ChatRole.user, content=msg).model_dump()
                )
            elif isinstance(msg, dict):
                # If it's a dict, validate it against ChatMessage Pydantic model
                try:
                    # Use ChatMessage to validate and convert to dict, ensuring correct structure
                    normalized_messages.append(
                        ChatMessage.model_validate(msg).model_dump()
                    )
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid message dictionary format: {msg}. Errors: {e}"
                    )
            elif isinstance(msg, ChatMessage):
                # If it's already a ChatMessage Pydantic object, convert it to dict
                normalized_messages.append(msg.model_dump())
            else:
                raise TypeError(
                    f"Invalid message type in input list: {type(msg)}. Expected str, dict, or ChatMessage."
                )
        return normalized_messages

    def _tools_payload(self, tools) -> Optional[List[Tools]]:
        api_tools_payload = None
        if tools:
            # Convert Pydantic Tool objects to their dictionary representation
            api_tools_payload = [tool for tool in tools]
        return api_tools_payload

    def _options_payload(self, other_kargs, **kargs):
        api_options_payload: Dict[str, Any] = {}
        for key, value in kargs.items():
            if key in self._OLLAMA_KNOWN_OPTIONS_KEYS:
                api_options_payload[key] = value
            else:
                other_kargs[key] = value
        return api_options_payload

    def _format_payload(self, response_format):
        api_response_format = None
        if response_format:
            api_response_format = {
                "type": "json_object",
                "schema": response_format.model_json_schema(),
            }
        return api_response_format

    def models(self) -> Optional[List[str]]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model.get("name") for model in models if "name" in model]
        except httpx.RequestError as e:
            raise RuntimeError(f"Error fetching models: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching models: {e}") from e


class OllamaChatResponse(ChatResponse):
    def __init__(self, response: dict) -> None:
        super().__init__(
            id=None,
            message=ChatMessage(
                role=response.get("message", {}).get("role"),
                content=response.get("message", {}).get("content"),
            ),
            model=response.get("model"),
            tool_calls=self._extract_tool_calls(response),
            usage=None,
            finish_reason=response.get("done_reason"),
        )

    def _extract_tool_calls(self, response: dict) -> Optional[List[ToolsCall]]:
        tool_calls_data = response.get("message", {}).get("tool_calls")
        if not tool_calls_data:
            return None

        return [
            ToolsCall(
                id=tool_call.get("id", None),
                function=ToolsCallFunction(
                    name=tool_call.get("function", {}).get("name", ""),
                    arguments=tool_call.get("function", {}).get("arguments", {}),
                ),
            )
            for tool_call in tool_calls_data
        ]
