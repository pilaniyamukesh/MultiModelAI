import requests
import httpx
from typing import Optional, List, Union, Dict, Type, Any, Generator
from pydantic import BaseModel
from utils.types import ChatMessage, ChatResponse, Tools, ChatRole
from providers.base import BaseProvider


class OllamaClient(BaseProvider):
    _OLLAMA_BASE_URL = "http://localhost:11434"
    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _OLLAMA_KNOWN_OPTIONS_KEYS = {
        "temperature", "top_k", "top_p", "min_p", "tfs_z", "typical_p",
        "repeat_last_n", "repeat_penalty", "seed", "num_predict", "stop",
        "num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread",
        "low_vram", "f16_kv", "vocab_only", "use_mmap", "use_mlock", "num_keep",
        "mirostat", "mirostat_eta", "mirostat_tau", "num_gqa"
    }

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name  # if model_name is not passed set default one
        self.base_url = self._OLLAMA_BASE_URL  # if ollama url is not passed set to default one
        self.client = None  # if ollama sdk is required, empty for now

    def chat(
            self,
            messages: List[Union[str, dict, ChatMessage]],
            tools: Optional[List[Tools]] = None,  # Todo: check Tools
            response_format: Optional[Type[BaseModel]] = None,
            stream: Optional[bool] = False,
            **kargs: Any
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
            # other_kargs
        }
        # Remove keys with None values
        payload = {k: v for k, v in payload.items() if v is not None}

        print(f"payload: {payload}")
        try:
            response = requests.post(
                self.base_url + self._CHAT_COMPLETION_ENDPOINT,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            # Optional: convert response to Pydantic model if specified
            if response_format:
                return response_format.parse_obj(data)
            print(f"response from Ollama: {data}")
            return ChatResponse(**data)  # or however you build ChatResponse

        except httpx.HTTPError as e:
            raise RuntimeError(f"Chat request failed: {e}") from e

    def _messages_payload(self, messages_input: List[Union[str, dict, ChatMessage]]) -> List[Dict[str, str]]:
        """
        Normalizes the incoming 'messages' list into the format expected by the Ollama API:
        List[{"role": "...", "content": "..."}]
        """
        normalized_messages = []
        for msg in messages_input:
            if isinstance(msg, str):
                # If it's a string, assume it's a user message
                normalized_messages.append(ChatMessage(
                    role=ChatRole.user, content=msg).model_dump())
                print(f"first normalized_messages: {normalized_messages}")
            elif isinstance(msg, dict):
                # If it's a dict, validate it against ChatMessage Pydantic model
                try:
                    # Use ChatMessage to validate and convert to dict, ensuring correct structure
                    normalized_messages.append(ChatMessage.model_validate(msg).model_dump())
                except ValidationError as e:
                    raise ValueError(f"Invalid message dictionary format: {msg}. Errors: {e}")
            elif isinstance(msg, ChatMessage):
                # If it's already a ChatMessage Pydantic object, convert it to dict
                normalized_messages.append(msg.model_dump())
            else:
                raise TypeError(
                    f"Invalid message type in input list: {type(msg)}. Expected str, dict, or ChatMessage.")
        print(f" last normalized_messages : {normalized_messages}")
        return normalized_messages

    def _tools_payload(self, tools) -> List[Dict]:
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

    def _format_payload(sef, response_format):
        api_response_format = None
        if response_format:
            api_response_format = {
                "type": "json_object",
                # "schema": response_format.model_json_schema()
            }
        return api_response_format

    def all_model(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            return f"Error fetching models: {e}"
