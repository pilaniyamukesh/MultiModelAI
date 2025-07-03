import importlib
import logging
import os
from typing import Any, Generator, List, Optional, Type, Union

from pydantic import BaseModel

from providers.base import BaseProvider
from utils.types import ChatMessage, ChatResponse, Tools


class Client:

    @classmethod
    def get_logger(cls) -> logging.Logger:
        logger = logging.getLogger(f"{cls.__name__.lower()}")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def __init__(
        self,
        provider: str,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.logger = self.get_logger()
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = self._get_api_key()
        self.endpoint = endpoint
        self.timeout = timeout
        self.client = self.__get_provider_client()


    def _get_api_key(self, credentials_file: str = os.path.expanduser("~/_pw")) -> Optional[str]:
        if self.provider.lower() == "ollama":
            return None

        env_var = f"{self.provider.upper()}_API_KEY"
        api_key = os.getenv(env_var)

        if not api_key:
            self.logger.debug(f"{env_var} environment variable not found, checking in credentials file.")
            try:
                with open(credentials_file, "r") as file:
                    credentials = dict(line.strip().split("=") for line in file)
                    api_key = credentials.get("_api")
                    os.environ[env_var] = api_key
            except FileNotFoundError:
                self.logger.error(f"The credentials file '{credentials_file}' is not found.")
                raise
            except Exception as e:
                self.logger.error(f"Failed to read credentials file: {e}. Please check the file path and format.")
                raise

        if not api_key:
            self.logger.error(f"Missing API key. Please set the {env_var} environment variable.")
            raise ValueError(f"Missing API key. Please set the {env_var} environment variable.")

        return api_key

    def __get_provider_client(self) -> BaseProvider:
        #try:
            module = importlib.import_module(
                f"providers.{self.provider}"
            )
            class_name = f"{self.provider.capitalize()}Client"
            provider_class = getattr(module, class_name)
            self.logger.info(f"Initializing client for provider: {self.provider}")

            return provider_class(
                self.model_name, self.api_key, self.endpoint, self.timeout
            )
        #except (ModuleNotFoundError, AttributeError):
        #    raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
        self,
        messages: List[Union[str, dict, ChatMessage]],
        model: Optional[str] = None,
        tools: Optional[List[Tools]] = None,  # Todo: check Tools
        response_format: Optional[Type[BaseModel]] = None,
        stream: Optional[bool] = False,
        **kargs: Any,
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        return self.client.chat(messages, model, tools, response_format, stream, **kargs)

    def models(self) -> Optional[List[str]]:
        return self.client.models()
