import importlib
from typing import Optional, List, Union, Dict, Type, Any, Generator
from pydantic import BaseModel
from providers.base import BaseProvider
from utils.logger import get_logger
from utils.types import ChatMessage, ChatResponse, Tools


class Client:
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.provider = provider.lower()
        self.model_name = model_name  # if model_name is not passed then set to default
        self.api_key = api_key  # Is this field is really required
        self.client = self.__get_provider_client()
        self.logger.info(f"Initialied client for provider: {self.provider}")

    def __get_provider_client(self) -> BaseProvider:
        print(f''''{importlib.import_module(f"providers.{self.provider}")}''')
        try:
            module = importlib.import_module(f"providers.{self.provider}")
            print(f"module: {module}")
            class_name = f"{self.provider.capitalize()}Client"
            print(f"class: {class_name}")
            provider_class = getattr(module, class_name)

            # Retrieve the API key from environment variables if not provided
            if self.provider != "ollama" and api_key is None:
                env_var = f"{self.provider.upper()}_API_KEY"
                api_key = os.getenv(env_var)
                if api_key is None:
                    self.logger.error(f"API key missing for provider: {self.provider}")
                    raise ValueError(
                        f"Missing API key. Please provide it explicitly or set the '{env_var}' environment variable."
                    )
                self.logger.debug(f"API key loaded from environment: {env_var}")

            return provider_class(self.model_name, self.api_key)
        except (ModuleNotFoundError, AttributeError) as e:
            self.logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
            self,
            message: List[Union[str, dict, ChatMessage]],
            tools: Optional[List[Tools]] = None,  # Todo: check Tools
            response_format: Optional[Type[BaseModel]] = None,
            stream: Optional[bool] = False,
            **kargs: Any
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        return self.client.chat(
            message,
            tools,
            response_format,
            stream,
            **kargs
        )

    def all_model(self):
        return self.client.all_model()
