import logging
from abc import ABC, abstractmethod
from typing import Any, Generator, List, Optional, Type, Union

from pydantic import BaseModel

from utils.types import ChatMessage, ChatResponse, Tools


class BaseProvider(ABC):

    @classmethod
    def get_logger(cls) -> logging.Logger:
        logger = logging.getLogger(f"providers.{cls.__name__.lower()}")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def __init__(self):
        self.logger = self.get_logger()
        self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def chat(
        self,
        messages: List[Union[str, dict, ChatMessage]],
        model: Optional[str] = None,
        tools: Optional[List[Tools]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        stream: Optional[bool] = False,
        **kargs: Any,
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        pass

    @abstractmethod
    def models(self) -> Optional[List[str]]:
        pass
