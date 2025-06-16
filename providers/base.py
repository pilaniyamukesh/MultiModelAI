from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Type, Any, Generator
from utils.types import Tools, ChatMessage, ChatResponse
from pydantic import BaseModel


class BaseProvider(ABC):

    @abstractmethod
    def chat(
            self,
            message: List[Union[str, dict, ChatMessage]],
            tools: Optional[List[Tools]] = None,  # Todo: check Tools
            response_format: Optional[Type[BaseModel]] = None,
            stream: Optional[bool] = False,
            **kargs: Any
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        pass

    @abstractmethod
    def all_model(self):
        pass
