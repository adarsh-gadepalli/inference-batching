from abc import ABC, abstractmethod
from typing import List, Any

class BaseModel(ABC):
    @abstractmethod
    def load(self):
        """load model weights and resources."""
        pass

    @abstractmethod
    def predict(self, inputs: List[Any]) -> List[Any]:
        """run batch inference."""
        pass

