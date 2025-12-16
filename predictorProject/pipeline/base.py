# pipeline/base.py
from abc import ABC, abstractmethod
import pandas as pd

class PipelineStep(ABC):

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
