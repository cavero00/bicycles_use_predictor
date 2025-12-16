# pipeline/normalize.py
import pandas as pd
from pipeline.base import PipelineStep
from classes.data_pipeline.normalized_data_pipeline import Normalized_data

class NormalizeStep(PipelineStep):

    def __init__(self, scalers_path: str, encoders_path: str):
        self.scalers_path = scalers_path
        self.encoders_path = encoders_path

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return Normalized_data(
            path_scalers=self.scalers_path,
            path_encoders=self.encoders_path,
            input_data=data
        ).get_ouput_data()
