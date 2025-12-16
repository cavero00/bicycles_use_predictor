# pipeline/preprocess.py
import pandas as pd
from pipeline.base import PipelineStep
from classes.data_pipeline.preprocess_data_pipeline import Preprocessed_data

class PreprocessStep(PipelineStep):

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return Preprocessed_data(input_data=data).preprocess_data()
