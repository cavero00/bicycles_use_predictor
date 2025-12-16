# pipeline/inference_pipeline.py
import yaml
import pandas as pd
import uuid
from pipeline.preprocess import PreprocessStep
from pipeline.normalize import NormalizeStep
from pipeline.predict import PredictStep

from pipeline.logger import get_logger

class InferencePipeline:

    def __init__(self, preprocess, normalize, predict):
        self.preprocess = preprocess
        self.normalize = normalize
        self.predict = predict
        
        self.logger = get_logger("InferencePipeline")

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            preprocess=PreprocessStep(),
            normalize=NormalizeStep(
                cfg["paths"]["scalers"],
                cfg["paths"]["encoders"]
            ),
            predict=PredictStep(**cfg["model"])
        )

    def run(self, raw_data: pd.DataFrame):
        run_id = str(uuid.uuid4())  # Genera un ID único por ejecución

        self.logger.info(f"[START] Pipeline completa run_id={run_id}")
        data = self.preprocess.transform(raw_data)
        data = self.normalize.transform(data)
        prediction = self.predict.predict(data, run_id=run_id)
        self.logger.info(f"[END] Pipeline completa run_id={run_id}, prediction={prediction}")

        return prediction
