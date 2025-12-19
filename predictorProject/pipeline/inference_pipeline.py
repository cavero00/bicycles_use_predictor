# pipeline/inference_pipeline.py
import time
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

        self.logger.info(f"[START]    ---  NEW execution--- run_id={run_id}")
        self.logger.info(f"[START] START Pipeline with run_id={run_id}")
        t0 = time.perf_counter()
        
        #      --- Preprocess ---
        t_pre = time.perf_counter()
        data = self.preprocess.transform(raw_data)
        t_pre = time.perf_counter() - t_pre
        self.logger.debug(f"[TIMING] Preprocess: {t_pre:.4f}s run_id={run_id}")
        
        #      --- Normalization ---
        t_norm = time.perf_counter()
        data = self.normalize.transform(data)
        t_norm = time.perf_counter() - t_norm
        self.logger.debug(f"[TIMING] Normalize: {t_norm:.4f}s run_id={run_id}")
        
        #      --- Prediction ---
        t_pred = time.perf_counter()
        prediction = self.predict.predict(data, run_id=run_id)
        t_pred = time.perf_counter() - t_pred
        self.logger.debug(f"[TIMING] Predict: {t_pred:.4f}s run_id={run_id}")
        
        total_time = time.perf_counter() - t0
        
        self.logger.info(
            f"[END] Pipeline run_id={run_id} | "
            f"total time = {total_time:.4f}s | "
            f"preprocess time = {t_pre:.4f}s | "
            f"normalize time = {t_norm:.4f}s | "
            f"predict time = {t_pred:.4f}s"
        )
        
        self.logger.info(f"[START]    ---  END execution--- run_id={run_id}")

        return prediction
