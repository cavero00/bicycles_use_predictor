# pipeline/predict.py
import pandas as pd
import numpy as np
from classes.predict_pipeline.predict_pipeline import Model_predictor

from pipeline.logger import get_logger

class PredictStep:

    def __init__(
        self,
        model_1_path: str,
        model_2_path: str,
        num_groups_extended_model_1: int,
        threshold_model_1: float,
        max_class_model_2: int,
        logger=None
    ):
        self.predictor = Model_predictor(
            model_1_path=model_1_path,
            model_2_path=model_2_path,
            num_groups_extended_model_1=num_groups_extended_model_1,
            threshold_model_1=threshold_model_1,
            max_class_model_2=max_class_model_2
        )
        self.logger = logger or get_logger("PredictStep")

    def predict(self, data: pd.DataFrame, run_id: str = None):
        run_info = f"run_id={run_id}" if run_id else ""
        self.logger.info(f"[START] Start prediction {run_info}")

        pred_2 = self.predictor.predict(data)
        y_pred_int = np.rint(pred_2).astype(int)
        y_pred_int = np.clip(y_pred_int, 2, self.predictor.max_class_model_2)
        y_pred_int = y_pred_int.item()

        self.logger.info(f"[OK] Prediction completed {run_info}: {y_pred_int} trips")
        return y_pred_int
