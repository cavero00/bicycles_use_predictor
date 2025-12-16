import pandas as pd

from data_pipeline.preprocess_data_pipeline import Preprocessed_data
from data_pipeline.normalized_data_pipeline import Normalized_data
from predict_pipeline.predict_pipeline import Model_predictor


pipeline_preprocess = Preprocessed_data(
    input_data=pd.DataFrame({
        "temperature": [23],
        "wind_speed": [0.4],
        "relative_humidity": [0.4],
        "precipitation": [0.02],
        "snow_depth": [0.0],
        "start_station_id": ['8fd03909-317e-4b6c-b8e9-f4c28b4c90b1'],
        "end_station_id": ['a3ad09b2-a135-11e9-9cda-0a87ae2ba916'],
        "started_at": [pd.to_datetime("2023-07-15 14:30:00")]
    })
)

pipeline_data = Normalized_data(
    path_scalers="data/scalers/",
    path_encoders="data/encoders/",
    input_data=pipeline_preprocess.preprocess_data()
)

pipeline_predictor = Model_predictor(
    model_1_path="models/model_7_6_dayOfYear.keras",
    model_2_path="models/model_7_9_dayOfYear.keras",
    num_groups_extended_model_1=4,
    threshold_model_1=0.23296134, # Valor obtenido en validación
    max_class_model_2=5
)

prediction = pipeline_predictor.predict(
    input_data=pipeline_data.get_ouput_data()
)
