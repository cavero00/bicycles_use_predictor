from pipeline.inference_pipeline import InferencePipeline
import pandas as pd

pipeline = InferencePipeline.from_config("config/inference.yaml")

prediction = pipeline.run(pd.DataFrame({
    "temperature": [23],
    "wind_speed": [0.4],
    "relative_humidity": [0.4],
    "precipitation": [0.02],
    "snow_depth": [0.0],
    "start_station_id": ['8fd03909-317e-4b6c-b8e9-f4c28b4c90b1'],
    "end_station_id": ['a3ad09b2-a135-11e9-9cda-0a87ae2ba916'],
    "started_at": [pd.to_datetime("2023-07-15 14:30:00")]
}))