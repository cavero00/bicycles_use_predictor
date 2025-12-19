import pandas as pd
import numpy as np

from classes.tools.events import events
from classes.tools.holidays import Holidays

class Preprocessed_data:
    def __init__(self, input_data: pd.DataFrame):
        
        self.events = {
            pd.to_datetime(d).date() for d in events
        }
        self.input_data = input_data
        
        self.character_pre_log = "\t"
    
    def preprocess_data(self):
        self.check_input_data()
        
        print(f"{self.character_pre_log}[...] Preprocesando los datos...")
        self.add_day_type()
        self.add_timme_variables()
        self.add_event()
        print(f"{self.character_pre_log}[OK] Preprocesamiento de datos completado")
        
        return self.input_data        

    def check_input_data(self) -> bool:
        """
        Valida que el DataFrame de entrada cumpla los requisitos del pipeline.
        Lanza excepción si algo no es válido.
        """

        print(f"{self.character_pre_log}[...] Comprobando los datos de entrada...")
        # 1️⃣ Tipo
        if not isinstance(self.input_data, pd.DataFrame):
            print(f"{self.character_pre_log}[NOK] input_data debe ser un pandas DataFrame")
            raise TypeError("input_data debe ser un pandas DataFrame")

        # 2️⃣ Vacío
        if self.input_data.empty:
            print(f"{self.character_pre_log}[NOK] input_data está vacío")
            raise ValueError("input_data está vacío")

        # 3️⃣ Columnas requeridas
        required_columns = {
            "temperature": (int, float, np.number),
            "wind_speed": (int, float, np.number),
            "relative_humidity": (int, float, np.number),
            "precipitation": (int, float, np.number),
            "snow_depth": (int, float, np.number),
            "start_station_id": (str,),
            "end_station_id": (str,),
            "started_at": (pd.Timestamp, str)
        }

        missing_cols = set(required_columns) - set(self.input_data.columns)
        if missing_cols:
            print(f"{self.character_pre_log}[NOK] Faltan columnas requeridas: {missing_cols}")
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

        # 4️⃣ Una sola fila (predicción online)
        if len(self.input_data) != 1:
            print(f"{self.character_pre_log}[NOK] Se esperaba 1 fila de entrada, se recibieron {len(self.input_data)}")
            raise ValueError(
                f"Se esperaba 1 fila de entrada, se recibieron {len(self.input_data)}"
            )

        # 5️⃣ Nulos
        if self.input_data[list(required_columns)].isnull().any().any():
            print(f"{self.character_pre_log}[NOK] Existen valores nulos en columnas obligatorias")
            raise ValueError("Existen valores nulos en columnas obligatorias")

        # 6️⃣ Tipos / convertibilidad
        try:
            pd.to_datetime(self.input_data["started_at"])
        except Exception:
            print(f"{self.character_pre_log}[NOK] La columna 'started_at' no es convertible a datetime")
            raise TypeError("La columna 'started_at' no es convertible a datetime")

        for col, allowed_types in required_columns.items():
            if col == "started_at":
                continue

            value = self.input_data[col].iloc[0]
            if not isinstance(value, allowed_types):
                raise TypeError(
                    f"Columna '{col}' tiene tipo inválido: "
                    f"{type(value)} (esperado {allowed_types})"
                )

        print(f"{self.character_pre_log}[OK] Datos de entrada correctos")
        
        return True

    def add_day_type(self):
        print(f"{self.character_pre_log}\t> Añadiendo la columna 'day_type'...")
        holidays_obj = Holidays(country="US", state="IL")

        self.input_data["day_type"] = (
            pd.to_datetime(self.input_data["started_at"])
            .dt.date
            .apply(holidays_obj.day_type)
        )

    def add_timme_variables(self):
        print(f"{self.character_pre_log}\t> Añadiendo las columnas relacionadas con la fecha...")
        self.input_data["time_hms_ms"] = self.input_data["started_at"].dt.time.apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))
        self.input_data["month"] = self.input_data["started_at"].dt.month - 1 # Referencia en 0
        self.input_data["year"] = self.input_data["started_at"].dt.year
        self.input_data["hour_float"] = self.input_data["time_hms_ms"].dt.total_seconds() / 3600
        self.input_data['dayofyear'] = self.input_data['started_at'].dt.dayofyear
        
    def add_event(self):
        """
        Añade columna booleana 'event' indicando si ese día hubo evento
        """
        print(f"{self.character_pre_log}\t> Añadiendo la columna 'event'...")
        event_dates = {
            pd.to_datetime(d).date() for d in self.events
        }

        self.input_data["event"] = (
            pd.to_datetime(self.input_data["started_at"])
            .dt.date
            .isin(event_dates)
        )
