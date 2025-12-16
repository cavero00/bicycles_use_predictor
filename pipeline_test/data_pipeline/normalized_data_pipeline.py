# Pipeline para normalizar los datos

import pickle
import pandas as pd
import numpy as np

class Normalized_data:
    def __init__(self, path_scalers, path_encoders, input_data: pd.DataFrame):
        self.get_scalers(path_scalers)
        self.get_encoders(path_encoders)
        
        self.day_type_values = [
            "Holiday",
            "Weekend",
            "Normal"
        ]
        
        self.input_data = pd.DataFrame()
        self.output_data = pd.DataFrame()
        
        self.set_input_data(input_df=input_data)
        
    def get_ouput_data(self):
        return self.output_data
    
    def get_scalers(self, path):
        '''
        Obtains the scalers
        
        Input: Path where scalers are located in pkl format
        '''
        
        with open(path + "scalerTemperature.pkl", "rb") as f:
            self.scaler_temperature = pickle.load(f)
            
        with open(path + "scalerWind.pkl", "rb") as f:
            self.scaler_wind = pickle.load(f)  
            
        with open(path + "scalerHumidity.pkl", "rb") as f:
            self.scaler_humidity = pickle.load(f)
            
        with open(path + "scalerPrecipitation.pkl", "rb") as f:
            self.scaler_precipitation = pickle.load(f)
            
        with open(path + "scalerSnowDepth.pkl", "rb") as f:
            self.scaler_snowDepth = pickle.load(f)  
            
    def get_encoders(self, path):
        '''
        Obtains the encoders
        
        Input: Path where encoders are located in pkl format
        '''
            
        with open(path + "le_station_encoder.pkl", "rb") as f:
            self.le_station_encoder = pickle.load(f)      
            
    def set_day_type_values(self, day_type_list: list):
        self.day_type_values = day_type_list
        
    def set_input_data(self, input_df: pd.DataFrame):
        """
        Guarda el DataFrame de entrada asegurando que sea válido.
        """
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("El parámetro 'input_df' debe ser un pandas DataFrame.")

        if input_df.empty:
            print("⚠️ Aviso: el DataFrame de entrada está vacío.")

        self.input_data = input_df.copy()
        
        print("[...] Iniciando normalizacion de datos...")
        
        # 1️⃣ OHE → devuelve DataFrame
        one_encoding = self.one_hot_encoding()

        # 2️⃣ Escalado → devuelve DataFrame
        scaled_df = self.apply_scalers()

        # 3️⃣ Encoders → devuelve DataFrame (si tienes columnas codificadas)
        encoder_df = self.apply_encoders()
        
        # 4️⃣ Variables ciclicas
        cyclic_df = self.modify_time_data()

        # FIN: Combinar todos
        dfs_to_concat = [one_encoding, scaled_df, encoder_df, cyclic_df]
        self.output_data = pd.concat([df.reset_index(drop=True) for df in dfs_to_concat], axis=1)
        
        # Se añade el event si existe
        print("\t> Añadiendo columna 'event'...")
        if "event" in self.input_data.columns:
            self.output_data ["event"] = self.input_data["event"].values
            
        # Se ordenan las columnas
        print("\t> Ordenando columnas finales...")
        self.output_data = self.output_data[[
            "year",
            "temperature",
            "wind_speed",
            "relative_humidity",
            "precipitation",
            "snow_depth",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "event",
            "normal_day",
            "weekend_day",
            "holiday_day",
            "doy_sin",
            "doy_cos",
            "start_station_id",
            "end_station_id"
        ]]
        
        print("[OK] Normalizacion de datos completado.")
        
    def one_hot_encoding(self):
        """
        Realiza el one-hot encoding de la columna 'day_type'
        y lo almacena en self.output_data.
        """
        print("\t> Realizando one-hot encoding para 'day_type'...")
        
        if "day_type" not in self.input_data.columns:
            raise ValueError("El DataFrame de entrada no contiene la columna 'day_type'.")

        # Obtenemos el valor
        day_type_value = self.input_data[
                "day_type"
            ].iloc[0]

        if day_type_value not in self.day_type_values:
            raise ValueError(
                f"Valor inesperado en 'day_type': {day_type_value}. "
                f"Valores válidos: {self.day_type_values}"
            )

        # Inicializamos las columnas a 0
        ohe = {
            "normal_day": False,
            "weekend_day": False,
            "holiday_day": False
        }

        # Activamos solo la correspondiente
        if day_type_value == "Normal":
            ohe["normal_day"] = True
        elif day_type_value == "Weekend":
            ohe["weekend_day"] = True
        elif day_type_value == "Holiday":
            ohe["holiday_day"] = True

        return pd.DataFrame([ohe])
        
    def apply_scalers(self):
        """
        Aplica los scalers a las columnas numéricas del input_data.
        Devuelve un DataFrame con las columnas escaladas y lo guarda en self.scaled_data.
        """
        print("\t> Aplicando scalers a columnas numericas...")

        if self.input_data.empty:
            raise ValueError("El DataFrame de entrada esta vacio. Llama a set_input_data() antes.")

        df_scaled = self.input_data[
                [
                    "temperature",
                    "wind_speed",
                    "relative_humidity",
                    "precipitation",
                    "snow_depth"
                ]
            ].copy()

        # Columnas numéricas a escalar
        numeric_cols = {
            "temperature": self.scaler_temperature,
            "wind_speed": self.scaler_wind,
            "relative_humidity": self.scaler_humidity,
            "precipitation": self.scaler_precipitation,
            "snow_depth": self.scaler_snowDepth
        }

        for col, scaler in numeric_cols.items():
            if col in df_scaled.columns:
                df_scaled[col] = scaler.transform(df_scaled[[col]])
            else:
                print(f"⚠️ Columna '{col}' no encontrada en input_data. Se omite.")

        return df_scaled
    
    def apply_encoders(self):
        """
        Docstring for apply_encoders
        
        :param self: Description
        """
        print("\t> Aplicando encoders a columnas categoricas...")
        
        if self.input_data.empty:
            raise ValueError("El DataFrame de entrada está vacío. Llama a set_input_data() antes.")      
        
        df_encoder = self.input_data[
                [
                    "start_station_id", 
                    "end_station_id"
                ]
            ].copy()
        
        # Columnas a codificar
        encoder_cols = {
            "start_station_id": self.le_station_encoder,
            "end_station_id": self.le_station_encoder
        }
        
        for col, encoder in encoder_cols.items():
            if col in df_encoder.columns:
                unknown_values = set(df_encoder[col]) - set(encoder.classes_)
                if unknown_values:
                    raise ValueError(f"La columna '{col}' contiene valores desconocidos para el encoder: {unknown_values}")
                df_encoder[col] = encoder.transform(df_encoder[col])
            else:
                print(f"⚠️ Columna '{col}' no encontrada en input_data. Se omite.")

        return df_encoder        

    def modify_time_data(self):
        """
        Convierte la coluna de hora y mes en variables cicliccas usando senos y cosenos.
        """
        print("\t> Modificando datos de tiempo a variables ciclicas...")
        
        df_data = self.input_data[
                [
                    "started_at",
                    "time_hms_ms",
                    "month",
                    "year",
                    "hour_float",
                    "dayofyear"
                ]
            ].copy()  
        
        # Se obtiene la hora, el mes y el año
        # df_data["time_hms_ms"] = df_data["started_at"].dt.time.apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))
        # df_data["month"] = df_data["started_at"].dt.month
        # df_data["year"] = df_data["started_at"].dt.year
        
        # Se convierte la hora a variable ciclica
        # df_data["hour_float"] = df_data["time_hms_ms"].dt.total_seconds() / 3600
        df_data["hour_sin"] = np.sin(2 * np.pi * df_data["hour_float"] / 24)
        df_data["hour_cos"] = np.cos(2 * np.pi * df_data["hour_float"] / 24)
        
        # Se convierte el mes a variable ciclica
        df_data["month_sin"] = np.sin(2 * np.pi * df_data["month"] / 12)
        df_data["month_cos"] = np.cos(2 * np.pi * df_data["month"] / 12) 
        
        # Se convierte el día del año a variable ciclica
        # df_data['dayofyear'] = df_data['started_at'].dt.dayofyear
        df_data['doy_sin'] = np.sin(2 * np.pi * df_data['dayofyear'] / 365)
        df_data['doy_cos'] = np.cos(2 * np.pi * df_data['dayofyear'] / 365)           
        
        return df_data[[
            "hour_sin", 
            "hour_cos", 
            "month_sin", 
            "month_cos",
            "doy_sin",
            "doy_cos",
            "year"
        ]]