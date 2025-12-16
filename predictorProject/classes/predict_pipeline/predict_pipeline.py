from tensorflow.keras.models import load_model
import numpy as np

class Model_predictor:
    def __init__(self, model_1_path, model_2_path, num_groups_extended_model_1, threshold_model_1, max_class_model_2):
        self.model_1_path = model_1_path
        self.model_2_path = model_2_path
        self.num_groups_extended_model_1 = num_groups_extended_model_1
        self.threshold_model_1 = threshold_model_1
        self.max_class_model_2 = max_class_model_2
        
        self.character_pre_log = "\t"
        
        self.load_models()
        
    def load_models(self):
        self.model_1 = load_model(self.model_1_path)
        self.model_2 = load_model(self.model_2_path)
        
    def predict(self, input_data):
        
        print(f"{self.character_pre_log}[...] Diviendo datos de entrada para prediccion")
        # Se divide la entrada en tres partes según las columnas
        x_context = input_data[[
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
            "doy_cos"
        ]]
        x_start_station = input_data[["start_station_id"]]
        x_end_station = input_data[["end_station_id"]]
        
        print(f"{self.character_pre_log}\t> x_context shape: {x_context.shape}")
        print(f"{self.character_pre_log}\t> x_start_station shape: {x_start_station.shape}")
        print(f"{self.character_pre_log}\t> x_end_station shape: {x_end_station.shape}")
        
        print(f"{self.character_pre_log}[OK] Datos divididos")
        print(f"{self.character_pre_log}[...] Reallizando prediccion del primer modelo")
        
        # Se realiza la primera predicción
        pred_1 = self.model_1.predict({
            "start_station": x_start_station,
            "end_station": x_end_station,
            "context": x_context
        }, batch_size=256)
        
        # Se obtiene la clase con mayor probabilidad y se deshace la extensión de la clase 1 (1 solo viaje)
        max_prob_class_1 = np.max(pred_1[:, 0:self.num_groups_extended_model_1], axis=1)
        
        # # Se aplica un threshold para decidir si es clase 1 o clase 2
        final_class_1 = (max_prob_class_1 >= self.threshold_model_1).astype(int)
        
        print(f"{self.character_pre_log}\t> Max prob clase 1 extendida: {max_prob_class_1}")
        print(f"{self.character_pre_log}\t> ¿Es clase 1? --> {'True' if final_class_1 else 'False'}")
        print(f"{self.character_pre_log}[OK] Prediccion del primer modelo completada")
        
        if final_class_1 == 1:
            print(f"{self.character_pre_log}\t> Se ha predicho 1 viaje")
            return 1
        
        print(f"{self.character_pre_log}[...] Reallizando prediccion del segundo modelo")
        
        # Si la clase es diferente de 1, se realiza la segunda predicción
        pred_2 = self.model_2.predict({
            "start_station": x_start_station,
            "end_station": x_end_station,
            "context": x_context
        }, batch_size=256)
        
        # Se redondea al entero mas cercano
        y_pred_int = np.rint(pred_2).astype(int)  # redondear al entero más cercano
        y_pred_int = np.clip(y_pred_int, 2, self.max_class_model_2)   # asegurar que esté entre 2 y 5
        
        print(f"{self.character_pre_log}[OK] Prediccion del segundo modelo completada")
        print(f"{self.character_pre_log}Se ha predicho {y_pred_int.item()} viajes")
        
        return y_pred_int.item()