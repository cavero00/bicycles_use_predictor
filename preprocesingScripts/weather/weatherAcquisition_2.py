import os
import pandas as pd

from tools.clean import CleanData

class WeatherAcq:
    def clean_data_dfs(self , dfs) -> pd.DataFrame:
        print("\n\t\t --- Grouping, cleaning and analisys data ---\n")
        dfs_limpios = []
        
        for key,df in dfs.items():
            print(f"\n>>>>> Analisys for {key} <<<<<")
            # Limpiar cada DataFrame (suponiendo que clean_data devuelve el df limpio)
            df_limpio = self.clean_data(df)
            dfs_limpios.append(df_limpio)

        # Unir todos los DataFrame limpios en uno solo
        df_result = pd.concat(dfs_limpios, ignore_index=True)   
        
        print("Estadisticas del df:")
        print(df_result.describe())
        
        return df_result   
        
    def clean_data(self, df) ->pd.DataFrame:
        # ------------------------------------------------------------------
        # -------------------- Basic info --------------------      
        print("\n\t\t --- Basic info ---\n")
        print(df.dtypes)
        print(df.describe())  
        print(df.head(10))  
        
        # ------------------------------------------------------------------
        # -------------------- Convert datetime columns --------------------        
        print("\n\t\t --- Convert data ---\n")
        datetime_columns = ["DATE"]
        df_preprocessed = CleanData().convert_2_datetime_list(
            df=df,
            columns=datetime_columns
        )
        print(f"\nColumns {datetime_columns} converted to datetime")
        print(df_preprocessed.dtypes)
        print("*" * 60)        
        
        # ------------------------------------------------------------------
        # -------------------- NaN to 0.0 --------------------
        print("\n\t\t --- NaN to 0.0 ---\n")
        # Contar NaN por columna
        nan_count = df_preprocessed.isna().sum()

        # Calcular porcentaje de NaN
        nan_percent = round((nan_count / len(df_preprocessed)) * 100, 2)

        # Combinar en un DataFrame para mostrarlo
        nan_summary = pd.DataFrame({
            "NaN_count": nan_count,
            "NaN_percent": nan_percent
        })
        
        # Se sustituyen
        df_preprocessed = df_preprocessed.fillna(0.0)
        
        print("NaN value per column:")
        print(nan_summary)
        print("*" * 60)         

        # ------------------------------------------------------------------
        # -------------------- Create new columns --------------------
        print("\n\t\t --- Create new columns ---\n")        
        df_preprocessed["year"] = df_preprocessed["DATE"].dt.year
        df_preprocessed["month"] = df_preprocessed["DATE"].dt.month
        df_preprocessed["day"] = df_preprocessed["DATE"].dt.day
        df_preprocessed["time_hms_ms"] = df_preprocessed["DATE"].dt.strftime("%H:%M:%S.%f").str[:-3]
        
        print("*" * 60)        
        
        # ------------------------------------------------------------------
        # -------------------- Check the rows -------------------- 
        print("\n\t\t --- Check the rows ---\n")    
        n_ref = df_preprocessed["STATION"].count()
        
        # Recorrer todas las columnas
        for col in df_preprocessed.columns:
            if df_preprocessed[col].count() < n_ref:
                df_preprocessed[col] = df_preprocessed[col].fillna(0.0)
                
        print(f"Reference rows are: {n_ref}")
        print(df_preprocessed.describe())
        
        print("*" * 60)     
        
        return df_preprocessed