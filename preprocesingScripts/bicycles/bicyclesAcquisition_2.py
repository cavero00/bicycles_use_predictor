# Execute: python -m dataAcquisition.dataAcquisition

import pandas as pd
import glob
import os

from tools.clean import CleanData
from tools.tools import Tools
from tools.holidays import Holidays
from data.bicycles.bicyclesInfo import Data

pd.set_option("display.max_rows", None)   # Muestra todas las filas
pd.set_option("display.max_columns", None)  # Muestra todas las columnas
pd.set_option("display.width", None)     # No corta la tabla en varias líneas
pd.set_option("display.max_colwidth", None)  # Muestra el contenido de celdas completo

class BicyclesAcq:    
    def __init__(self, years):
        self.data = Data(years)
        
    def obtain_data(self, origin_path:str, allowed_dirs:list, dest_dir:str, extract_files:bool=True, print_debug:bool=False) -> dict:
        
        if extract_files:
            print("\n\t\t --- Uncompress files ---\n")
            Tools().extract_zip_files(
                origin_path=origin_path,
                allowed_dirs=allowed_dirs,
                dest_dir=dest_dir,
                print_debug=print_debug
            )
  
        print("\n\t\t --- Reading csvs and creating groups by year ---\n")
        return self.group_csv_files(
            dest_dir=dest_dir,
            print_debug=print_debug
        )   
        
    def clean_data_dfs(self, dfs) -> Data:
        """
        Limpia cada DataFrame usando self.clean_data y devuelve un único DataFrame combinado.
        """
        print("\n\t\t --- Grouping, cleaning and analisys data ---\n")
        dfs_limpios = []

        for key,df in dfs.items():
            print(f"\n>>>>> Analisys for {key} <<<<<")
            # Limpiar cada DataFrame (suponiendo que clean_data devuelve el df limpio)
            df_limpio = self.clean_data(df)
            dfs_limpios.append(df_limpio)

        # Unir todos los DataFrame limpios en uno solo
        df_result = pd.concat(dfs_limpios, ignore_index=True)
        
        # Se obtienen los valores unicos del df para las columnas indicadas
        print("\n\t\t --- Obtain unique values ---\n")
        unique_values = self.obtain_unique_values_list(
            df=df_result,
            columns=["rideable_type", "member_casual"]
        )
        self.data.rideable_type = unique_values["rideable_type"]
        self.data.member_casual = unique_values["member_casual"]
        
        print(f"Unique value for \"rideable_type\": {self.data.rideable_type}")
        print(f"Unique value for \"member_casual\": {self.data.member_casual}")
        
        print("*" * 60) 
        print("\n\t\t --- Grouping, cleaning and analisys data FINISHED ---\n")         
        
        print("\n\t\t --- Changes data types ---\n")
        df_result = CleanData().convert_2_int_list(
            df=df_result,
            columns=["year", "month", "day"]
        )
        
        df_result = CleanData().convert_2_timedelta_list(
            df=df_result,
            columns=["time_hms_ms"]
        )        
        
        print("\n>> Column type")
        print(df_result.dtypes)
        print("\n>> Dataframe describe")
        print(df_result.describe())
        print(">> Primeros 10 valores")
        print(df_result.head(10))
        print(">> Shape del dataframe")
        print(df_result.shape)
        
        self.data.df_data = df_result
        return self.data
        
    def clean_data(self, df) -> pd.DataFrame:
        
        # ------------------------------------------------------------------
        # -------------------- Basic info --------------------      
        print("\n\t\t --- Basic info ---\n")
        print(df.dtypes)
        print(df.describe())  

        print("*" * 60)

        # ------------------------------------------------------------------
        # -------------------- Convert datetime columns --------------------
        print("\n\t\t --- Convert data ---\n")
        datetime_columns = ["started_at", "ended_at"]
        df_preprocessed = CleanData().convert_2_datetime_list(
            df=df,
            columns=datetime_columns
        )
        print(f"\nColumns {datetime_columns} converted to datetime")
        print(df_preprocessed.dtypes)
        print("*" * 60)
        
        # ------------------------------------------------------------------
        # -------------------- Create new columns --------------------
        print("\n\t\t --- Create new columns ---\n")        
        df_preprocessed["year"] = df_preprocessed["started_at"].dt.year
        df_preprocessed["month"] = df_preprocessed["started_at"].dt.month
        df_preprocessed["day"] = df_preprocessed["started_at"].dt.day
        df_preprocessed["time_hms_ms"] = df_preprocessed["started_at"].dt.strftime("%H:%M:%S.%f").str[:-3]
        
        print("*" * 60)

        # # ------------------------------------------------------------------
        # # -------------------- NULL data analysis by year --------------------
        # print("\n\t\t --- NULL data analysis by year ---\n")

        # # Calcular % de nulos por columna y por año
        # null_percent_by_year = (
        #     df_preprocessed
        #     .groupby("year", group_keys=False)
        #     .apply(lambda g: g.isnull().sum() / len(g) * 100, include_groups=False)
        # )

        # print("\nNull percent value by year (sin eliminar datos):")
        # print(null_percent_by_year)

        # print("*" * 60)

        # ------------------------------------------------------------------
        # -------------------- NULL data delete --------------------
        print("\n\t\t --- NULL data delete ---\n")
        null_percent_before = round(df_preprocessed.isnull().sum() / len(df_preprocessed) * 100, 2)
        print("\nNull percent value before cleaning:")
        print(null_percent_before)

        df_without_null = CleanData().delete_null_list(
            df=df_preprocessed,
            columns=["started_at", "ended_at", "start_station_name", "start_station_id", "end_station_name", "end_station_id"]
        )

        null_percent_after = round(df_without_null.isnull().sum() / len(df_without_null) * 100, 2)
        print("\nNull percent value after cleaning:")
        print(null_percent_after)

        print("\nPercent data lost:")
        print(f"\tOriginal data: {len(df_preprocessed)}")
        print(f"\tWithout null data: {len(df_without_null)}")
        print(f"\t% data lost: {((len(df_preprocessed)-len(df_without_null))/len(df_preprocessed))*100:.2f}%")
        
        df_preprocessed = df_without_null
        
        print("*" * 60)
    
        # ------------------------------------------------------------------
        # -------------------- Adding holidays --------------------
        print("\n\t\t --- Adding holidays ---\n")
        self.data.df_holidays = Holidays().get_holidays_list(
            years=self.data.years,
            state=self.data.state
        )

        # Creamos un set para buscar más rápido
        holidays_set = set(self.data.df_holidays["date"])

        df_preprocessed["day_type"] = df_preprocessed["started_at"].apply(
            lambda d: self.date_classification(d, holidays_set)
        )
        print("*" * 60)
        
        # ------------------------------------------------------------------
        # -------------------- Adding events --------------------
        print("\n\t\t --- Adding events ---\n")
        # Creamos un set para buscar más rápido
        events_set = set(self.data.events)

        df_preprocessed["event"] = df_preprocessed["started_at"].apply(
            lambda d: self.event_detect(d, events_set)
        )
        print("*" * 60)        

        self.data.df_data = df_preprocessed
        
        return df_preprocessed

    def obtain_unique_values_list(self, df: pd.DataFrame, columns: list) -> dict:
        """
        Devuelve un diccionario con los valores únicos de cada columna indicada
        """
        return {col: df[col].unique() for col in columns}
    
    def date_classification(self, date, holidays_set) -> str:
        """
        Clasifica la fecha como Holiday, Weekend o Normal
        """
        fecha_dia = date.normalize()

        if fecha_dia in holidays_set:
            return "Holiday"
        elif date.weekday() >= 5:  # 5 = sábado, 6 = domingo
            return "Weekend"
        else:
            return "Normal"
        
    def event_detect(self, date, events_set) -> bool:
        """
        Indica si hubo evento ese día
        """
        fecha_dia = date.normalize()

        if fecha_dia in events_set:
            return True
        return False      
        
    def group_csv_files(self, dest_dir, print_debug=False) -> dict:
        dfs_por_directorio = {}
    
        for subdir_name in os.listdir(dest_dir):
            subdir_path = os.path.join(dest_dir, subdir_name)
            if os.path.isdir(subdir_path):
                if print_debug:
                    print(f"Procesando subdirectorio: {subdir_path}")

                # Buscar todos los CSV
                archivos_csv = glob.glob(os.path.join(subdir_path, "*.csv"))
                if archivos_csv:
                    df_list = [pd.read_csv(f) for f in archivos_csv]
                    df_combinado = pd.concat(df_list, ignore_index=True)
                    dfs_por_directorio[subdir_name] = df_combinado
                    if print_debug:
                        print(f"\t{len(archivos_csv)} archivos CSV combinados -> {df_combinado.shape[0]} filas\n")
                else:
                    if print_debug:
                        print(f"No se encontraron archivos CSV en {subdir_path}")          
                        
        return dfs_por_directorio