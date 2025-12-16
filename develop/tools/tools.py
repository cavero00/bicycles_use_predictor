import zipfile
import os
import pandas as pd

class Tools:
    def extract_zip_files(self, origin_path, allowed_dirs, dest_dir, print_debug:bool=False):
        """
        Recorre subdirectorios de 'origin_path', y si el nombre del subdirectorio
        está en 'allowed_dirs', descomprime todos los .zip en 'directorio_destino'.
        """
        for root, dirs, files in os.walk(origin_path):
            for dir_name in dirs:                
                if dir_name in allowed_dirs:
                    path_actual = os.path.join(root, dir_name)
                    destino_actual = os.path.join(dest_dir, dir_name)

                    if print_debug:
                        print(f"Procesando directorio: {path_actual}")
                        print(f"Destino de extracción: {destino_actual}")

                    os.makedirs(destino_actual, exist_ok=True)

                    # Buscar y extraer archivos .zip
                    for file in os.listdir(path_actual):
                        if file.endswith(".zip"):
                            zip_path = os.path.join(path_actual, file)
                            if print_debug:
                                print(f"\tExtrayendo {zip_path}...")
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(destino_actual)   
                                
    def read_csv_files(self, origin_path, allowed_dirs, separator:str=",", print_debug:bool=False) -> dict:
        df_per_directory = dict()
        
        for root, dirs, files in os.walk(origin_path):
            for dir_name in dirs:                
                if dir_name in allowed_dirs:
                    path_actual = os.path.join(root, dir_name)

                    if print_debug:
                        print(f"Procesando directorio: {path_actual}")

                    # Busca y lee el psv
                    for file in os.listdir(path_actual):
                        if file.endswith(".psv"):
                            psv_path = os.path.join(path_actual, file)
                            if print_debug:
                                print(f"\tObteniendo {psv_path}...")
                                df_per_directory[dir_name] = pd.read_csv(psv_path, sep=separator)   
                                    
        return df_per_directory       