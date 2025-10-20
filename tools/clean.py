import pandas as pd

class CleanData:
    def delete_duplicates(self, df, column):
        """
        Elimina duplicados en la columna indicada.
        Devuelve el DataFrame limpio y el número de duplicados eliminados.
        """
        duplicates_count = df[column].duplicated().sum()
        if duplicates_count > 0:
            df_result = df.drop_duplicates(subset=[column], keep="first").copy()
            return df_result, duplicates_count
        return df.copy(), duplicates_count

    def convert_2_datetime_list(self, df, columns):
        """
        Convierte varias columnas a tipo datetime.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        for column in columns:
            df_result = self.convert_2_datetime(df_result, column)
        return df_result

    def convert_2_datetime(self, df, column):
        """
        Convierte una columna a tipo datetime.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        df_result[column] = pd.to_datetime(df_result[column], errors="coerce")
        return df_result
    
    def delete_null_list(self, df, columns):
        """
        Elimina filas con valores nulos en las columnas indicadas.
        Devuelve un DataFrame limpio.
        """
        df_aux = df.copy()
        for column in columns:
            df_aux = df_aux.dropna(subset=[column])
        return df_aux
    
    def convert_2_int_list(self, df, columns):
        """
        Convierte varias columnas a tipo int.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        for col in columns:
            df_result = self.convert_2_int(df_result, col)
        return df_result

    def convert_2_int(self, df, column):
        """
        Convierte una columna a tipo int.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        df_result[column] = df_result[column].astype(int)
        return df_result            
        
    def convert_2_timedelta_list(self, df, columns):
        """
        Convierte varias columnas a tipo timedelta.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        for col in columns:
            df_result = self.convert_2_timedelta(df_result, col)
        return df_result

    def convert_2_timedelta(self, df, column):
        """
        Convierte una columna a tipo timedelta.
        Devuelve el DataFrame modificado.
        """
        df_result = df.copy()
        df_result[column] = pd.to_timedelta(df_result[column], errors="coerce")
        return df_result