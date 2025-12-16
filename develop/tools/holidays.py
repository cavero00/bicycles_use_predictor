import pandas as pd
import holidays

class Holidays:
    def get_holidays(self, year:int, state:str='IL'):
        """
        Devuelve un DataFrame con los festivos de una Ciudad/estado de EEUU
        para los años indicados.
        """
        # Festivos en Illinois (Chicago pertenece a este estado)
        us_holidays = holidays.US(years=year, state=state)

        # Convertir en DataFrame ordenado
        df_holidays = pd.DataFrame(
            sorted(us_holidays.items()), 
            columns=["date", "holiday"]
        )
        df_holidays["date"] = pd.to_datetime(df_holidays["date"])

        return df_holidays

    def get_holidays_list(self, years: list, state:str='IL'):
        """
        Devuelve un DataFrame con los festivos de varios años
        en un estado de EEUU.
        """
        df_list = []
        for year in years:
            df_year = self.get_holidays(int(year), state)  # Convertir a int si vienen como str
            df_list.append(df_year)
        df_result = pd.concat(df_list, ignore_index=True)
        return df_result
