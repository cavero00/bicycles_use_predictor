import pandas as pd
import holidays

class Holidays:
    def __init__(self, country="ES", prov=None, state=None):
        """
        country: código del país (ES, FR, US, etc.)
        prov/state: opcional (ej: prov="CA" en España no suele usarse)
        """
        self.country = country
        self.prov = prov
        self.state = state

        self.holidays = holidays.country_holidays(
            country=country,
            prov=prov,
            state=state
        )

    def day_type(self, date) -> str:
        """
        Devuelve: 'Holiday', 'Weekend' o 'Normal'
        """
        # Asegurar formato date
        if isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, str):
            date = pd.to_datetime(date).date()

        if date in self.holidays:
            return "Holiday"
        elif date.weekday() >= 5:  # sábado o domingo
            return "Weekend"
        else:
            return "Normal"
