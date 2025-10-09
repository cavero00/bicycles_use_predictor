import pandas as pd

class Data:
    state:str = "IL"
    
    df_data = pd.DataFrame()
    df_holidays = pd.DataFrame()
    
    member_casual: list = []
    rideable_type: list = []
    years: list = []    
    day_type: list = ["Holiday", "Weekend", "Normal"]
    events = [
        #             ------ 2022 ------
        
        pd.Timestamp("2022-02-12"), # Feria automovil
        pd.Timestamp("2022-02-13"), # Feria automovil
        pd.Timestamp("2022-02-14"), # Feria automovil
        pd.Timestamp("2022-02-15"), # Feria automovil
        pd.Timestamp("2022-02-16"), # Feria automovil
        pd.Timestamp("2022-02-17"), # Feria automovil
        pd.Timestamp("2022-02-18"), # Feria automovil
        pd.Timestamp("2022-02-19"), # Feria automovil
        pd.Timestamp("2022-02-20"), # Feria automovil        
        pd.Timestamp("2022-02-21"), # Feria automovil
        
        pd.Timestamp("2022-05-28"), # Festival musica (Sueños)
        pd.Timestamp("2022-05-29"), # Festival musica (Sueños)
        
        # No hubo NASCAR
        
        pd.Timestamp("2022-09-08"), # Taste of Chicago
        pd.Timestamp("2022-09-09"), # Taste of Chicago
        pd.Timestamp("2022-09-10"), # Taste of Chicago         
        
        pd.Timestamp("2022-07-15"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        pd.Timestamp("2022-07-16"), # Festival musica (Pitchfork Music) (cancelado en 2025)        
        pd.Timestamp("2022-07-17"), # Festival musica (Pitchfork Music) (cancelado en 2025)  
              
        pd.Timestamp("2022-07-28"), # Festival musica (Lollapalooza) 
        pd.Timestamp("2022-07-29"), # Festival musica (Lollapalooza) 
        pd.Timestamp("2022-07-30"), # Festival musica (Lollapalooza) 
        pd.Timestamp("2022-07-31"), # Festival musica (Lollapalooza) 
                        
        pd.Timestamp("2022-08-20"), # Exhibicion aerea (Chicago Air & Water Show) 
        pd.Timestamp("2022-08-21"), # Exhibicion aerea (Chicago Air & Water Show)
        
        pd.Timestamp("2022-09-01"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2022-09-02"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2022-09-03"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2022-09-04"), # Festival musica (Chicafo jazz)
        
        pd.Timestamp("2022-09-16"), # Festival musica (Riot fest)
        pd.Timestamp("2022-09-17"), # Festival musica (Riot fest)        
        pd.Timestamp("2022-09-18"), # Festival musica (Riot fest)
        
        pd.Timestamp("2022-10-09"), # Maraton anual  
        
         
        pd.Timestamp("2022-10-12"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-13"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-14"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-15"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-16"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-17"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-18"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-19"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-20"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-21"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2022-10-22"), # Festival cine (60th Chicago International Film Festival)  
        pd.Timestamp("2022-10-23"), # Festival cine (60th Chicago International Film Festival)                                               
        
        #             ------ 2023 ------
        
        pd.Timestamp("2023-02-11"), # Feria automovil
        pd.Timestamp("2023-02-12"), # Feria automovil
        pd.Timestamp("2023-02-13"), # Feria automovil
        pd.Timestamp("2023-02-14"), # Feria automovil
        pd.Timestamp("2023-02-15"), # Feria automovil
        pd.Timestamp("2023-02-16"), # Feria automovil
        pd.Timestamp("2023-02-17"), # Feria automovil
        pd.Timestamp("2023-02-18"), # Feria automovil
        pd.Timestamp("2023-02-19"), # Feria automovil
        pd.Timestamp("2023-02-20"), # Feria automovil
        
        pd.Timestamp("2023-05-27"), # Festival musica (Sueños)
        pd.Timestamp("2023-05-28"), # Festival musica (Sueños)
        
        pd.Timestamp("2023-07-02"), # Nascar
        
        pd.Timestamp("2023-07-21"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        pd.Timestamp("2023-07-22"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        pd.Timestamp("2023-07-23"), # Festival musica (Pitchfork Music) (cancelado en 2025)    
        
        pd.Timestamp("2023-08-03"), # Festival musica (Lollapalooza)
        pd.Timestamp("2023-08-04"), # Festival musica (Lollapalooza)
        pd.Timestamp("2023-08-05"), # Festival musica (Lollapalooza)
        pd.Timestamp("2023-08-06"), # Festival musica (Lollapalooza)  
        
        pd.Timestamp("2023-08-19"), # Exhibicion aerea (Chicago Air & Water Show)
        pd.Timestamp("2023-08-20"), # Exhibicion aerea (Chicago Air & Water Show) 
        
        pd.Timestamp("2023-08-31"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2023-09-01"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2023-09-02"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2023-09-03"), # Festival musica (Chicafo jazz) 
        
        pd.Timestamp("2023-09-08"), # Taste of Chicago
        pd.Timestamp("2023-09-09"), # Taste of Chicago
        pd.Timestamp("2023-09-10"), # Taste of Chicago     
        
        pd.Timestamp("2023-09-15"), # Festival musica (Riot fest)
        pd.Timestamp("2023-09-16"), # Festival musica (Riot fest)
        pd.Timestamp("2023-09-17"), # Festival musica (Riot fest) 
        
        pd.Timestamp("2023-10-08"), # Maraton anual  
        
        pd.Timestamp("2023-10-11"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-12"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-13"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-14"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-15"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-16"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-17"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-18"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-19"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-20"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-21"), # Festival cine (60th Chicago International Film Festival)                                
        pd.Timestamp("2023-10-22"), # Festival cine (60th Chicago International Film Festival)                                
        
        #             ------ 2024 ------
        
        pd.Timestamp("2024-02-10"), # Feria automovil
        pd.Timestamp("2024-02-11"), # Feria automovil
        pd.Timestamp("2024-02-12"), # Feria automovil
        pd.Timestamp("2024-02-13"), # Feria automovil
        pd.Timestamp("2024-02-14"), # Feria automovil
        pd.Timestamp("2024-02-15"), # Feria automovil
        pd.Timestamp("2024-02-16"), # Feria automovil
        pd.Timestamp("2024-02-17"), # Feria automovil
        pd.Timestamp("2024-02-18"), # Feria automovil
        pd.Timestamp("2024-02-19"), # Feria automovil
        
        pd.Timestamp("2024-05-26"), # Festival musica (Sueños)
        
        pd.Timestamp("2024-07-07"), # Nascar
        
        pd.Timestamp("2024-07-19"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        pd.Timestamp("2024-07-20"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        pd.Timestamp("2024-07-21"), # Festival musica (Pitchfork Music) (cancelado en 2025)
        
        pd.Timestamp("2024-08-01"), # Festival musica (Lollapalooza)
        pd.Timestamp("2024-08-02"), # Festival musica (Lollapalooza)
        pd.Timestamp("2024-08-03"), # Festival musica (Lollapalooza)
        pd.Timestamp("2024-08-04"), # Festival musica (Lollapalooza)
        
        pd.Timestamp("2024-08-10"), # Exhibicion aerea (Chicago Air & Water Show)
        pd.Timestamp("2024-08-11"), # Exhibicion aerea (Chicago Air & Water Show)
        
        pd.Timestamp("2024-08-29"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2024-08-30"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2024-08-31"), # Festival musica (Chicafo jazz)
        pd.Timestamp("2024-09-01"), # Festival musica (Chicafo jazz)
        
        pd.Timestamp("2024-09-06"), # Taste of Chicago
        pd.Timestamp("2024-09-08"), # Taste of Chicago
        pd.Timestamp("2024-09-09"), # Taste of Chicago
        
        pd.Timestamp("2024-09-20"), # Festival musica (Riot fest)
        pd.Timestamp("2024-09-21"), # Festival musica (Riot fest)
        pd.Timestamp("2024-09-22"), # Festival musica (Riot fest)
             
        pd.Timestamp("2024-10-13"), # Maraton anual
        
        pd.Timestamp("2024-10-16"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-17"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-18"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-19"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-20"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-21"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-22"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-23"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-24"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-25"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-26"), # Festival cine (60th Chicago International Film Festival)
        pd.Timestamp("2024-10-27"), # Festival cine (60th Chicago International Film Festival)     
    ]    
    
    def __init__(self, years):
        self.years = years