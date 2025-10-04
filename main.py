import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from preprocesingScripts.bicycles.bicyclesAcquisition_2 import BicyclesAcq
from data.bicycles.bicyclesInfo import Data

# ---------------------------------------------------------------------------
origin_path = r'D:\masters\master IA IMF\TFM\data\bicycles\raw'
dest_dir = r'D:\masters\master IA IMF\TFM\data\bicycles\raw_excel'

allowed_dirs = ["2022", "2023", "2024"]
years = allowed_dirs

extract_files = False
obtain_data = False
clean_data = True

# ---------------------------------------------------------------------------

data = Data(years)
data_operations = BicyclesAcq(years)
dfs_per_directory = dict()

# ------------------------------------------------------------------
# -------------------- Obtain data --------------------
if obtain_data:
    dfs_per_directory = data_operations.obtain_data(
        origin_path=origin_path,
        allowed_dirs=allowed_dirs,
        dest_dir=dest_dir,
        extract_files=extract_files,
        print_debug=True
    )

    # La variable obtenida es un diccionario de dataframes
    with open("data/df_data_raw_per_directory.pk1", "wb") as f:
        pickle.dump(dfs_per_directory, f)
else:
    # La variable obtenida es un diccionario de dataframes
    with open("data/df_data_raw_per_directory.pk1", "rb") as f:
        dfs_per_directory = pickle.load(f)

# ------------------------------------------------------------------
# -------------------- Clean data --------------------
if clean_data:
    data = data_operations.clean_data_dfs(
        dfs=dfs_per_directory
    )

    data.df_data.to_pickle("data/bicycles/df_bicycles_preprocesed.pk1")
else:
    data.df_data = pd.read_pickle("data/bicycles/df_bicycles_preprocesed.pk1")

print()
print("+-"*60)
print("\t\t ANALISYS")

counts = data.df_data['day_type'].value_counts().reindex(["Normal", "Holiday", "Weekend"], fill_value=0)
print(counts)

# Como DataFrame legible
counts_df = counts.reset_index()
counts_df.columns = ['day_type', 'n_trips']
counts_df['pct'] = counts_df['n_trips'] / counts_df['n_trips'].sum() * 100
print(counts_df)

print("Dimension del DF")
print(data.df_data.shape)

# Supongamos que ya tienes counts_df como en tu ejemplo
# counts_df:
#   day_type    n_trips    pct

plt.figure(figsize=(8,5))
sns.barplot(x='day_type', y='n_trips', data=counts_df, palette='Set2')

# Añadir valores encima de cada barra
for index, row in counts_df.iterrows():
    plt.text(x=index, y=row['n_trips']+0.02*row['n_trips'], s=f"{row['n_trips']}", ha='center')

plt.title("Cantidad de viajes por tipo de día")
plt.ylabel("Número de viajes")
plt.xlabel("Tipo de día")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.pie(counts_df['n_trips'], labels=counts_df['day_type'], autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title("Proporción de viajes por tipo de día")
plt.show()

#HACER ANALISIS POR SEPARADO DE CADA AÑO

#REVISAR EVENTOS DEPORTIVOS NFL NBA NHL --> igul son muchos