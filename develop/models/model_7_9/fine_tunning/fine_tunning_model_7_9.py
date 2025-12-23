# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import json

pd.set_option("display.max_rows", None)   # Muestra todas las filas
pd.set_option("display.max_columns", None)  # Muestra todas las columnas
pd.set_option("display.width", None)     # No corta la tabla en varias lineas
pd.set_option("display.max_colwidth", None)  # Muestra el contenido de celdas completo

version = "7_9_fine_tunning"

with open("../../data/df_normalized_dayOfYear.pk1", "rb") as f:
    df_data = pickle.load(f)
    
# Se redonde al minuto mas cercano
df_data["started_minute"] = df_data["started_at"].dt.round("min")

df_agg = df_data.groupby([
    "start_station_idx",
    "end_station_idx",
    "started_minute"
]).agg(
    n_viajes=("ride_id", "count"),
    year=("year", "first"),
    temp_std=("temp_std", "first"),
    wind_std=("wind_std", "first"),
    rel_humidity_std=("rel_humidity_std", "first"),
    precipitation_std=("precipitation_std", "first"),
    snow_depth_std=("snow_depth_std", "first"),
    hour_sin=("hour_sin", "first"),
    hour_cos=("hour_cos", "first"),
    month_sin=("month_sin", "first"),
    month_cos=("month_cos", "first"),
    event=("event", "any"),  # True si al menos un dato es true
    normal_day=("day_type_Normal", "any"),  # True si al menos un dato es true
    weekend_day=("day_type_Weekend", "any"),  # True si al menos un dato es true
    holiday_day=("day_type_Holiday", "any"),  # True si al menos un dato es true
    doy_sin=("doy_sin", "first"),
    doy_cos=("doy_cos", "first"),
    #member_casual=("member_casual_bool", "any"),  # True si al menos un dato es true
    #classic_bike=("rideable_type_classic_bike", "any"),  # True si al menos un dato es true
    #docked_bike=("rideable_type_docked_bike", "any"),  # True si al menos un dato es true
    #electric_bike=("rideable_type_electric_bike", "any"),  # True si al menos un dato es true
    #duration_min_mean=("duration_min", "mean"),
).reset_index()


# Se agrupan ciertos viajes y se elimina la categoria 1
df_travel_witOut_1 = df_agg[df_agg['n_viajes'] != 1]

df_travel_witOut_1 = df_travel_witOut_1.copy()
df_travel_witOut_1['n_viajes_agg'] = df_travel_witOut_1['n_viajes']
df_travel_witOut_1.loc[df_travel_witOut_1['n_viajes'] >= 5, 'n_viajes_agg'] = 5

# Se aplica SMOTE para igualar las clases
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- 1. Separar features y target ---
df = df_travel_witOut_1.copy()

data_raw = df.drop(columns=['started_minute'])  # quitar datetime
y = df['n_viajes_agg'].astype(int)

print("Antes:", Counter(y))

# --- 3. Ajustar estrategia SMOTE ---
# Como tus clases 7 y 8 tienen muy pocos ejemplos,
# usamos k_neighbors=2 y fijamos targets razonables.

target_sizes = {
    4: 20_000,
    5: 10_000
}

sm = SMOTE(
    sampling_strategy=target_sizes,
    k_neighbors=2,
    random_state=42
)

# --- 4. Aplicar SMOTE ---
data_smote, y_smote = sm.fit_resample(data_raw, y)

print("Despues:", Counter(y_smote))

# 4. Reconstruir DataFrame final
df_smote = pd.DataFrame(data_smote, columns=df_agg.columns.drop("started_minute"))
df_smote['n_viajes_agg'] = y_smote

# Se obtienen los datos
x_context = df_smote.drop(columns=[
    "n_viajes",
    #"started_minute",
    "start_station_idx",
    "end_station_idx",
    "n_viajes_agg"
])

x_start = df_smote[[
    "start_station_idx",
]]

x_end = df_smote[[
    "end_station_idx",
]]

y = df_smote["n_viajes_agg"]

with open("../../data/le_station_encoder.pkl", "rb") as f:
    le_station = pickle.load(f)

from sklearn.model_selection import train_test_split

x_ctx_train, x_ctx_test, x_start_train, x_start_test, x_end_train, x_end_test, y_train, y_test = \
    train_test_split(x_context, x_start, x_end, y, test_size=0.2, random_state=42, stratify=y)

x_ctx_train, x_ctx_val, x_start_train, x_start_val, x_end_train, x_end_val, y_train, y_val = \
    train_test_split(x_ctx_train, x_start_train, x_end_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Se calculan los pesos para equilibrar aun mas las clases
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

classes_present = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_present,
    y=y_train
)

class_weights = {i: w for i, w in zip(classes_present, class_weights)}
print(class_weights)

# CREACION DEL MODELO
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import keras_tuner as kt

num_stations = len(le_station.classes_)
num_features = x_context.shape[1]

def build_model(hp):
    embed_dim = hp.Choice("embed_dim", [int(np.sqrt(num_stations)),
                                        int(1.5*np.sqrt(num_stations)),
                                        int(2*np.sqrt(num_stations))])

    l2_reg = hp.Choice("l2_reg", [1e-5, 1e-4, 1e-3])
    drop_rate = hp.Choice("drop_rate", [0.1, 0.3, 0.5])
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])

    input_start = Input(shape=(1,), name="start_station")
    input_end = Input(shape=(1,), name="end_station")
    input_context = Input(shape=(num_features,), name="context")

    emb_s = layers.Embedding(num_stations, embed_dim,
                             embeddings_regularizer=regularizers.l2(l2_reg))(input_start)
    emb_e = layers.Embedding(num_stations, embed_dim,
                             embeddings_regularizer=regularizers.l2(l2_reg))(input_end)

    x = layers.Concatenate()([
        layers.Flatten()(emb_s),
        layers.Flatten()(emb_e),
        input_context
    ])

    for units in [128, 64, 32]:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(drop_rate)(x)

    out = layers.Dense(1, activation="linear")(x)

    model = Model([input_start, input_end, input_context], out)

    model.compile(
        optimizer=Adam(lr),
        loss="mse",
        metrics=["mae"]
    )

    return model
    
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_mae",
    max_trials=15,
    executions_per_trial=2,
    directory="kt_tuning_with_weights",
    project_name="stations_context_bayes"
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

#Se buscan las mejores variables
tuner.search(
    {
        "start_station": x_start_train,
        "end_station": x_end_train,
        "context": x_ctx_train
    },
    y_train,
    validation_data=(
        {
            "start_station": x_start_val,
            "end_station": x_end_val,
            "context": x_ctx_val
        },
        y_val
    ),
    epochs=30,
    batch_size=1024,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Se obtienen los mejores hiperparametros
best_hp = tuner.get_best_hyperparameters(1)[0]

print("Mejores hiperparaetros:")
for k, v in best_hp.values.items():
    print(f"{k}: {v}")

# Se reentrena el mejor modelo
best_model = tuner.hypermodel.build(best_hp)

history = best_model.fit(
    {
        "start_station": x_start_train,
        "end_station": x_end_train,
        "context": x_ctx_train
    },
    y_train,
    validation_data=(
        {
            "start_station": x_start_val,
            "end_station": x_end_val,
            "context": x_ctx_val
        },
        y_val
    ),
    epochs=30,
    batch_size=1024,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Grafico
import matplotlib.pyplot as plt

def show_history(history, model_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 filas, 1 columna

    # P�rdida (loss)
    ax1.plot(history.history['loss'],     label='Training Loss',  color='green')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Precisi�n (accuracy)
    ax2.plot(history.history['mae'],     label='Training MAE',  color='green')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='blue')
    ax2.set_title('Training vs Validation MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()

    # T�tulo global
    fig.suptitle(model_name, fontsize=16)

    # Ajustar m�rgenes
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
show_history(history, "Model 2 fine tunning")

# Evaluacion del modelo
test_metrics = best_model.evaluate(
    {
        "start_station": x_start_test,
        "end_station": x_end_test,
        "context": x_ctx_test
    },
    y_test,
    verbose=0
)

print(dict(zip(best_model.metrics_names, test_metrics)))

best_model.save(f"model_{version}.keras")

with open(f'history_{version}.json', 'w') as f:
    json.dump(history.history, f)
    
    
# Predicciones con todos los datos
y_pred_probs = best_model.predict({
    'start_station': x_start,
    'end_station': x_end,
    'context': x_context
}, batch_size=1024)

import numpy as np

y_pred_int = np.rint(y_pred_probs).astype(int)  # redondear al entero m�s cercano
y_pred_int = np.clip(y_pred_int, 2, 5)   # asegurar que est� entre 2 y 5

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import matplotlib.pyplot as plt

mae = mean_absolute_error(y, y_pred_int)
mse = mean_squared_error(y, y_pred_int)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Exact match redondeado
acc = accuracy_score(y, y_pred_int)
print(f"Exact match accuracy: {acc*100:.2f}%")

# Matriz de confusi�n
cm = confusion_matrix(y, y_pred_int)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()