# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import json

pd.set_option("display.max_rows", None)   # Muestra todas las filas
pd.set_option("display.max_columns", None)  # Muestra todas las columnas
pd.set_option("display.width", None)     # No corta la tabla en varias lineas
pd.set_option("display.max_colwidth", None)  # Muestra el contenido de celdas completo

version = "7_6_fine_tunning"

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

# Se divide la clase 1 en varias sublclases
df_class_1 = df_agg[df_agg['n_viajes'] == 1]

with open("../../data/le_station_encoder.pkl", "rb") as f:
    le_station = pickle.load(f)
    
num_stations = len(le_station.classes_)

df_class_1 = df_agg[df_agg['n_viajes'] == 1].copy()

def assign_subclass(row):
    #if row['start_station_idx'] < (num_stations / 2):
    #    return 1  # Estaciones "bajas"
    #else:
    # Estaciones "altas", subdividir por hora
    h = row['started_minute'].hour
    if h <= 10:
        return 1
    elif h <= 17:
        return 2
    else:
        return 3

df_agg.loc[df_class_1.index, 'subclass'] = df_class_1.apply(assign_subclass, axis=1)

mask = df_agg['n_viajes'] > 1
df_agg.loc[mask, 'subclass'] = df_agg.loc[mask, 'n_viajes'] + 2

mask = df_agg['subclass'] >= 4
df_agg.loc[mask, 'subclass'] = 4

# convertir a entero
df_agg['subclass'] = df_agg['subclass'].astype(int)

# Simulacion con SMOTE de clase minoritarias

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- 1. Separar features y target ---
df = df_agg.copy()

data_raw = df.drop(columns=['started_minute'])  # quitar datetime
y = df['subclass'].astype(int)

print("Antes:", Counter(y))

# --- 3. Ajustar estrategia SMOTE ---
# Como tus clases 7 y 8 tienen muy pocos ejemplos,
# usamos k_neighbors=2 y fijamos targets razonables.

target_sizes = {
    4: 1_000_000,
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
df_smote['subclass'] = y_smote

# FIN SMOTE

# Se obtienen los datos
df_smote = df_agg.copy()
df_smote = df_smote.drop(columns=['started_minute'])  # quitar datetime

x_context = df_smote.drop(columns=[
    "n_viajes",
    #"started_minute",
    "start_station_idx",
    "end_station_idx",
    "subclass",
])

x_start = df_smote[[
    "start_station_idx",
]]

x_end = df_smote[[
    "end_station_idx",
]]

y = df_smote["subclass"]

with open("../../data/le_station_encoder.pkl", "rb") as f:
    le_station = pickle.load(f)
    
len(le_station.classes_)

# El target empieza en 1 ? lo pasamos a 0
y_cls = y - 1

#num_classes = y_cls.max() + 1
num_classes = int(y_cls.max()) + 1
print("Clases totales:", num_classes)

from sklearn.model_selection import train_test_split

x_ctx_train, x_ctx_test, x_start_train, x_start_test, x_end_train, x_end_test, y_train, y_test = \
    train_test_split(x_context, x_start, x_end, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

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

from tensorflow.keras.utils import to_categorical

#y_train_oh = to_categorical(y_train, num_classes=num_classes)
#y_val_oh   = to_categorical(y_val,   num_classes=num_classes)
#y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

num_stations = len(le_station.classes_)
num_features = x_context.shape[1]

# Se identifica la clase minoritaria
#minor_class = int(np.argmin(np.bincount(y_train)))
#print("-------------------------------------------------------------------------------------")
#print("Clase minoritaria:", minor_class)


def categorical_focal_loss(gamma=2.5, alpha=0.9):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        ce = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred
        )

        p_t = tf.exp(-ce)
        fl = alpha * tf.pow(1.0 - p_t, gamma) * ce
        return fl
    return loss

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

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model([input_start, input_end, input_context], out)

    model.compile(
        optimizer=Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    #model.compile(
    #    optimizer=Adam(lr),
    #    loss=categorical_focal_loss(gamma=2.5, alpha=0.9),
    #    metrics=[
    #        tf.keras.metrics.Recall(class_id=minor_class, name="recall"),
    #        tf.keras.metrics.Precision(class_id=minor_class, name="precision"),
    #    ],
    #)  

    return model
    
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=15,
    executions_per_trial=2,
    directory="kt_tuning_with_SMOTE",
    project_name="stations_context_bayes"
)
#tuner = kt.BayesianOptimization(
#    build_model,
#    objective=kt.Objective("val_recall", direction="max"),
#    max_trials=15,
#    executions_per_trial=1, #TODO cambiar a 2
#    directory="kt_tuning_categorical_focal_loss",
#    project_name="stations_context_bayes"
#)

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

#callbacks = [
#    tf.keras.callbacks.EarlyStopping(
#        monitor="val_recall",
#        mode="max",
#        patience=5,
#        restore_best_weights=True,
#    ),
#    tf.keras.callbacks.ReduceLROnPlateau(
#        monitor="val_loss",
#        factor=0.5,
#        patience=3,
#        min_lr=1e-6,
#    ),
#]


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
    batch_size=8192,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)
#tuner.search(
#    {
#        "start_station": x_start_train,
#        "end_station": x_end_train,
#        "context": x_ctx_train
#    },
#    y_train_oh,
#    validation_data=(
#        {
#            "start_station": x_start_val,
#            "end_station": x_end_val,
#            "context": x_ctx_val
#        },
#        y_val_oh
#    ),
#    epochs=30,
#    batch_size=8192,
#    callbacks=callbacks,
#    class_weight=class_weights,
#    verbose=1
#)

# Se obtienen los mejores hiperparametros
best_hp = tuner.get_best_hyperparameters(1)[0]

print("Mejores hiperparametros:")
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
    batch_size=8192,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Grafico
import matplotlib.pyplot as plt

def show_history(history, model_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))  # 2 filas, 1 columna

    # Perdida (loss)
    ax1.plot(history.history['loss'],     label='Training Loss',  color='green')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    ax1.set_title('Loss evolution')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Precision (accuracy)
    ax2.plot(history.history['accuracy'],     label='Training Accuracy',  color='green')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
    ax2.set_title('Accuracy evolution')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Titulo global
    fig.suptitle(model_name, fontsize=16)

    # Ajustar margenes
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
show_history(history, "Model 1 fine tunning")

# Evaluacion del modelo
y_prob = best_model.predict(
    {
        "start_station": x_start_test,
        "end_station": x_end_test,
        "context": x_ctx_test
    },
    batch_size=8192
)

#y_pred = np.argmax(y_prob, axis=1)

#from sklearn.metrics import classification_report, confusion_matrix
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))

#print(dict(zip(best_model.metrics_names, test_metrics)))

best_model.save(f"model_{version}.keras")

with open(f'history_{version}.json', 'w') as f:
    json.dump(history.history, f)
    
# Predicciones como probabilidades
y_pred_probs = best_model.predict({
    'start_station': x_start,
    'end_station': x_end,
    'context': x_context
}, batch_size=8192)

# Elegimos la clase con mayor probabilidad
y_pred_classes = np.argmax(y_pred_probs, axis=1)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
acc = accuracy_score(y_cls, y_pred_classes)
print("Accuracy test:", acc)

# Reporte detallado por clase
print(classification_report(y_cls, y_pred_classes))

# Matriz de confusion
cm = confusion_matrix(y_cls, y_pred_classes)
print(cm)

# Se elimina la extension de la clase 1
y_cls.unique()
np.unique(y_pred_classes)
conteo_clases = y_cls.value_counts()
print(conteo_clases)

mapping = {
    0: 1,
    1: 1,
    2: 1,
    3: 2
}

y_cls_real = y_cls.map(mapping)
y_pred_real = np.vectorize(lambda x: mapping.get(x, x))(y_pred_classes)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
acc = accuracy_score(y_cls_real, y_pred_real)
print("Accuracy test:", acc)

# Reporte detallado por clase
print(classification_report(y_cls_real, y_pred_real))

# Matriz de confusin
cm = confusion_matrix(y_cls_real, y_pred_real)
print(cm)

# Grafico
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizacion con heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=set(y_cls_real), yticklabels=set(y_cls_real))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()