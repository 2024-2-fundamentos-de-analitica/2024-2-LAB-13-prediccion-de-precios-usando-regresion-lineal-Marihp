import pandas as pd
import zipfile

#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.


def clean_dataset(path):
    """Carga y limpia los datasets."""
    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f)

    df.dropna(inplace=True)
    df["Age"] = 2021 - df["Year"]
    df.drop_duplicates()
    df.drop(columns=["Year", "Car_Name"])

    return df


# Cargar datasets
df_test = clean_dataset("./files/input/test_data.csv.zip")
df_train = clean_dataset("./files/input/train_data.csv.zip")

# Separar variables
x_train = df_train.drop(columns=["Present_Price"])
y_train = df_train["Present_Price"]

x_test = df_test.drop(columns=["Present_Price"])
y_test = df_test["Present_Price"]
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import json

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"
):
    """Evalúa el modelo en los conjuntos de entrenamiento y prueba y guarda las métricas."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for dataset, (x, y) in zip(
            ["train", "test"], [(x_train, y_train), (x_test, y_test)]
        ):
            y_pred = model.predict(x)
            metrics_train = {
                "type": "metrics",
                "dataset": "train",
                "r2": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred)),
                "mad": float(median_absolute_error(y_pred, y_train_pred)),
            }

    # Calcular métricas para el conjunto de prueba
    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred)),
    }

    # Crear carpeta si no existe
    output_dir = "../files/output"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar las métricas en un archivo JSON
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")

    print(f"Métricas guardadas en {file_path}")
