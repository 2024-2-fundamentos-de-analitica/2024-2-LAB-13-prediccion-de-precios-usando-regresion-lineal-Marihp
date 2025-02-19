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
import os
import json
import zipfile
import pickle
import gzip
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


# Función para limpiar los datos
def clean_dataset(path):
    """Carga y limpia los datasets."""
    with zipfile.ZipFile(path, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("El archivo ZIP no contiene archivos CSV")
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    df.dropna(inplace=True)
    df["Age"] = 2021 - df["Year"]
    df.drop_duplicates(inplace=True)
    df.drop(columns=["Year", "Car_Name"], inplace=True)

    return df


# Cargar datasets
df_train = clean_dataset("./files/input/train_data.csv.zip")
df_test = clean_dataset("./files/input/test_data.csv.zip")

# Separar variables
x_train = df_train.drop(columns=["Present_Price"])
y_train = df_train["Present_Price"]

x_test = df_test.drop(columns=["Present_Price"])
y_test = df_test["Present_Price"]


# Construcción del pipeline
def build_pipeline():
    """Construye un pipeline para el modelo de regresión."""
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", MinMaxScaler(), numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("k_best", SelectKBest(f_regression, k=10)),
            ("model", LinearRegression),
        ]
    )

    return pipeline


# Optimización con GridSearchCV
def optimize_pipeline(pipeline, x_train, y_train):
    param_grid = {
        "k_best__k": range(1, 12),
        "model__fit_intercept": [True, False],
        "model__positive": [True, False],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    print("Optimizando hiperparámetros con GridSearchCV...")
    grid_search.fit(x_train, y_train)
    print("Optimización finalizada.")
    print("Mejores parámetros:", grid_search.best_params_)

    return grid_search


# Guardar el modelo
def save_model(model, file_path="./files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {file_path}")


# Evaluación del modelo
def evaluate_model(
    model, x_train, y_train, x_test, y_test, file_path="./files/output/metrics.json"
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for dataset, (x, y) in zip(
            ["train", "test"], [(x_train, y_train), (x_test, y_test)]
        ):
            y_pred = model.best_estimator_.predict(x)
            metrics = {
                "type": "metrics",
                "dataset": dataset,
                "r2": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred)),
                "mad": float(median_absolute_error(y, y_pred)),
            }
            f.write(json.dumps(metrics) + "\n")

    print(f"Métricas guardadas en {file_path}")


# Ejecución
print("Construcción del pipeline...")
pipeline = build_pipeline()

print("Optimización del modelo...")
best_pipeline = optimize_pipeline(pipeline, x_train, y_train)

print("Guardando el modelo...")
save_model(best_pipeline)

print("Evaluando el modelo y guardando métricas...")
evaluate_model(best_pipeline, x_train, y_train, x_test, y_test)

print("¡Proceso completado con éxito!")
