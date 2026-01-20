# Importamos pandas para trabajar con datos en forma de tablas (DataFrame)
import pandas as pd

# Importamos pickle para guardar el modelo entrenado en un archivo
import pickle

# Función para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Modelo de regresión lineal (aprendizaje supervisado)
from sklearn.linear_model import LinearRegression

# Métricas para evaluar el modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar datos(CSV separado por punto y coma)
# Leemos el archivo CSV y lo guardamos en un DataFrame
df = pd.read_csv("datos_helados.csv", sep=";")

# Limpiar nombres de columnas (quitar espacios y poner en minúsculas)
df.columns = df.columns.str.strip().str.lower()

print("COLUMNAS:", list(df.columns))
print(df.head())

# Separar X e y
# X → variables de entrada (características)
# En este caso solo usamos la temperatura
# IMPORTANTE: X debe ser una tabla (2D), por eso va entre doble corchete
X = df[["temperatura"]]

# y → variable objetivo (lo que queremos predecir)
# Aquí son las ventas de helados
y = df["ventas"]

# Dividir train/test
# Separamos los datos en:
# - 80% para entrenar el modelo
# - 20% para probarlo
# random_state sirve para que siempre salga la misma división
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
# Creamos el modelo de regresión lineal
modelo = LinearRegression()

# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Evaluar
# Usamos el modelo entrenado para hacer predicciones
preds = modelo.predict(X_test)

# Calculamos las métricas de evaluación
# MAE: error medio absoluto
mae = mean_absolute_error(y_test, preds)

# RMSE: raíz del error cuadrático medio
rmse = mean_squared_error(y_test, preds) ** 0.5

# R2: qué tan bien se ajusta el modelo a los datos
r2 = r2_score(y_test, preds)

# Mostramos los resultados por pantalla
print("RESULTADOS DEL MODELO (TEST):")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 2))

# Guardar modelo con pickle (biblioteca estándar de Python)
# Abrimos un archivo en modo escritura binaria ("wb")
with open("modelo_helados.pkl", "wb") as f:
    # Guardamos el modelo entrenado en el archivo
    pickle.dump(modelo, f)

print("Modelo guardado como modelo_helados.pkl")
