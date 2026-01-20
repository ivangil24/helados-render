import os
import pickle
from flask import Flask, request, jsonify

# Creamos la aplicación Flask
app = Flask(__name__)

# Cargamos el modelo entrenado desde el archivo .pkl
with open("modelo_helados.pkl", "rb") as f:
    modelo = pickle.load(f)

# Endpoint raíz: información básica de la API
@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "mensaje": "API REST - Prediccion de ventas de helados",
        "endpoint": "/predict",
        "metodo": "POST",
        "ejemplo_entrada": {
            "temperatura": 28
        }
    })

# Endpoint de comprobación (health check)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"estado": "ok"})

# Endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    # Obtener datos en formato JSON
    datos = request.get_json()

    # Comprobar que llega el campo temperatura
    if datos is None or "temperatura" not in datos:
        return jsonify({
            "error": "Debes enviar un JSON con el campo 'temperatura'"
        }), 400

    # Convertir temperatura a número
    temperatura = float(datos["temperatura"])

    # El modelo espera una tabla (2D): [[valor]]
    prediccion = modelo.predict([[temperatura]])[0]

    # Devolver resultado en JSON
    return jsonify({
        "temperatura": temperatura,
        "prediccion_ventas": round(float(prediccion), 2)
    })

# Arranque de la aplicación
if __name__ == "__main__":
    # Render usa la variable de entorno PORT
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto)
