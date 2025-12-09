import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="FlightOnTime DS API")

# --- 1. CARGA DEL MODELO ---
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "flight_model_v1.joblib")

model = None
encoders = {}

try:
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    encoders['companhia'] = artifacts['le_companhia']
    encoders['origem'] = artifacts['le_origem']
    encoders['destino'] = artifacts['le_destino']
    print(f"✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error cargando el modelo: {e}")

# --- 2. NUEVO CONTRATO DE DATOS (Sin distancia) ---
class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str # Esperamos formato ISO

def safe_transform(encoder, value):
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return 0

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict_flight(flight: FlightInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        # Procesar fecha (Pandas maneja ISO 8601 y UTC automáticamente)
        fecha = pd.to_datetime(flight.data_partida)
        
        input_data = pd.DataFrame([{
            'companhia_encoded': safe_transform(encoders['companhia'], flight.companhia),
            'origem_encoded': safe_transform(encoders['origem'], flight.origem),
            'destino_encoded': safe_transform(encoders['destino'], flight.destino),
            'hora_partida': fecha.hour,
            'dia_semana': fecha.dayofweek,
            'mes': fecha.month
        }])
        
        # Predecir
        prob = model.predict_proba(input_data)[0][1]
        
        # Lógica Booleana (A petición de Ricardo)
        es_atrasado = bool(prob > 0.5)
        
        return {
            "atrasado": es_atrasado,      # Devuelve true/false
            "probabilidade": round(float(prob), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)