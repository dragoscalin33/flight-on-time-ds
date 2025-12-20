import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import holidays
import catboost # Necessário para o pickle carregar o objeto
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINIÇÃO DA CLASSE SAFE ENCODER ---
class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = {}
        self.unknown_token = -1

    def fit(self, y):
        unique_labels = pd.Series(y).unique()
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        return pd.Series(y).apply(lambda x: self.classes_.get(str(x), self.unknown_token))

app = FastAPI(title="FlightOnTime AI Service (V4 - Weather Aware)")

# --- 2. CARGA DE ARTEFATOS ---
# [MUDANÇA V4] Nome do arquivo atualizado
MODEL_FILENAME = "flight_classifier_v4.joblib"
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, MODEL_FILENAME)

artifacts = None
br_holidays = holidays.Brazil()

try:
    print(f" Carregando modelo de: {model_path}")
    artifacts = joblib.load(model_path)
    
    model = artifacts['model']
    encoders = artifacts['encoders']
    expected_features = artifacts.get('features', [])
    metadata = artifacts.get('metadata', {})
    
    # [MUDANÇA V4] Recupera o threshold salvo (0.40)
    THRESHOLD = metadata.get('threshold', 0.40)
    
    print(f"✅ Modelo V4 carregado! Versão: {metadata.get('versao')}")
    print(f" Threshold configurado: {THRESHOLD}")
    print(f" Features esperadas: {expected_features}")

except Exception as e:
    print(f" ERRO CRÍTICO ao carregar modelo: {e}")
    model = None
    THRESHOLD = 0.40

# --- 3. MODELO DE DADOS (INPUT) ---
class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str  # ISO Format: "2023-12-25T14:00:00"
    distancia_km: float
    # [MUDANÇA V4] Campos de clima opcionais (default para bom tempo)
    precipitation: float = 0.0
    wind_speed: float = 5.0

# --- 4. ENDPOINT DE PREDIÇÃO ---
@app.post("/predict")
def predict_flight(flight: FlightInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor")

    try:
        # A. Processar Data e Feriado
        dt = pd.to_datetime(flight.data_partida)
        is_holiday = 1 if dt.date() in br_holidays else 0

        # B. Criar DataFrame base
        # [MUDANÇA V4] Incluindo precipitation e wind_speed
        input_dict = {
            'companhia': [str(flight.companhia)],
            'origem': [str(flight.origem)],
            'destino': [str(flight.destino)],
            'distancia_km': [float(flight.distancia_km)],
            'hora': [dt.hour],
            'dia_semana': [dt.dayofweek],
            'mes': [dt.month],
            'is_holiday': [is_holiday],
            'precipitation': [float(flight.precipitation)],
            'wind_speed': [float(flight.wind_speed)]
        }
        df_input = pd.DataFrame(input_dict)

        # C. Aplicar Encoders
        for col in ['companhia', 'origem', 'destino']:
            if col in encoders:
                # O encoder já trata valores desconhecidos retornando -1
                df_input[f'{col}_encoded'] = encoders[col].transform(df_input[col])
            else:
                df_input[f'{col}_encoded'] = -1

        # D. Garantir ordem das features (Muito importante no CatBoost)
        X_final = df_input[expected_features]
        
        # E. Predição
        prob = float(model.predict_proba(X_final)[0][1])
        
        # F. Lógica de Semáforo (Ajustada para V4)
        if prob < THRESHOLD:
            status = "PONTUAL"
            risco = "BAIXO"
            msg = "Voo com boas condições operacionais."
        elif THRESHOLD <= prob < 0.60:
            status = "ALERTA"
            risco = "MEDIO"
            msg = f"Risco operacional detectado ({prob:.1%}). Monitore."
        else: # >= 0.60
            status = "ATRASADO"
            risco = "ALTO"
            msg = f"Alta probabilidade de atraso ({prob:.1%}) devido a condições adversas."

        return {
            "previsao": status,
            "probabilidade": round(prob, 4),
            "nivel_risco": risco,
            "mensagem": msg,
            "detalhes": {
                "clima": {
                    "chuva": f"{flight.precipitation}mm",
                    "vento": f"{flight.wind_speed}km/h"
                },
                "is_feriado": bool(is_holiday),
                "threshold_usado": THRESHOLD
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)