import httpx
from app.config import settings


TOOL_DEFINITION = {
    "name": "predict_flight_delay",
    "description": (
        "Predicts the probability of a flight delay for a Brazilian domestic flight. "
        "Uses a CatBoost ML model with real-time weather data from OpenMeteo. "
        "Returns delay probability, risk level (green/yellow/red), and weather conditions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "airline": {
                "type": "string",
                "description": "Airline name (e.g., 'GOL', 'TAM', 'AZUL')",
            },
            "origin": {
                "type": "string",
                "description": "Origin airport full name (e.g., 'Congonhas', 'Guarulhos')",
            },
            "destination": {
                "type": "string",
                "description": "Destination airport full name (e.g., 'Santos Dumont', 'Galeão')",
            },
            "departure_datetime": {
                "type": "string",
                "description": "Departure date and time in ISO format (e.g., '2025-12-24T14:00:00')",
            },
        },
        "required": ["airline", "origin", "destination", "departure_datetime"],
    },
}


async def execute(
    airline: str,
    origin: str,
    destination: str,
    departure_datetime: str,
) -> dict:
    payload = {
        "companhia": airline,
        "origem": origin,
        "destino": destination,
        "data_partida": departure_datetime,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.flightontime_ds_url}/predict",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "prediction": data.get("previsao", "UNKNOWN"),
                "probability": data.get("probabilidade", 0.0),
                "risk_color": data.get("cor", "unknown"),
                "details": {
                    "distance_km": data.get("dados_utilizados", {}).get("distancia", 0),
                    "precipitation_mm": data.get("dados_utilizados", {}).get("chuva", 0),
                    "wind_speed_kmh": data.get("dados_utilizados", {}).get("vento", 0),
                    "weather_source": data.get("dados_utilizados", {}).get("fonte_clima", "unknown"),
                },
            }
    except httpx.HTTPStatusError as e:
        return {"error": f"Prediction service returned {e.response.status_code}", "details": str(e)}
    except httpx.ConnectError:
        return {"error": "Could not connect to prediction service", "details": f"URL: {settings.flightontime_ds_url}"}
