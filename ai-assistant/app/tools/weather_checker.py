import httpx
from datetime import datetime

TOOL_DEFINITION = {
    "name": "check_weather",
    "description": (
        "Checks current or forecast weather conditions at a specific location using OpenMeteo API. "
        "Useful for understanding weather-related delay factors at airports. "
        "Returns temperature, precipitation, wind speed, and cloud cover."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude of the location (e.g., -23.43 for Guarulhos)",
            },
            "longitude": {
                "type": "number",
                "description": "Longitude of the location (e.g., -46.47 for Guarulhos)",
            },
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. If not provided, uses today.",
            },
        },
        "required": ["latitude", "longitude"],
    },
}

MAJOR_AIRPORT_COORDS: dict[str, dict[str, float]] = {
    "GRU": {"lat": -23.4356, "lon": -46.4731},
    "CGH": {"lat": -23.6261, "lon": -46.6564},
    "GIG": {"lat": -22.8100, "lon": -43.2506},
    "SDU": {"lat": -22.9105, "lon": -43.1631},
    "BSB": {"lat": -15.8711, "lon": -47.9186},
    "CNF": {"lat": -19.6244, "lon": -43.9719},
    "SSA": {"lat": -12.9086, "lon": -38.3225},
    "REC": {"lat": -8.1264, "lon": -34.9231},
    "CWB": {"lat": -25.5285, "lon": -49.1758},
    "POA": {"lat": -29.9944, "lon": -51.1714},
}


async def execute(
    latitude: float,
    longitude: float,
    date: str | None = None,
) -> dict:
    target_date = date or datetime.now().strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum,wind_speed_10m_max,temperature_2m_max,temperature_2m_min,weathercode",
        "timezone": "America/Sao_Paulo",
        "start_date": target_date,
        "end_date": target_date,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            return {
                "date": target_date,
                "location": {"latitude": latitude, "longitude": longitude},
                "conditions": {
                    "precipitation_mm": daily.get("precipitation_sum", [0])[0],
                    "max_wind_speed_kmh": daily.get("wind_speed_10m_max", [0])[0],
                    "temp_max_c": daily.get("temperature_2m_max", [0])[0],
                    "temp_min_c": daily.get("temperature_2m_min", [0])[0],
                    "weather_code": daily.get("weathercode", [0])[0],
                },
            }
    except httpx.HTTPError as e:
        return {"error": f"Weather API error: {str(e)}"}
