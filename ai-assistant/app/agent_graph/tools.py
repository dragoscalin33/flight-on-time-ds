from langchain_core.tools import tool

from app.tools import airport_lookup, flight_predictor, weather_checker


@tool
async def predict_flight_delay(
    airline: str,
    origin: str,
    destination: str,
    departure_datetime: str,
) -> dict:
    """Predict the delay probability of a Brazilian domestic flight.

    Uses a CatBoost ML model with real-time weather data from OpenMeteo.
    Returns delay probability, risk level (green/yellow/red), and weather conditions.

    Args:
        airline: Airline name (e.g., 'GOL', 'TAM', 'AZUL').
        origin: Origin airport full name (e.g., 'Congonhas', 'Guarulhos').
        destination: Destination airport full name (e.g., 'Santos Dumont', 'Galeão').
        departure_datetime: Departure date and time in ISO format (e.g., '2025-12-24T14:00:00').
    """
    return await flight_predictor.execute(
        airline=airline,
        origin=origin,
        destination=destination,
        departure_datetime=departure_datetime,
    )


@tool
async def lookup_airport(query: str) -> dict:
    """Look up airport information from the FlightOnTime database.

    Searches by IATA code (e.g., 'GRU') or partial name (e.g., 'Congonhas').
    Returns IATA code and full airport name.

    Args:
        query: IATA code or partial airport name to search for.
    """
    return await airport_lookup.execute(query=query)


@tool
async def check_weather(
    latitude: float,
    longitude: float,
    date: str | None = None,
) -> dict:
    """Check current or forecast weather conditions at a location using OpenMeteo.

    Useful for understanding weather-related delay factors at airports.
    Returns temperature, precipitation, wind speed, and weather code.

    Args:
        latitude: Latitude of the location (e.g., -23.43 for Guarulhos).
        longitude: Longitude of the location (e.g., -46.47 for Guarulhos).
        date: Date in YYYY-MM-DD format; defaults to today.
    """
    return await weather_checker.execute(latitude=latitude, longitude=longitude, date=date)


AGENT_TOOLS = [predict_flight_delay, lookup_airport, check_weather]
