import httpx
from app.config import settings


TOOL_DEFINITION = {
    "name": "lookup_airport",
    "description": (
        "Looks up airport information from the FlightOnTime database. "
        "Can search by IATA code (e.g., 'GRU') or partial name (e.g., 'Congonhas'). "
        "Returns IATA code and full airport name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "IATA code or partial airport name to search for",
            },
        },
        "required": ["query"],
    },
}


async def execute(query: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.flightontime_backend_url}/airports")
            response.raise_for_status()
            airports = response.json()

            query_lower = query.lower()
            matches = [
                airport
                for airport in airports
                if query_lower in airport.get("iataCode", "").lower()
                or query_lower in airport.get("fullName", "").lower()
            ]

            if not matches:
                return {"found": False, "message": f"No airports found matching '{query}'"}

            return {
                "found": True,
                "count": len(matches),
                "airports": [
                    {"iata_code": a.get("iataCode"), "full_name": a.get("fullName")}
                    for a in matches[:10]
                ],
            }
    except httpx.ConnectError:
        return {"error": "Could not connect to backend service", "details": f"URL: {settings.flightontime_backend_url}"}
