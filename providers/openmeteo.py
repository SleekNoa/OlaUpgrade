# providers/openmeteo.py
"""
Open-Meteo Weather Provider
===========================

This module fetches real current weather data for any city using the **Open-Meteo API**
(free, no API key required).

Architecture:
- Two-step process:
  1. Geocoding: Convert city name → latitude/longitude
  2. Weather fetch: Get current temperature and weather condition

Why this design?
- Open-Meteo is reliable, fast, and doesn't require authentication.
- We use `httpx` for async-friendly HTTP requests (though used synchronously here for simplicity).
- Weather codes are mapped to human-readable descriptions using the official WMO standard.
- Returns a clean dataclass for easy use in the MCP tool.

Error handling:
- Returns `None` on any failure (network issue, city not found, etc.)
- The MCP tool will catch this and raise a user-friendly error.
"""

import httpx
from dataclasses import dataclass
from typing import Optional



@dataclass
class WeatherResult:
    """
    Structured result returned by the weather provider.

    Attributes:
        temp_c: Temperature in Celsius
        conditions: Human-readable weather description (e.g. "Partly cloudy")
        provider: Name of the data source (for transparency)
    """
    temp_c: float
    conditions: str
    provider: str = "openmeteo"


# Official World Meteorological Organization (WMO) weather code descriptions
# Source: https://www.open-meteo.com/en/docs
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    # Add more codes here if needed in the future
}


def get_weather(city: str) -> Optional[WeatherResult]:
    """
    Fetch current weather for a given city using Open-Meteo API.

    Process (Step-by-step logic):
    1. Geocode the city name to get latitude and longitude.
    2. Use those coordinates to fetch current temperature and weather code.
    3. Map the WMO weather code to a readable description.
    4. Return a WeatherResult object or None if anything fails.

    Args:
        city: City name (e.g. "Marion, IA" or "London, UK")

    Returns:
        WeatherResult object on success, None on failure (city not found, network error, etc.)

    Note: No API key is needed. All requests have a 10-second timeout for safety.
    """
    try:
        # ── Step 1: Geocoding - Convert city name to coordinates ──
        print(f"[WeatherProvider] Geocoding city: {city}")  # Helpful debug log

        geo_response = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,  # Only get the best match
                "language": "en",
                "format": "json"
            },
            timeout=10,
        )
        geo_response.raise_for_status()  # Raise error if HTTP status is not 2xx

        results = geo_response.json().get("results")
        if not results:
            print(f"[WeatherProvider] City not found: {city}")
            return None

        # Take the first (best) result
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        # ── Step 2: Fetch current weather using lat/lon ──
        print(f"[WeatherProvider] Fetching weather for coordinates: {lat}, {lon}")

        weather_response = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weathercode",  # Only request what we need
                "temperature_unit": "celsius",
            },
            timeout=10,
        )
        weather_response.raise_for_status()

        current = weather_response.json()["current"]

        # ── Step 3: Map weather code to human-readable text ──
        weather_code = current["weathercode"]
        conditions = WMO_CODES.get(weather_code, f"Unknown (code {weather_code})")

        # ── Step 4: Return structured result ──
        result = WeatherResult(
            temp_c=current["temperature_2m"],
            conditions=conditions,
        )

        print(f"[WeatherProvider] Success: {result.temp_c}°C, {result.conditions}")
        return result

    except httpx.RequestError as e:
        print(f"[WeatherProvider] Network error while fetching weather: {e}")
        return None
    except Exception as e:
        print(f"[WeatherProvider] Unexpected error for city '{city}': {e}")
        return None