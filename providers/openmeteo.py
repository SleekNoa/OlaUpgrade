# providers/openmeteo.py
"""
Open-Meteo Weather Provider
===========================

This module fetches real current weather data for any city using the Open-Meteo API.

Why this exists:
- Keeps weather lookup logic separate from the MCP tool layer
- Returns a small structured result the tool can expose cleanly
- Avoids printing debug text to stdout, which would break MCP stdio JSON-RPC
- Adds fallback geocoding for U.S. state abbreviations like "Marion, IA"
- Emits diagnostics through logging/stderr so failures are visible but MCP stays valid
"""

import logging
from dataclasses import dataclass

import httpx


log = logging.getLogger("weather-provider")


@dataclass
class WeatherResult:
    """Normalized weather payload returned to the MCP tool."""
    temp_c: float
    conditions: str
    provider: str = "openmeteo"


# Official WMO weather code descriptions used by Open-Meteo.
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
}


# Used to match common U.S. city inputs like "Marion, IA".
US_STATE_ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


def _split_us_city(city: str) -> tuple[str, str | None]:
    """Split inputs like 'Marion, IA' into city and state abbreviation."""
    parts = [part.strip() for part in city.split(",", 1)]
    if len(parts) != 2:
        return city.strip(), None

    state_abbr = parts[1].upper()
    if state_abbr in US_STATE_ABBR:
        return parts[0], state_abbr
    return city.strip(), None


def _pick_best_result(results: list[dict], state_abbr: str | None) -> dict | None:
    """
    Prefer an exact U.S. state match when the input included a state abbreviation.

    Open-Meteo can return multiple fuzzy matches. This helps choose the intended city.
    """
    if not results:
        return None
    if not state_abbr:
        return results[0]

    state_name = US_STATE_ABBR[state_abbr].lower()

    for result in results:
        if result.get("country_code") == "US" and (result.get("admin1") or "").lower() == state_name:
            return result

    for result in results:
        if result.get("country_code") == "US":
            return result

    return results[0]


def _build_search_variants(city: str) -> list[tuple[str, dict, str | None]]:
    """Build a few increasingly-specific geocoding strategies."""
    city_name, state_abbr = _split_us_city(city)
    variants: list[tuple[str, dict, str | None]] = [
        (
            "original_query",
            {"name": city, "count": 10, "language": "en", "format": "json"},
            state_abbr,
        ),
    ]

    if state_abbr:
        state_name = US_STATE_ABBR[state_abbr]
        variants.extend([
            (
                "us_filtered_city_only",
                {"name": city_name, "count": 10, "language": "en", "format": "json", "countryCode": "US"},
                state_abbr,
            ),
            (
                "expanded_state_name",
                {"name": f"{city_name}, {state_name}", "count": 10, "language": "en", "format": "json", "countryCode": "US"},
                state_abbr,
            ),
            (
                "expanded_state_and_country",
                {"name": f"{city_name}, {state_name}, United States", "count": 10, "language": "en", "format": "json", "countryCode": "US"},
                state_abbr,
            ),
        ])

    return variants


def _geocode_city(city: str) -> tuple[float, float] | None:
    """
    Resolve a city name to coordinates using a few fallback query forms.
    """
    for label, params, state_abbr in _build_search_variants(city):
        log.info("Geocoding attempt '%s' for %s", label, city)
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params=params,
            timeout=10,
        )
        geo.raise_for_status()
        payload = geo.json()
        results = payload.get("results") or []
        log.info("Geocoding attempt '%s' returned %s result(s)", label, len(results))
        best = _pick_best_result(results, state_abbr)
        if best:
            log.info(
                "Selected geocoding result: %s, %s, %s",
                best.get("name"),
                best.get("admin1"),
                best.get("country_code"),
            )
            return best["latitude"], best["longitude"]

    log.warning("No geocoding match found for %s", city)
    return None


def get_weather(city: str) -> WeatherResult | None:
    """
    Fetch current weather for a city.

    Flow:
    1. Geocode the city name to latitude/longitude.
    2. Request current temperature + weather code.
    3. Convert the weather code into readable text.
    4. Return a WeatherResult, or None if anything fails.
    """
    try:
        coords = _geocode_city(city)
        if not coords:
            return None

        lat, lon = coords
        log.info("Fetching forecast for coordinates (%s, %s)", lat, lon)
        wx = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weathercode",
                "temperature_unit": "celsius",
            },
            timeout=10,
        )
        wx.raise_for_status()
        current = wx.json()["current"]
        log.info("Forecast payload keys: %s", sorted(current.keys()))

        return WeatherResult(
            temp_c=current["temperature_2m"],
            conditions=WMO_CODES.get(current["weathercode"], f"Code {current['weathercode']}"),
        )
    except Exception:
        # Log to stderr so we can diagnose the failure without corrupting MCP stdout.
        log.exception("Weather lookup failed for %s", city)
        return None
