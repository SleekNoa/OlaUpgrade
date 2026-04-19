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
    Pick the best geocoding result with strong preference for the requested state.
    """
    if not results:
        log.info("No geocoding results to choose from")
        return None

    if not state_abbr:
        log.info("No state provided, returning first result")
        return results[0]

    state_name = US_STATE_ABBR[state_abbr].lower()
    state_abbr_lower = state_abbr.lower()

    log.info("Scoring %d results for state '%s' (%s)", len(results), state_abbr, state_name)

    scored = []
    for idx, r in enumerate(results, 1):
        name = r.get("name", "Unknown")
        admin1 = (r.get("admin1") or "").lower().strip()
        pop = int(r.get("population") or 0)

        score = 0

        if r.get("country_code") == "US":
            # Strong exact state match
            if admin1 == state_name:
                score += 40
            elif state_abbr_lower in admin1 or state_name in admin1:
                score += 20
            # Bonus if state abbreviation appears in the city name
            if state_abbr_lower in name.lower():
                score += 12

            # Penalty for wrong state
            if admin1 and admin1 != state_name:
                score -= 10

            # Population bonus
            score += min(pop / 8000, 15)

        scored.append((score, -pop, idx, r))

        # Detailed logging
        log.info(
            f"  [{idx:2d}] {name:18} | {admin1.title():15} | pop: {pop:7,} | score: {score:6.1f}"
        )

    # Sort: best score first, then highest population
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    best = scored[0][3]
    best_pop = best.get("population") or 0

    log.info("→ WINNER: %s, %s (pop: %s, score: %.1f)",
             best.get("name"), best.get("admin1"), f"{best_pop:,}", scored[0][0])

    return best


def _build_search_variants(city: str) -> list[tuple[str, dict, str | None]]:
    """Build geocoding strategies optimized for Open-Meteo's quirks."""
    city_name, state_abbr = _split_us_city(city)

    variants: list[tuple[str, dict, str | None]] = [
        # Best strategy: search only city name → Open-Meteo returns all matches including Iowa
        (
            "city_only",
            {"name": city_name, "count": 20, "language": "en", "format": "json"},
            state_abbr,
        ),
    ]

    if state_abbr:
        state_name = US_STATE_ABBR[state_abbr]
        variants.extend([
            # Try with full state name
            (
                "city_state_name",
                {"name": f"{city_name}, {state_name}", "count": 10, "language": "en", "format": "json"},
                state_abbr,
            ),
            # Original query as last resort
            (
                "original_query",
                {"name": city, "count": 10, "language": "en", "format": "json"},
                state_abbr,
            ),
        ])

    return variants


def _geocode_city(city: str) -> tuple[float, float] | None:
    """Resolve a city name to coordinates using multiple fallback strategies."""
    for label, params, state_abbr in _build_search_variants(city):
        log.info("Geocoding attempt '%s' for %s", label, city)

        try:
            geo = httpx.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
                timeout=10,
            )
            geo.raise_for_status()
            payload = geo.json()
            results = payload.get("results") or []

            log.info("Geocoding attempt '%s' returned %s result(s)", label, len(results))

            if results:
                best = _pick_best_result(results, state_abbr)
                if best:
                    log.info(
                        "Selected geocoding result: %s, %s, %s",
                        best.get("name"),
                        best.get("admin1"),
                        best.get("country_code"),
                    )
                    return best["latitude"], best["longitude"]
        except Exception as e:
            log.warning("Geocoding attempt '%s' failed: %s", label, e)
            continue

    log.warning("No geocoding match found for %s", city)
    return None


def get_weather(city: str) -> WeatherResult | None:
    """
    Fetch current weather for a city.
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
        log.exception("Weather lookup failed for %s", city)
        return None