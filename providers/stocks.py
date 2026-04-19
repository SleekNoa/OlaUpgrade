# providers/stocks.py
"""Yahoo Finance-backed stock/market data provider for the OLA tool layer."""

from dataclasses import dataclass
import time
import functools
from typing import Optional

import requests
import yfinance as yf
from curl_cffi import requests


@dataclass
class StockResult:
    """Structured market data payload returned to the MCP tool."""
    ticker: str
    price: float | None = None
    currency: str = "USD"
    company: str = ""
    change_pct: float | None = None
    market_cap: str = ""
    error: str | None = None


# Create one reusable session (impersonates Chrome - best balance of compatibility)
_session = requests.Session(impersonate="chrome")


@functools.lru_cache(maxsize=200)
def get_stock(ticker: str) -> StockResult:
    """
    Fetch current stock price and basic company data via yfinance.
    Uses curl_cffi session for better Yahoo compatibility.
    """
    ticker = ticker.upper().strip()

    for attempt in range(3):  # Up to 2 retries
        try:
            start_time = time.time()
            timeout = 12.0

            # Pass the curl_cffi session — this is the proper fix
            stock = yf.Ticker(ticker, session=_session)

            # === Fast path: Use fast_info ===
            try:
                fast_info = stock.fast_info
                price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
                change_pct = fast_info.get('regularMarketChangePercent')
                market_cap = fast_info.get('marketCap')
            except Exception:
                price = None
                change_pct = None
                market_cap = None

            # === Fallback to .info ===
            if price is None or market_cap is None:
                if time.time() - start_time > timeout:
                    return StockResult(ticker=ticker, error="Request timed out")

                try:
                    info = stock.info
                    if price is None:
                        price = (
                            info.get("currentPrice")
                            or info.get("regularMarketPrice")
                            or info.get("previousClose")
                            or info.get("regularMarketPreviousClose")
                        )
                    if change_pct is None:
                        change_pct = info.get("regularMarketChangePercent") or info.get("52WeekChange")
                    if market_cap is None:
                        market_cap = info.get("marketCap")
                except Exception as info_exc:
                    error_str = str(info_exc).lower()
                    if any(x in error_str for x in ["rate limit", "too many requests", "429"]):
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                            continue
                    raise

            if price is None:
                return StockResult(ticker=ticker, error="Price data not available")

            # === Format market cap ===
            if market_cap and isinstance(market_cap, (int, float)):
                if market_cap >= 1_000_000_000_000:
                    market_cap_text = f"${market_cap / 1_000_000_000_000:.2f}T"
                elif market_cap >= 1_000_000_000:
                    market_cap_text = f"${market_cap / 1_000_000_000:.2f}B"
                elif market_cap >= 1_000_000:
                    market_cap_text = f"${market_cap / 1_000_000:.2f}M"
                else:
                    market_cap_text = f"${market_cap:,.0f}"
            else:
                market_cap_text = "N/A"

            # === Get company name ===
            company = ""
            try:
                info = getattr(stock, 'info', {}) or {}
                company = info.get("longName") or info.get("shortName") or ticker
            except Exception:
                company = ticker

            return StockResult(
                ticker=ticker,
                price=round(float(price), 2),
                currency="USD",
                company=company,
                change_pct=round(float(change_pct or 0), 2),
                market_cap=market_cap_text,
            )

        except Exception as exc:
            error_msg = str(exc).lower()
            if any(x in error_msg for x in ["timeout", "connection", "rate limit", "too many requests", "429"]):
                if attempt < 2:
                    sleep_time = 1.5 * (attempt + 1) + 0.5
                    time.sleep(sleep_time)
                    continue
                friendly_msg = "Network timeout or rate limit - please try again later"
            else:
                friendly_msg = str(exc)[:180]

            return StockResult(ticker=ticker, error=friendly_msg)

    return StockResult(ticker=ticker, error="Failed after retries - please try again")