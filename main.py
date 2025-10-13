from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests

load_dotenv()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
TWELVE_BASE = "https://api.twelvedata.com"

app = FastAPI(title="Stockseer API")

# Allow the frontend to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "has-twelve_key": bool(TWELVE_DATA_KEY)}

@app.get("/search")
def search(q: str):
    """
    Autocomplete company/symbol using Twelve Data's symbol_search.
    Return a compact array of {symbol, name, exchange}.
    """
    if not q:
        return []
    
    if not TWELVE_DATA_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_DATA_KEY is not set in .env")
    
    try:
        r = requests.get(
            f"{TWELVE_BASE}/symbol_search",
            params={"symbol": q, "apikey": TWELVE_DATA_KEY},
            timeout=10,
        )
        js = r.json()
        data = js.get("data") or []
        # Normalize the payload to what our UI expects
        out = []
        for d in data:
            sym = d.get("symbol")
            if not sym:
                continue
            out.append({
                "symbol": sym,
                "name": d.get("instrument_name") or d.get("name"),
                "exchange": d.get("exchange"),
            })
        return out[:10]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {e}")
    
@app.get("/ohlc")
def ohlc(symbol: str):
    """
    Latest daily OHLC for the given symbol via Twelve Data.
    Example: /ohlc?symbol=AAPL
    """
    if not TWELVE_DATA_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_DATA_KEY is not set in .env")

    try:
        r = requests.get(
            f"{TWELVE_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": 1,   # latest bar only
                "apikey": TWELVE_DATA_KEY,
            },
            timeout=10,
        )
        js = r.json()

        # If the API returns an error message, bubble it up to help debugging
        if "values" not in js or not js["values"]:
            raise HTTPException(status_code=400, detail=js)

        v = js["values"][0]
        return {
            "date": v["datetime"],
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {e}")
