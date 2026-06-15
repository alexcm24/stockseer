from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import HTTPException

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




def fetch_closes_td(symbol: str, outputsize: int = 730) -> pd.DataFrame:
    """
    Fetch up to ~2 years of daily closes from Twelve Data.
    Returns a DataFrame with columns ['date', 'close'] sorted oldest -> newest.
    """
    if not TWELVE_DATA_KEY:
        raise HTTPException(status_code=500, detail="TWELVE_DATA_KEY is not set")

    try:
        r = requests.get(
            f"{TWELVE_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": outputsize,
                "order": "ASC",
                "apikey": TWELVE_DATA_KEY,
            },
            timeout=15,
        )
        js = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Twelve Data error: {e}")

    # Twelve Data returns {"status":"error", ...} or no "values" on failure
    if "values" not in js or not js["values"]:
        raise HTTPException(status_code=400, detail=js)

    rows = [
        {"date": v["datetime"], "close": float(v["close"])}
        for v in js["values"]
        if v.get("close") is not None
    ]
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


@app.get("/forecast")
def forecast(symbol: str):
    """
    Linear Regression baseline on ~2 years of daily closes (Twelve Data).
    Returns next-day prediction, MSE, directional accuracy vs naive,
    and the full series for charting.
    """
    sym_u = symbol.upper()
    closes = fetch_closes_td(sym_u)

    # Need enough data to split train/test
    if len(closes) < 30:
        raise HTTPException(400, "Not enough data to train/test a model")

    # Feature: simple time index
    closes["t"] = np.arange(len(closes), dtype=float)
    X = closes[["t"]].values
    y = closes["close"].values

    # Time-aware split: last 20 points or 10%
    split = max(20, int(0.1 * len(closes)))
    if len(closes) < split + 5:
        raise HTTPException(400, "Not enough data to train/test a model")

    X_train, y_train = X[:-split], y[:-split]
    X_test,  y_test  = X[-split:], y[-split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions + MSE
    y_hat = model.predict(X_test)
    mse = float(np.mean((y_hat - y_test) ** 2))

    # Directional accuracy vs naive (tomorrow = today)
    if len(y_test) >= 2:
        actual = y_test[1:]
        model_dir = np.sign(y_hat[1:] - y_test[:-1])
        naive_dir = np.sign(y_test[1:] - y_test[:-1])
        actual_dir = np.sign(actual - y_test[:-1])
        model_acc = float((model_dir == actual_dir).mean())
        naive_acc = float((naive_dir == actual_dir).mean())
    else:
        model_acc = float("nan")
        naive_acc = float("nan")

    # Full-series prediction for charting
    closes["pred"] = model.predict(X)

    # Next day prediction at t_last + 1
    next_t = np.array([[closes["t"].iloc[-1] + 1.0]])
    pred_next = float(model.predict(next_t)[0])
    current_close = float(closes["close"].iloc[-1])

    series = [
        {"date": str(pd.to_datetime(d).date()), "actual": float(a), "predicted": float(p)}
        for d, a, p in zip(closes["date"], closes["close"], closes["pred"])
    ]

    return {
        "symbol": sym_u,
        "current_close": current_close,
        "predicted_close": pred_next,
        "mse": mse,
        "model_acc": model_acc,
        "naive_acc": naive_acc,
        "series": series,
    }
