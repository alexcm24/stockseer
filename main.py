from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
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




def extract_closes(df: pd.DataFrame | pd.Series, sym_u: str) -> pd.DataFrame:
    """
    Normalize yfinance output to a 2-col DataFrame: ['date','close'].
    Handles Series, single-level DataFrame, and MultiIndex (either orientation).
    """
    # 1) Plain Series (rare)
    if isinstance(df, pd.Series):
        s = df.dropna()
        out = s.reset_index()
        out.columns = ["date", "close"]
        return out

    # 2) MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Try level-0 == 'Close'
        try:
            part = df.xs("Close", level=0, axis=1)
            if isinstance(part, pd.Series):
                s = part
            else:
                s = part[sym_u] if sym_u in part.columns else part.iloc[:, 0]
            out = s.dropna().reset_index()
            out.columns = ["date", "close"]
            return out
        except Exception:
            pass
        # Try level-1 == 'Close' (your screenshot shows tuples like ('Close','AAPL'))
        try:
            part = df.xs("Close", level=1, axis=1)
            if isinstance(part, pd.Series):
                s = part
            else:
                s = part[sym_u] if sym_u in part.columns else part.iloc[:, 0]
            out = s.dropna().reset_index()
            out.columns = ["date", "close"]
            return out
        except Exception:
            pass
        raise HTTPException(500, f"Unexpected MultiIndex columns: {df.columns.tolist()}")

    # 3) Single-level DataFrame
    # Standard: columns like ['Open','High','Low','Close','Adj Close','Volume']
    if "Close" in df.columns:
        s = df["Close"].dropna()
        out = s.reset_index()
        out.columns = ["date", "close"]
        return out

    # Lowercase or ticker-named column fallbacks
    cols_lower = [str(c).lower() for c in df.columns]
    if "close" in cols_lower:
        s = df[df.columns[cols_lower.index("close")]].dropna()
        out = s.reset_index()
        out.columns = ["date", "close"]
        return out
    if sym_u in df.columns:
        s = df[sym_u].dropna()
        out = s.reset_index()
        out.columns = ["date", "close"]
        return out
    for c in df.columns:
        if str(c).lower() == sym_u.lower():
            s = df[c].dropna()
            out = s.reset_index()
            out.columns = ["date", "close"]
            return out

    raise HTTPException(500, f"Unexpected columns from yfinance: {df.columns.tolist()}")


@app.get("/forecast")
def forecast(symbol: str):
    """
    Linear Regression baseline on ~2 years of daily closes (yfinance).
    Robust to yfinance output shapes. Returns next-day prediction, MSE,
    directional accuracy vs naive, and full series for charting.
    """
    sym_u = symbol.upper()
    end = datetime.utcnow()
    start = end - timedelta(days=365 * 2 + 14)

    df = yf.download(
        sym_u,
        start=start.date(),
        end=end.date(),
        progress=False,
        auto_adjust=False,  # ensure classic OHLC columns exist
        actions=False,
        group_by="ticker",  # yfinance may still return MultiIndex; we handle it
    )

    closes = extract_closes(df, sym_u)

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
