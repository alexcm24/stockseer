# Stockseer

A full-stack web app that predicts next-day stock prices using machine learning trained on historical market data. Enter any stock ticker, get a forecast, and visualize actual vs. predicted closing prices on an interactive chart.

**Live demo:** [stockseer-three.vercel.app](https://stockseer-three.vercel.app/)
**API:** [stockseer-api.onrender.com](https://stockseer-api.onrender.com/health)

> The backend is hosted on Render's free tier, which sleeps after ~15 minutes of inactivity. The first request after a nap may take 30–60 seconds to wake the server.

---

## Features

- Ticker autocomplete with debounce and caching (Twelve Data API)
- Next-day price forecast using linear regression trained on 2 years of closing prices
- Interactive area chart showing actual vs. predicted closes (Recharts)
- Bullish/bearish color themes based on prediction direction
- Light and dark mode toggle
- FastAPI backend served separately from the Next.js frontend

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS, Recharts |
| Backend | FastAPI, scikit-learn, pandas, NumPy |
| External API | Twelve Data API (search, OHLC, and historical closes) |
| Deployment | Vercel (frontend), Render (backend) |

---

## How It Works

1. The user searches for a stock ticker. The frontend calls the backend's `/search` endpoint, which proxies Twelve Data's symbol search for autocomplete suggestions.
2. On selection, the backend fetches ~2 years of daily closing prices from the Twelve Data `time_series` endpoint.
3. A linear regression model is trained on that data and predicts the next closing price.
4. The prediction and historical prices are returned to the frontend and rendered as a chart.

---

## Local Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Create a `.env` file in the `backend` directory with your [Twelve Data](https://twelvedata.com/) API key:

```
TWELVE_DATA_KEY=your_key_here
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Create a `.env.local` file in the frontend directory pointing at your local backend:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Project Structure

```
stockseer/
├── backend/
│   ├── main.py          # FastAPI app and prediction logic
│   └── requirements.txt
└── frontend/
    ├── app/             # Next.js app router
    ├── components/      # Chart, search, theme toggle
    └── lib/             # API helpers
```

---

## Author

Alex Canizares — [LinkedIn](https://www.linkedin.com/in/canizaresalex/) · [Portfolio](https://my-portfolio-alexcanizares.vercel.app/)
