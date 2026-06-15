# Stockseer

A full-stack web app that predicts next-day stock prices using machine learning trained on historical market data. Enter any stock ticker, get a forecast, and visualize actual vs. predicted closing prices on an interactive chart.

**Live demo:** _coming soon_

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
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Backend | FastAPI, scikit-learn, yfinance |
| External API | Twelve Data API |
| Deployment | Vercel (frontend), Render / Railway (backend) |

---

## How It Works

1. The user searches for a stock ticker. The frontend calls the Twelve Data API for autocomplete suggestions.
2. On selection, the backend fetches 2 years of daily closing prices via `yfinance`.
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

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Create a `.env.local` file in the frontend directory:

```
NEXT_PUBLIC_TWELVE_DATA_API_KEY=your_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
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
