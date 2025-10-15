# Stockseer — Real-Time Stock Price Predictor

A full-stack application that forecasts next-day stock prices using **linear regression** trained on historical market data.  
Built with **FastAPI**, **scikit-learn**, **Next.js 14**, **Tailwind CSS**, and **Recharts**.

---

## Features

- 🔍 **Autocomplete search** (Twelve Data API) with debounce and caching  
- 📈 **Next-day price forecast** using 2 years of closing prices (via `yfinance`)  
- 🎨 **Dynamic chart** with bullish (green) / bearish (red) themes  
- 🌓 **Light / Dark mode** toggle  
- 📊 **Recharts area chart** of actual vs. predicted closes  
- ⚡ **FastAPI + Next.js** full-stack integration  

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Next.js 14 (React + TypeScript), Tailwind CSS, Recharts |
| Backend | FastAPI, scikit-learn, yfinance |
| APIs | Twelve Data API |
| Deployment | Vercel (frontend), Render or Railway (backend) |

---

## 🛠️ Local Setup

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000