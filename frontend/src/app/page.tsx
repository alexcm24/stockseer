"use client";

import { useEffect, useMemo, useState } from "react";
import SearchBox from "@/components/SearchBox";
import ForecastChart from "@/components/ForecastChart";
import { ThemeToggle } from "@/components/ThemeToggle";

const API = process.env.NEXT_PUBLIC_BACKEND_URL!;

type Ohlc = { date: string; open: number; high: number; low: number; close: number };
type SeriesPoint = { date: string; actual: number; predicted: number };
type Forecast = {
  symbol: string;
  current_close: number;
  predicted_close: number;
  mse: number;
  model_acc: number;
  naive_acc: number;
  series: SeriesPoint[];
};

export default function Home() {
  const [symbol, setSymbol] = useState<string>("");
  const [ohlc, setOhlc] = useState<Ohlc | null>(null);
  const [fcast, setFcast] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const positive = useMemo(
    () => (fcast ? fcast.predicted_close >= fcast.current_close : true),
    [fcast]
  );

  useEffect(() => {
    if (!symbol) return;
    let cancel = false;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const [o, f] = await Promise.all([
          fetch(`${API}/ohlc?symbol=${encodeURIComponent(symbol)}`).then((r) => r.json()),
          fetch(`${API}/forecast?symbol=${encodeURIComponent(symbol)}`).then((r) => r.json()),
        ]);
        if (!cancel) {
          if (o.detail) throw new Error(typeof o.detail === "string" ? o.detail : JSON.stringify(o.detail));
          if (f.detail) throw new Error(typeof f.detail === "string" ? f.detail : JSON.stringify(f.detail));
          setOhlc(o);
          setFcast(f);
        }
      } catch (e: any) {
        if (!cancel) setErr(e?.message ?? "Failed to load data");
      } finally {
        if (!cancel) setLoading(false);
      }
    })();
    return () => {
      cancel = true;
    };
  }, [symbol]);

  return (
    <main className="mx-auto max-w-3xl p-4 space-y-4">
      <div className="flex items-center justify-between gap-4">
        <SearchBox onPick={(s) => setSymbol(s)} />
        <ThemeToggle />
      </div>

      {symbol && (
        <div className="text-sm opacity-70">
          Selected: <span className="font-semibold">{symbol}</span>
        </div>
      )}

      {err && (
        <div className="rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
          {err}
        </div>
      )}

      {fcast && (
        <div
          className={`rounded-xl border p-4 ${
            positive ? "border-green-400" : "border-red-400"
          }`}
        >
          <div className="text-xl font-semibold">
            {fcast.symbol}: ${fcast.predicted_close.toFixed(2)}
            <span className="ml-2 text-sm opacity-80">MSE {fcast.mse.toFixed(2)}</span>
          </div>
          <div className="text-xs opacity-70">
            model acc: {(fcast.model_acc * 100).toFixed(0)}% • naive: {(fcast.naive_acc * 100).toFixed(0)}%
          </div>
        </div>
      )}

      {ohlc && (
        <div className="rounded-xl border p-4">
          <div className="mb-1 text-sm font-medium">Most-recent OHLC</div>
          <div className="grid grid-cols-5 gap-2 text-sm">
            <div>{ohlc.date}</div>
            <div>Open {ohlc.open.toFixed(2)}</div>
            <div>High {ohlc.high.toFixed(2)}</div>
            <div>Low {ohlc.low.toFixed(2)}</div>
            <div>Close {ohlc.close.toFixed(2)}</div>
          </div>
        </div>
      )}

      {loading && <div className="text-sm opacity-70">Loading…</div>}

      {fcast?.series && (
        <ForecastChart series={fcast.series} positive={positive} />
      )}

      <div className="text-xs opacity-60">
        ⚠️ Educational only — not financial advice.
      </div>
    </main>
  );
}