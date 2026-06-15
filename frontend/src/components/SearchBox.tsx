"use client";

import { useEffect, useMemo, useState } from "react";
const API = process.env.NEXT_PUBLIC_BACKEND_URL!;

type Hit = { symbol: string; name?: string; exchange?: string };

export default function SearchBox({ onPick }: { onPick: (s: string) => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(async () => {
      if (!q || q.trim().length < 2) {
        setHits([]);
        setOpen(false);
        setErr(null);
        return;
      }
      try {
        setLoading(true);
        setErr(null);
        const res = await fetch(`${API}/search?q=${encodeURIComponent(q.trim())}`);
        if (!res.ok) {
          let msg = `Search failed (${res.status})`;
          try {
            const j = await res.json();
            msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
          } catch {}
          throw new Error(msg);
        }
        const data = (await res.json()) as Hit[] | unknown;
        setHits(Array.isArray(data) ? data : []);
        setOpen(true);
      } catch (e: unknown) {
        setHits([]);
        setOpen(false);
        setErr(e instanceof Error ? e.message : "Network error");
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [q]);

  // ✅ Deduplicate by symbol+exchange to avoid duplicates
  const deduped = useMemo(() => {
    const seen = new Set<string>();
    const out: Hit[] = [];
    for (const h of hits) {
      const key = `${h.symbol}|${h.exchange ?? ""}`.toUpperCase();
      if (!seen.has(key)) {
        seen.add(key);
        out.push(h);
      }
    }
    return out.slice(0, 10);
  }, [hits]);

  return (
    <div className="relative w-full">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Type a company or symbol (e.g., AAPL)…"
        className="w-full rounded-xl border px-4 py-2"
        onFocus={() => deduped.length && setOpen(true)}
      />
      {open && (
        <div className="absolute z-10 mt-1 w-full overflow-hidden rounded-xl border bg-white shadow dark:bg-neutral-900">
          {loading && <div className="px-3 py-2 text-sm opacity-70">Searching…</div>}
          {!loading &&
            deduped.map((h, i) => (
              <button
                // ✅ Unique, stable key (symbol+exchange) + index fallback
                key={`${h.symbol}-${h.exchange ?? "NA"}-${i}`}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-800"
                onClick={() => {
                  onPick(h.symbol);
                  setOpen(false);
                  setQ(h.symbol);
                }}
              >
                <span className="font-medium">{h.symbol}</span>
                <span className="truncate text-xs opacity-70">
                  {h.name} {h.exchange ? `• ${h.exchange}` : ""}
                </span>
              </button>
            ))}
          {!loading && deduped.length === 0 && (
            <div className="px-3 py-2 text-sm opacity-60">No results</div>
          )}
        </div>
      )}
      {err && <div className="mt-2 text-xs text-red-600 dark:text-red-300">{err}</div>}
    </div>
  );
}