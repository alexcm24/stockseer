"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ForecastChart({
  series,
  positive,
}: {
  series: { date: string; actual: number; predicted: number }[];
  positive: boolean;
}) {
  const stroke = positive ? "#16a34a" : "#ef4444"; // green / red
  return (
    <div className={`rounded-xl border p-4 ${positive ? "border-green-400" : "border-red-400"}`}>
      <div className="mb-2 text-sm opacity-80">Actual vs. Predicted Close</div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" hide />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="actual" name="Actual" fillOpacity={0.18} stroke={stroke} fill={stroke} />
            <Area type="monotone" dataKey="predicted" name="Predicted" fillOpacity={0.12} stroke="#64748b" fill="#64748b" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-xs opacity-70">
        Predictions are for educational purposes only and may be wrong.
      </div>
    </div>
  );
}