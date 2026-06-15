'use client';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;

  const next = resolvedTheme === 'dark' ? 'light' : 'dark';
  return (
    <button
      className="rounded-xl border px-3 py-1 text-sm"
      onClick={() => setTheme(next)}
      aria-label="Toggle theme"
    >
      Toggle {next} mode
    </button>
  );
}