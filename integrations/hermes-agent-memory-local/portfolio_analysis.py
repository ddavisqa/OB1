#!/usr/bin/env python3
"""
Weekly Portfolio Analysis — sector concentration, rebalancing triggers,
FX exposure, and risk alerts. Runs on demand or via cron.

Usage:
  python3 portfolio_analysis.py          # full analysis
  python3 portfolio_analysis.py --brief  # condensed for Telegram
"""

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from collections import defaultdict

# Config
CHROMA = "http://localhost:8000"
COLL_BASE = f"{CHROMA}/api/v2/tenants/default_tenant/databases/default_database/collections"
HOLDINGS_PATH = os.path.expanduser("~/Dropbox/Obsidian/Business/Holdings.md")

# Current target allocation (from Rebalancing Blueprint)
SECTOR_TARGETS = {
    "Aerospace & Defense": 0.15,
    "Broad Market": 0.26,
    "Technology": 0.12,
    "Dividend Income": 0.07,
    "Semiconductors": 0.08,
    "Infrastructure": 0.03,
    "Crypto": 0.10,
    "Real Estate": 0.01,
    "Cash": 0.13,
}

SECTORS = {
    "Aerospace & Defense": {"ITA","PPA","XAR","EUAD","SHLD","SAFRF","BAESF","EADSF","RYCEY","FINMF","RNMBF","ACHR"},
    "Semiconductors": {"SOXX","SMH","XSD","CHPX","NVDA"},
    "Technology": {"QQQ","XLK","CLOU","CIBR","KTEC","GOOG","AMZN","ONDS"},
    "Infrastructure": {"GRID","IFRA","GLFOX","GRDU","NAT"},
    "Dividend Income": {"VIG","DIVO","VEIRX","DEM","IDV"},
    "Broad Market": {"SPY","SCHG","SCHK","ONEQ","FVAL","VSCIX","ESGV","DAX","SWVXX"},
    "Real Estate": {"VGSLX"},
    "Crypto": {"BTC.COIN","XRP.COIN"},
    "Cash": {"SWVXX"},
}

FX_EXPOSURE = {"EUAD", "SAFRF", "BAESF", "EADSF", "RYCEY", "DAX"}  # unhedged EUR/GBP


def log(msg):
    print(msg)


def load_positions():
    """Load ticker→shares from Holdings.md."""
    if not os.path.exists(HOLDINGS_PATH):
        return None
    text = open(HOLDINGS_PATH).read()
    positions = {}
    for m in re.finditer(r'\| +([A-Z][A-Z0-9.]+) +\| +([^|]+?) +\| +([\d,.]+) +\|', text):
        t = m.group(1).strip()
        s = m.group(3).replace(",", "")
        try:
            s = float(s)
            if s > 0: positions[t] = s
        except: pass
    return positions if positions else None


def get_sector(ticker):
    for s, ts in SECTORS.items():
        if ticker.upper() in ts:
            return s
    return "Other"


class Analysis:
    def __init__(self, positions, brief=False):
        self.positions = positions
        self.brief = brief
        self.prices = {}
        self.portfolio = 0.0
        self.sector_values = defaultdict(float)
        self.sector_pcts = {}
        self.errors = []
        self.fx_value = 0.0
        self.movers = []
        self.rebalance_triggers = []

    def fetch(self):
        import yfinance as yf
        tickers = list(self.positions.keys())
        # Map crypto
        yt_map = {}
        for t in tickers:
            if t == "BTC.COIN": yt_map["BTC-USD"] = "BTC.COIN"
            elif t == "XRP.COIN": yt_map["XRP-USD"] = "XRP.COIN"
            else: yt_map[t] = t

        if not self.brief: log(f"  Fetching {len(yt_map)} tickers...")

        for yt, rt in yt_map.items():
            try:
                tk = yf.Ticker(yt)
                h = tk.history(period="2d")
                if h.empty:
                    self.errors.append(f"{rt}: no data")
                    continue
                close = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else None
                chg = float((close - prev) / prev * 100) if prev and prev != 0 else 0
                self.prices[rt] = {"price": close, "prev_close": prev, "change_pct": chg}
            except Exception as e:
                self.errors.append(f"{rt}: {e}")

    def analyze(self):
        """Compute all metrics."""
        total = 0.0
        for ticker, shares in self.positions.items():
            pi = self.prices.get(ticker)
            if not pi: continue
            val = shares * pi["price"]
            total += val
            sec = get_sector(ticker)
            self.sector_values[sec] += val
            if ticker in FX_EXPOSURE:
                self.fx_value += val
            if abs(pi["change_pct"]) >= 2:
                self.movers.append((ticker, pi["change_pct"], val))

        self.portfolio = total

        # Sector percentages
        for s, v in sorted(self.sector_values.items(), key=lambda x: -x[1]):
            self.sector_pcts[s] = v / total if total > 0 else 0

        # Rebalancing triggers
        for sector, actual_pct in self.sector_pcts.items():
            target = SECTOR_TARGETS.get(sector, 0)
            drift = actual_pct - target
            if abs(drift) > 0.05:
                direction = "OVER" if drift > 0 else "UNDER"
                self.rebalance_triggers.append((sector, actual_pct, target, drift, direction))

    def generate(self):
        lines = []
        et = datetime.fromtimestamp(time.time() - (5 if time.localtime().tm_isdst == 0 else 4) * 3600)
        lines.append(f"📊 Weekly Portfolio Analysis — {et.strftime('%a %b %d, %Y')}")
        lines.append("")

        # Portfolio value
        lines.append(f"**Portfolio Value:** ${self.portfolio:,.0f} ({len([t for t in self.prices if self.prices[t]])} positions)")
        lines.append("")

        # Sector allocation vs targets
        lines.append("**Sector Allocation vs Target:**")
        for s in sorted(self.sector_pcts.keys(), key=lambda x: -self.sector_pcts[x]):
            ap = self.sector_pcts[s]
            tgt = SECTOR_TARGETS.get(s, 0)
            drift = ap - tgt
            marker = ""
            if abs(drift) > 0.05:
                marker = " ⚠️" if abs(drift) > 0.08 else ""
                marker += " OVER" if drift > 0 else " UNDER"
            lines.append(f"  {s:25s} {ap*100:5.1f}% (target {tgt*100:5.1f}%) {'🟢' if abs(drift)<0.05 else '🔴'}{marker}")
        lines.append("")

        # Rebalancing triggers
        if self.rebalance_triggers:
            lines.append(f"**Rebalance Triggers (>5% drift):**")
            for s, ap, tgt, drift, d in sorted(self.rebalance_triggers, key=lambda x: -abs(x[3])):
                action = f"Sell ${self.portfolio*abs(drift):,.0f}" if d == "OVER" else f"Buy ${self.portfolio*abs(drift):,.0f}"
                lines.append(f"  {s:25s} {d:5s} by {abs(drift)*100:.1f}% — {action}")
            lines.append("")

        # FX exposure
        if self.fx_value > 0:
            fx_pct = self.fx_value / self.portfolio * 100 if self.portfolio > 0 else 0
            lines.append(f"**FX Exposure (unhedged EUR/GBP):** ${self.fx_value:,.0f} ({fx_pct:.1f}%)")
            lines.append("")

        # Movers
        if self.movers:
            lines.append("**Notable Movers (>2%):**")
            for t, chg, val in sorted(self.movers, key=lambda x: -abs(x[1])):
                d = "🟢" if chg > 0 else "🔴"
                lines.append(f"  {d} {t}: {chg:+.1f}% (${val:,.0f})")
            lines.append("")

        # Top 5
        if not self.brief:
            lines.append("**Top 5 Holdings:**")
            sorted_pos = sorted(self.positions.items(), key=lambda x: self.prices.get(x[0], {}).get("price", 0) * x[1] if x[0] in self.prices else 0, reverse=True)
            for t, s in sorted_pos[:5]:
                pi = self.prices.get(t)
                if pi:
                    v = s * pi["price"]
                    c = pi["change_pct"]
                    m = "🟢" if c >= 0 else "🔴"
                    lines.append(f"  {m} {t:8s} ${v:>8,.0f} ({c:+.2f}%)")
            lines.append("")

        # Errors
        if self.errors and not self.brief:
            lines.append(f"⚠️ {len(self.errors)} errors")

        lines.append(f"\n---\nGenerated: {et.strftime('%H:%M ET')}")
        return "\n".join(lines)


def main():
    brief = "--brief" in sys.argv or "--weekly" in sys.argv

    if not brief:
        print("=" * 60)
        print("📊 WEEKLY PORTFOLIO ANALYSIS")
        print("=" * 60)
        print()

    positions = load_positions()
    if not positions:
        print("  ❌ Load Holdings.md first via ingest_financials.py")
        sys.exit(1)

    if not brief:
        print(f"  ✓ {len(positions)} positions loaded")
        print()

    a = Analysis(positions, brief=brief)
    a.fetch()
    a.analyze()
    print(a.generate())


if __name__ == "__main__":
    main()
