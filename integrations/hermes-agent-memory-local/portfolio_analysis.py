#!/usr/bin/env python3
"""
Weekly Portfolio Analysis — sector concentration, rebalancing triggers,
FX exposure, and risk alerts. Runs on demand or via cron.

Dual-source: FMP REST API (preferred) → yfinance fallback.
FMP free tier covers ~5 US tickers + crypto; most tickers fall back to yfinance.
Premium tier ($49/mo) would cover everything via FMP.

Usage:
  python3 portfolio_analysis.py          # full analysis
  python3 portfolio_analysis.py --brief  # condensed for Telegram
"""

import json
import os
import re
import sys
import ssl
import time
import urllib.request
from collections import defaultdict
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
# Secret lives in ~/.hermes/.env (sourced by the cron wrapper). Never hardcode:
# this repo's history is public-push-adjacent and a committed key leaks.
FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()
FMP_BASE = "https://financialmodelingprep.com/stable"

HOLDINGS_PATH = os.path.expanduser("~/Dropbox/Obsidian/Business/Holdings.md")

SECTOR_TARGETS = {
    "Aerospace & Defense": 0.15, "Broad Market": 0.26, "Technology": 0.12,
    "Dividend Income": 0.07, "Semiconductors": 0.08, "Infrastructure": 0.03,
    "Crypto": 0.10, "Real Estate": 0.01,
    # "Cash" = SECURITIES cash only (SWVXX money market). External cash, DTEM
    # operating float and property reserves live in the vault "Cash & Reserves"
    # note, NOT here. The old 0.13 assumed $20k of 401(k) cash that had already
    # been invested, so it fired a permanent bogus "Buy $24,741" trigger.
    "Cash": 0.07,
    "International Equity": 0.18,
}
# WARNING: these targets currently SUM TO 113% (95% pre-existing + 18% here).
# The blueprint predates the six newly-classified sleeves, so it is not a
# coherent 100% allocation policy yet. Rebalance the set before acting on drift.
# Any sector in SECTORS but absent from SECTOR_TARGETS is UNTARGETED: it is
# still reported as a share of the book, but generates NO rebalance trigger.

SECTORS = {
    "Aerospace & Defense": {"ITA","PPA","XAR","EUAD","SHLD","SAFRF","BAESF","EADSF","RYCEY","FINMF","RNMBF","ACHR","TKMSF"},
    "Semiconductors": {"SOXX","SMH","XSD","CHPX","NVDA"},
    "Technology": {"QQQ","XLK","CLOU","CIBR","KTEC","GOOG","AMZN","ONDS","BBAI"},
    "Infrastructure": {"GRID","IFRA","GLFOX","GRDU","NAT"},
    "Dividend Income": {"VIG","DIVO","VEIRX","DEM","IDV","DON","SDY","DES"},
    "Broad Market": {"SPY","SCHG","SCHK","ONEQ","FVAL","VSCIX","ESGV","DAX","VTI","VMCIX","BRK.B"},
    "International Equity": {"VSGX","VEA","VWO","VEMAX","VTSNX","ZIM","TYEKF"},
    "Fixed Income": {"BND","BNDX","VCEB","VGIT","VGSH"},
    "Precious Metals": {"SIVR"},
    "Currency": {"FXE"},
    "Healthcare": {"XLV"},
    "Consumer Discretionary": {"F"},
    "Real Estate": {"VGSLX","VNQ"},
    "Crypto": {"BTC.COIN","XRP.COIN"},
    "Cash": {"SWVXX"},
}

FX_EXPOSURE = {"EUAD", "SAFRF", "BAESF", "EADSF", "RYCEY", "DAX"}

# yfinance uses dashes for dashed share classes (BRK.B -> BRK-B).
_YF_ALIAS = {"BRK.B": "BRK-B"}

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def load_positions():
    if not os.path.exists(HOLDINGS_PATH):
        return None
    text = open(HOLDINGS_PATH).read()
    positions = {}
    # Ticker may be a single letter (F, T, V) — hence the '*' not '+'.
    for m in re.finditer(r'\| +([A-Z][A-Z0-9.]*) +\| +([^|]+?) +\| +([\d,.]+) +\|', text):
        t = m.group(1).strip()
        try:
            s = float(m.group(3).replace(",", ""))
            if s > 0:
                # SUM, don't overwrite: the same ticker can be held in more than
                # one account (e.g. VTI in both the brokerage and the Carry.com
                # 401(k) roboadvisor). Overwriting silently dropped one account.
                positions[t] = positions.get(t, 0.0) + s
        except Exception:
            pass
    return positions if positions else None


def get_sector(ticker):
    for s, ts in SECTORS.items():
        if ticker.upper() in ts:
            return s
    return "Other"


def _fmp_sym(ticker):
    return "BTCUSD" if ticker == "BTC.COIN" else "XRPUSD" if ticker == "XRP.COIN" else ticker


def _yf_sym(ticker):
    if ticker == "BTC.COIN":
        return "BTC-USD"
    if ticker == "XRP.COIN":
        return "XRP-USD"
    return _YF_ALIAS.get(ticker, ticker)


def _fmp_quote(symbol):
    """Fetch quote via FMP REST API. Returns (price, change_pct) or None."""
    if not FMP_API_KEY:
        return None
    url = f"{FMP_BASE}/quote?symbol={symbol}&apikey={FMP_API_KEY}"
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Accept": "application/json"}),
            timeout=10, context=_CTX)
        data = json.loads(resp.read())
        if not data or not isinstance(data, list):
            return None
        d = data[0]
        if "price" not in d:
            return None
        return d["price"], d.get("changePercentage", 0)
    except Exception:
        return None


def _yf_quote(ticker):
    """Fetch quote via yfinance. Returns (price, change_pct) or None."""
    try:
        import yfinance as yf
        tk = yf.Ticker(_yf_sym(ticker))
        h = tk.history(period="2d")
        if h.empty:
            return None
        close = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2]) if len(h) > 1 else None
        chg = float((close - prev) / prev * 100) if prev and prev != 0 else 0
        return close, chg
    except Exception:
        return None


class Analysis:
    def __init__(self, positions, brief=False):
        self.positions = positions
        self.brief = brief
        self.prices = {}
        self.sector_values = defaultdict(float)
        self.sector_pcts = {}
        self.errors = []
        self.fx_value = 0.0
        self.movers = []
        self.rebalance_triggers = []
        self.source = {}

    def fetch(self):
        tickers = list(self.positions.keys())
        if not self.brief:
            print(f"  Fetching {len(tickers)} tickers (FMP → yfinance)…")

        for t in tickers:
            # FMP first
            r = _fmp_quote(_fmp_sym(t))
            if r:
                self.prices[t] = {"price": r[0], "change_pct": r[1]}
                self.source[t] = "fmp"
                continue
            # yfinance fallback
            r = _yf_quote(t)
            if r:
                self.prices[t] = {"price": r[0], "change_pct": r[1]}
                self.source[t] = "yf"
                continue
            self.errors.append(f"{t}: no data")

        if not self.brief:
            f = sum(1 for s in self.source.values() if s == "fmp")
            y = sum(1 for s in self.source.values() if s == "yf")
            print(f"    FMP: {f} | yfinance: {y} | errors: {len(self.errors)}")

    def analyze(self):
        for ticker, shares in self.positions.items():
            pi = self.prices.get(ticker)
            if not pi:
                continue
            val = shares * pi["price"]
            self.sector_values[get_sector(ticker)] += val
            if ticker in FX_EXPOSURE:
                self.fx_value += val
            if abs(pi["change_pct"]) >= 2:
                self.movers.append((ticker, pi["change_pct"], val))

        self.portfolio = sum(
            shares * self.prices[t]["price"]
            for t, shares in self.positions.items()
            if t in self.prices
        )

        for s, v in sorted(self.sector_values.items(), key=lambda x: -x[1]):
            self.sector_pcts[s] = v / self.portfolio if self.portfolio else 0

        for s, ap in self.sector_pcts.items():
            tgt = SECTOR_TARGETS.get(s)
            if tgt is None:          # untargeted sleeve — no drift signal
                continue
            drift = ap - tgt
            if abs(drift) > 0.05:
                self.rebalance_triggers.append((s, ap, tgt, drift, "OVER" if drift > 0 else "UNDER"))

    def generate(self):
        et = datetime.fromtimestamp(time.time() - 5 * 3600)
        lines = [f"📊 Weekly Portfolio Analysis — {et.strftime('%a %b %d, %Y')}", ""]

        n = len([t for t in self.prices if self.prices[t]])
        lines.append(f"**Portfolio Value:** ${self.portfolio:,.0f} ({n} positions)")
        lines.append("")

        lines.append("**Sector Allocation vs Target:**")
        for s in sorted(self.sector_pcts, key=lambda x: -self.sector_pcts[x]):
            ap = self.sector_pcts[s]
            tgt = SECTOR_TARGETS.get(s)
            if tgt is None:
                lines.append(f"  {s:25s} {ap*100:5.1f}% (no target)      ⚪")
                continue
            drift = ap - tgt
            m = ""
            if abs(drift) > 0.05:
                m = (" ⚠️" if abs(drift) > 0.08 else "") + (" OVER" if drift > 0 else " UNDER")
            lines.append(f"  {s:25s} {ap*100:5.1f}% (target {tgt*100:5.1f}%) "
                         f"{'🟢' if abs(drift)<0.05 else '🔴'}{m}")
        lines.append("")

        if self.rebalance_triggers:
            lines.append("**Rebalance Triggers (>5% drift):**")
            for s, ap, tgt, drift, d in sorted(self.rebalance_triggers, key=lambda x: -abs(x[3])):
                amt = f"Sell ${self.portfolio*abs(drift):,.0f}" if d == "OVER" else f"Buy ${self.portfolio*abs(drift):,.0f}"
                lines.append(f"  {s:25s} {d:5s} by {abs(drift)*100:.1f}% — {amt}")
            lines.append("")

        if self.fx_value:
            pct = self.fx_value / self.portfolio * 100 if self.portfolio else 0
            lines.append(f"**FX Exposure (unhedged EUR/GBP):** ${self.fx_value:,.0f} ({pct:.1f}%)")
            lines.append("")

        if self.movers:
            lines.append("**Notable Movers (>2%):**")
            for t, chg, val in sorted(self.movers, key=lambda x: -abs(x[1])):
                lines.append(f"  {'🟢' if chg > 0 else '🔴'} {t}: {chg:+.1f}% (${val:,.0f})")
            lines.append("")

        if not self.brief:
            lines.append("**Top 5 Holdings:**")
            sp = sorted(self.positions.items(),
                        key=lambda x: self.prices.get(x[0], {}).get("price", 0) * x[1] if x[0] in self.prices else 0,
                        reverse=True)[:5]
            for t, s in sp:
                pi = self.prices.get(t)
                if pi:
                    lines.append(f"  {'🟢' if pi['change_pct'] >= 0 else '🔴'} {t:8s} ${s*pi['price']:>8,.0f} ({pi['change_pct']:+.2f}%)")
            lines.append("")

        if self.errors and not self.brief:
            lines.append(f"⚠️ {len(self.errors)} errors")
        if not self.brief:
            f = sum(1 for s in self.source.values() if s == "fmp")
            y = sum(1 for s in self.source.values() if s == "yf")
            lines.append(f"*FMP: {f} · yfinance: {y}*")

        lines.append(f"\n---\nGenerated: {et.strftime('%H:%M ET')}")
        return "\n".join(lines)


def main():
    brief = "--brief" in sys.argv or "--weekly" in sys.argv
    if not brief:
        print("=" * 60)
        print("📊 WEEKLY PORTFOLIO ANALYSIS")
        print("=" * 60, "\n")

    if not FMP_API_KEY:
        print("  ⚠️  FMP_API_KEY not set — every ticker will fall back to yfinance")
        print("      Set it in ~/.hermes/.env (the cron wrapper sources that file).\n")

    positions = load_positions()
    if not positions:
        print("  ❌ Load Holdings.md first via ingest_financials.py")
        sys.exit(1)

    if not brief:
        print(f"  ✓ {len(positions)} positions loaded\n")

    a = Analysis(positions, brief=brief)
    a.fetch()
    a.analyze()
    print(a.generate())


if __name__ == "__main__":
    main()
