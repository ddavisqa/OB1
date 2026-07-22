#!/usr/bin/env python3
"""
Pre-Market Briefing — daily portfolio review before market open.
Reads positions from OB1 (ChromaDB) or Holdings.md, pulls live prices via yfinance.
Delivers a structured morning briefing.

Usage:
  python3 pre_market_briefing.py          # stdout briefing
  python3 pre_market_briefing.py --brief  # shorter version for Telegram
  python3 pre_market_briefing.py --watchlist-only  # just tickers with prices
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Config
CHROMA = "http://localhost:8000"
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v1/search"

# Ticker → company name for news queries
TICKER_NAMES = {
    "NVDA": "NVIDIA", "GOOG": "Alphabet Google", "AMZN": "Amazon",
    "ESGV": "Vanguard ESG", "SWVXX": "Schwab money market",
    "BTC.COIN": "Bitcoin", "XRP.COIN": "XRP Ripple",
    "ITA": "iShares aerospace defense ETF", "SOXX": "iShares semiconductor",
    "SMH": "VanEck semiconductor", "VIG": "Vanguard dividend",
    "BAESF": "BAE Systems", "RNMBF": "Rheinmetall", "FINMF": "Leonardo",
    "SAFRF": "Safran", "EADSF": "Airbus", "RYCEY": "Rolls-Royce",
    "ACHR": "Archer Aviation", "NAT": "Nordic American Tankers",
    "ONDS": "Ondas", "ZIM": "Zimplats", "BBAI": "BigBear.ai",
    "KTEC": "KraneShares Hang Seng Tech",
    "TKMSF": "ThyssenKrupp Marine", "TYEKF": "thyssenkrupp",
    "XLK": "Technology Select Sector SPDR",
}
COLL_BASE = f"{CHROMA}/api/v2/tenants/default_tenant/databases/default_database/collections"
COLL = "ob1_memory"
HOLDINGS_MD = os.path.expanduser("~/Dropbox/Obsidian/Business/Holdings.md")
CACHE_TTL = 3600  # 1 hour price cache

# Sector tagging (from ingest_financials.py)
SECTORS = {
    "Aerospace & Defense": {"ITA","PPA","XAR","EUAD","SHLD","SAFRF","BAESF","EADSF","RYCEY","FINMF","RNMBF","ACHR"},
    "Semiconductors": {"SOXX","SMH","XSD","CHPX","NVDA"},
    "Technology": {"QQQ","XLK","CLOU","CIBR","KTEC","GOOG","AMZN","ONDS"},
    "Infrastructure": {"GRID","IFRA","GLFOX","GRDU","NAT"},
    "Dividend Income": {"VIG","DIVO","VEIRX","DEM","IDV"},
    "Broad Market": {"SPY","SCHG","SCHK","ONEQ","FVAL","VSCIX","ESGV","DAX","SWVXX"},
    "Real Estate": {"VGSLX"},
    "Crypto": {"BTC.COIN","XRP.COIN"},
}


def log(msg):
    print(msg)


def get_cid():
    try:
        r = json.loads(urllib.request.urlopen(f"{COLL_BASE}?name={COLL}", timeout=5).read())
        matched = [c for c in (r if isinstance(r, list) else []) if c.get("name") == COLL]
        if matched: return matched[0]["id"]
    except: pass
    return None


def load_positions():
    """Load positions from Holdings.md (primary) or OB1 (fallback).
    
    Holdings.md has the canonical share counts in table format:
    | TICKER | Name | Shares | Price | Value |
    """
    positions = {}
    
    # Primary: Holdings.md (has share counts)
    if os.path.exists(HOLDINGS_MD):
        text = open(HOLDINGS_MD).read()
        # Parse table rows: | TICKER | Name | Shares | Price | Value |
        for match in re.finditer(
            r'\| +([A-Z][A-Z0-9.]+) +\| +([^|]+?) +\| +([\d,.]+) +\|',
            text
        ):
            ticker = match.group(1).strip()
            shares_str = match.group(3).replace(",", "")
            try:
                shares = float(shares_str)
            except:
                continue
            if shares > 0:
                positions[ticker] = {"shares": shares, "source": "holdings_md"}
    
    if positions:
        return positions
    
    # Fallback: OB1 memory
    cid = get_cid()
    if not cid:
        return None
    try:
        body = json.dumps({"where": {"$and": [{"workspace_id": "ddavis"}, {"content_type": "position"}]},
                          "limit": 100}).encode()
        req = urllib.request.Request(f"{COLL_BASE}/{cid}/get", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except:
        return None
    
    for i, doc_id in enumerate((data.get("ids") or [])):
        meta = (data.get("metadatas") or [{}])[i] if i < len(data.get("metadatas", [])) else {}
        ticker = meta.get("ticker", "")
        if ticker:
            positions[ticker] = {"source": "ob1"}
    return positions if positions else None


class Briefing:
    def __init__(self, positions, brief_mode=False):
        self.positions = positions
        self.brief_mode = brief_mode
        self.prices = {}
        self.errors = []
        self.timestamp = datetime.now(timezone.utc)

    def fetch_prices(self):
        """Fetch live prices via yfinance for all tickers individually."""
        import yfinance as yf

        tickers = list(self.positions.keys())
        yahoo_tickers = []
        crypto_map = {}
        for t in tickers:
            if t == "BTC.COIN":
                yahoo_tickers.append("BTC-USD")
                crypto_map["BTC-USD"] = "BTC.COIN"
            elif t == "XRP.COIN":
                yahoo_tickers.append("XRP-USD")
                crypto_map["XRP-USD"] = "XRP.COIN"
            else:
                yahoo_tickers.append(t)

        if self.brief_mode:
            log(f"  Fetching {len(yahoo_tickers)} tickers...")

        for yt in yahoo_tickers:
            real_ticker = crypto_map.get(yt, yt)
            try:
                tk = yf.Ticker(yt)
                hist = tk.history(period="2d")
                if hist.empty:
                    self.errors.append(f"{yt}: no data")
                    continue
                close_series = hist["Close"].dropna()
                if close_series.empty:
                    self.errors.append(f"{yt}: no valid close prices")
                    continue
                close = float(close_series.iloc[-1])
                prev = float(close_series.iloc[-2]) if len(close_series) > 1 else None
                chg = float((close - prev) / prev * 100) if prev and prev != 0 else 0
                self.prices[real_ticker] = {
                    "price": close,
                    "prev_close": prev,
                    "change_pct": chg,
                }
            except Exception as e:
                self.errors.append(f"{yt}: {e}")

    def get_sector(self, ticker):
        for sector, tickers in SECTORS.items():
            if ticker.upper() in tickers:
                return sector
        return "Other"

    def latest_close_time(self):
        """Human-readable time of last price."""
        now = datetime.now(timezone.utc)
        if now.hour >= 13 and now.weekday() < 5:  # After 9 AM ET on weekdays
            return "market hours"
        return "previous close"

    def fetch_news(self, tickers, max_results=5):
        """Fetch news headlines via yfinance (Yahoo Finance API). Returns list of (ticker, title, url)."""
        import yfinance as yf
        headlines = []
        seen = set()
        for ticker in tickers[:8]:  # max 8 searches to keep latency down
            if ticker.endswith(".COIN"):
                continue  # skip crypto tickers
            try:
                tk = yf.Ticker(ticker)
                news = tk.news or []
                per_ticker = 0
                for item in news[:8]:
                    content = item.get("content") or item
                    title = (content.get("title") or "").strip()
                    url = None
                    for key in ("clickThroughUrl", "canonicalUrl"):
                        u = content.get(key, {})
                        if isinstance(u, dict):
                            url = u.get("url") or url
                    if not title or not url:
                        continue
                    # Skip low-quality finance aggregators
                    if any(d in url for d in ["stocktwits.com", "marketbeat.com", "benzinga.com", "tradingview.com"]):
                        continue
                    if title not in seen:
                        seen.add(title)
                        headlines.append((ticker, title, url))
                        per_ticker += 1
                        if per_ticker >= 2:  # max 2 per ticker
                            break
            except Exception as e:
                self.errors.append(f"News fetch {ticker}: {e}")
        return headlines[:max_results]

    def generate(self):
        now_str = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        brief = []

        # Header
        est_offset = -5 if time.localtime().tm_isdst == 0 else -4
        et = datetime.fromtimestamp(time.time() + est_offset * 3600)
        brief.append(f"📊 Pre-Market Briefing — {et.strftime('%a %b %d, %Y')}")
        brief.append(f"   Prices: {self.latest_close_time()}")
        brief.append("")

        # Portfolio value
        total_value = 0
        total_prev = 0
        priced_count = 0
        movers = []
        mover_tickers = set()
        sector_values = {}

        for ticker, pos in self.positions.items():
            price_info = self.prices.get(ticker)
            if not price_info:
                continue
            shares = pos.get("shares", 0)
            if shares <= 0:
                continue

            current_val = shares * price_info["price"]
            prev_val = shares * price_info["prev_close"] if price_info["prev_close"] else current_val
            total_value += current_val
            total_prev += prev_val
            priced_count += 1

            change_pct = price_info["change_pct"]
            sector = self.get_sector(ticker)
            sector_values[sector] = sector_values.get(sector, 0) + current_val

            if abs(change_pct) >= 2:
                direction = "🟢" if change_pct > 0 else "🔴"
                shares_str = f"{shares:,.0f} sh" if shares >= 1 else f"{shares:,.4f} sh"
                movers.append(f"  {direction} {ticker}: {change_pct:+.1f}% (${current_val:,.0f}, {shares_str})")
                mover_tickers.add(ticker)

        # Portfolio summary
        pnl = total_value - total_prev
        pnl_pct = (pnl / total_prev * 100) if total_prev > 0 else 0
        pnl_symbol = "🟢" if pnl >= 0 else "🔴"

        brief.append(f"**Portfolio Value:** ${total_value:,.0f} ({priced_count} positions)")
        brief.append(f"  {pnl_symbol} Change: ${pnl:+,.0f} ({pnl_pct:+.2f}%)")
        brief.append("")

        # Sector breakdown
        if not self.brief_mode:
            brief.append("**Sector Allocation:**")
            for sector, val in sorted(sector_values.items(), key=lambda x: -x[1]):
                pct = val / total_value * 100
                brief.append(f"  {sector:25s} ${val:>8,.0f} ({pct:5.1f}%)")
            brief.append("")

        # Movers
        if movers:
            brief.append(f"**Notable Movers (>2%):**")
            brief.extend(movers)
            brief.append("")

        # Top holdings
        top_tickers = []
        if not self.brief_mode:
            brief.append("**Top 5 Holdings:**")
            sorted_pos = sorted(self.positions.items(), key=lambda x: self.prices.get(x[0], {}).get("price", 0) * x[1].get("shares", 0) if x[0] in self.prices else 0, reverse=True)
            for ticker, pos in sorted_pos[:5]:
                pi = self.prices.get(ticker)
                if pi:
                    val = pos.get("shares", 0) * pi["price"]
                    cp = pi["change_pct"]
                    m = "🟢" if cp >= 0 else "🔴"
                    brief.append(f"  {m} {ticker:8s} ${val:>8,.0f} ({cp:+.2f}%)")
                    top_tickers.append(ticker)
            brief.append("")

        # News (movers + top 5)
        news_tickers = list(mover_tickers | set(top_tickers))
        if news_tickers and FIRECRAWL_KEY:
            headlines = self.fetch_news(news_tickers, max_results=5)
            if headlines:
                brief.append("**📰 Recent Headlines:**")
                for ticker, title, url in headlines:
                    brief.append(f"  [{ticker}] [{title}]({url})")
                brief.append("")

        # Errors
        if self.errors:
            price_errs = [e for e in self.errors if not e.startswith("News fetch")]
            news_errs = [e for e in self.errors if e.startswith("News fetch")]
            if price_errs:
                brief.append(f"⚠️ {len(price_errs)} price fetch errors")
                if self.brief_mode:
                    brief.append(f"   Tickers: {', '.join(e.split(':')[0] for e in price_errs[:5])}")
                else:
                    for e in price_errs[:10]:
                        brief.append(f"  {e}")
            if news_errs and not self.brief_mode:
                brief.append(f"📡 {len(news_errs)} news fetch errors")
                for e in news_errs[:5]:
                    brief.append(f"  {e}")
            brief.append("")

        # Timestamp
        brief.append(f"---")
        brief.append(f"Generated: {now_str}")

        return "\n".join(brief)


def main():
    brief_mode = "--brief" in sys.argv
    watchlist_only = "--watchlist-only" in sys.argv

    if brief_mode:
        log("📊 Pre-Market Briefing (brief)")
    else:
        log("=" * 60)
        log("📊 PRE-MARKET BRIEFING")
        log("=" * 60)
        log("")

    # Load positions
    positions = load_positions()

    if not positions:
        log("  ❌ No positions found. Run ingest_financials.py or check Holdings.md.")
        sys.exit(1)

    if not brief_mode:
        log(f"  ✓ {len(positions)} positions loaded")

    # Build briefing
    b = Briefing(positions, brief_mode=brief_mode)
    b.fetch_prices()

    if watchlist_only:
        for ticker in sorted(positions.keys()):
            pi = b.prices.get(ticker)
            if pi:
                cp = pi["change_pct"]
                m = "🟢" if cp >= 0 else "🔴"
                print(f"{m} {ticker:8s} ${pi['price']:>8.2f} ({cp:+.2f}%)")
        return

    result = b.generate()
    print(result)


if __name__ == "__main__":
    main()
