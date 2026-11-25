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
from zoneinfo import ZoneInfo

# Config
CHROMA = "http://localhost:8000"
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v1/search"

# ── Duplicate suppression ────────────────────────────────────────────────────────────────────
# The briefing fires at 08:00 / 08:30 / 09:00 / 09:15. With no pre-open data movement those four
# runs used to emit near-identical messages. Now each run compares its *material* numbers against
# the last DELIVERED briefing of the same day and stays silent unless something actually changed.
# The decision lives here rather than in the shell wrapper because the comparison needs real
# numbers; a text hash in the wrapper was defeated by the freshness stamp, which advances every
# minute and would have made every run look different.
STATE_PATH = os.path.expanduser("~/.hermes/scripts/.premarket_brief_state.json")
MATERIALITY_PCT = float(os.environ.get("BRIEF_MATERIALITY_PCT", "0.10"))


def _load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(d):
    try:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        sys.stderr.write(f"dedup: could not persist state ({e})\n")


def _pub_epoch(content):
    """Yahoo news `pubDate` (ISO-8601 Z) -> epoch seconds; 0.0 when absent/unparseable.

    Used instead of comparing title sets: the feed reshuffles its window on every call, so
    "a title I have not seen" is noisy, while "an article published after the last delivery"
    is exactly the event worth interrupting someone for.
    """
    raw = content.get("pubDate") or content.get("displayTime")
    if not raw:
        return 0.0
    try:
        return datetime.strptime(str(raw)[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0


def should_deliver(b, today_iso):
    """Return (deliver: bool, reason: str) for this run's briefing."""
    if os.environ.get("BRIEF_DEDUP", "1") != "1":
        return True, "dedup disabled (BRIEF_DEDUP!=1)"

    s = getattr(b, "summary", None)
    if not s:
        return True, "no summary available"

    st = _load_state()
    if st.get("date") != today_iso:
        return True, "first briefing of the day"

    if int(st.get("priced", -1)) != int(s["priced"]):
        return True, f"priced-position count changed ({st.get('priced')} -> {s['priced']})"
    if int(st.get("errors", -1)) != int(s["errors"]):
        return True, f"fetch-error count changed ({st.get('errors')} -> {s['errors']})"

    try:
        prev = float(st.get("value") or 0.0)
    except (TypeError, ValueError):
        prev = 0.0
    if prev > 0:
        delta = abs(s["value"] - prev) / prev * 100.0
    else:
        delta = 100.0
    if delta >= MATERIALITY_PCT:
        return True, f"portfolio moved {delta:.2f}% (threshold {MATERIALITY_PCT}%)"

    if list(st.get("movers") or []) != list(s["movers"]):
        return True, "notable-mover set changed"
    # NOTE — the news feed is deliberately NOT a delivery trigger. Yahoo publishes continuously
    # (a 4-run test fired on "article published since the last briefing" in 2 of the runs, 12
    # seconds apart), and the sampled window varies with the mover set, so any freshness-based
    # news trigger degenerates into "always deliver" — the exact spam being removed here.
    # Headlines still ride along whenever a delivery happens for a real reason. `newest_pub` is
    # recorded in the state file for provenance and future tuning.
    # NOTE — the MarketWatch snapshot (Dow/S&P/Nasdaq/VIX...) is likewise NOT a trigger: it
    # ticks continuously, so comparing it made every run look "changed".

    return False, f"unchanged (portfolio {delta:.3f}% off the last briefing)"

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
    """Progress/diagnostic output.

    Goes to STDERR on purpose: for a `no_agent` cron job, stdout is the delivered message, so
    anything logged here would ship to Telegram. Worse, when duplicate suppression decides to
    stay silent, a stdout-bound log line produces a non-empty message anyway — which is how a
    58-byte "📊 Pre-Market Briefing (brief) / Fetching 62 tickers..." stub leaked through a
    suppressed run on 2026-09-23. stderr is captured in the cron log and never delivered.
    """
    sys.stderr.write(msg + "\n")


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
            r'\| +([A-Z][A-Z0-9.]*) +\| +([^|]+?) +\| +([\d,.]+) +\|',
            text
        ):
            ticker = match.group(1).strip()
            shares_str = match.group(3).replace(",", "")
            try:
                shares = float(shares_str)
            except:
                continue
            if shares > 0:
                # SUM across accounts — the same ticker may appear in the main
                # table AND the Carry.com 401(k) section; overwriting dropped one.
                if ticker in positions:
                    positions[ticker]["shares"] += shares
                else:
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
        self.sessions = {}      # ticker -> "pre-market" | "market hours" | "after hours" | "previous close"
        self.as_of = {}         # ticker -> tz-aware datetime of the bar the price came from
        self.timestamp = datetime.now(timezone.utc)

    @staticmethod
    def _session_of(ts):
        """Classify an intraday bar timestamp (converted to ET) into a US session."""
        try:
            et = ts.tz_convert("America/New_York") if ts.tzinfo else ts
            hm = et.hour * 60 + et.minute
            if hm < 9 * 60 + 30:
                return "pre-market"
            if hm < 16 * 60:
                return "market hours"
            return "after hours"
        except Exception:
            return "intraday"

    def fetch_prices(self):
        """Fetch live prices via yfinance for all tickers individually."""
        import yfinance as yf

        tickers = list(self.positions.keys())
        yahoo_tickers = []
        # Rename map: yahoo symbol -> canonical Holdings.md symbol (results are always stored
        # under the canonical key). yfinance wants dashes for dashed share classes.
        yf_alias = {"BRK.B": "BRK-B"}
        crypto_map = {}
        for t in tickers:
            if t == "BTC.COIN":
                yahoo_tickers.append("BTC-USD")
                crypto_map["BTC-USD"] = "BTC.COIN"
            elif t == "XRP.COIN":
                yahoo_tickers.append("XRP-USD")
                crypto_map["XRP-USD"] = "XRP.COIN"
            else:
                # yfinance wants dashes for dashed share classes (BRK.B -> BRK-B); store the
                # result back under the canonical Holdings.md symbol via crypto_map.
                yt = yf_alias.get(t, t)
                yahoo_tickers.append(yt)
                crypto_map[yt] = t

        if self.brief_mode:
            log(f"  Fetching {len(yahoo_tickers)} tickers...")

        for yt in yahoo_tickers:
            real_ticker = crypto_map.get(yt, yt)
            try:
                tk = yf.Ticker(yt)
                # 5d, NOT 2d. A 2-day window silently drops symbols whose Yahoo daily bar
                # lags (mutual-fund NAVs post late), and it fails *by time of day*: the same
                # run priced 61/62 at 08:00 and 55/62 at 12:54 on 2026-09-23, a phantom
                # $33.7k "loss". 5d clears weekends/holidays and the NAV lag.
                hist = tk.history(period="5d")
                if hist.empty:
                    self.errors.append(f"{yt}: no data")
                    continue
                close_series = hist["Close"].dropna()
                if close_series.empty:
                    self.errors.append(f"{yt}: no valid close prices")
                    continue
                # Last COMPLETED session = the previous close. If today's daily bar already
                # exists (i.e. we are after the close), step back one so we compare like with like.
                series = close_series
                try:
                    today_et = datetime.now(ZoneInfo("America/New_York")).date()
                    last_is_today = len(series) > 1 and series.index[-1].date() == today_et
                except Exception:
                    last_is_today = False
                prev = float(series.iloc[-2] if last_is_today else series.iloc[-1])

                # LIVE price: intraday bars INCLUDING extended hours. This is what makes the
                # 08:00/08:30/09:00/09:15 briefings differ from one another — daily closes do not
                # move before the open, so without this every slot re-reported the same numbers.
                price = float(series.iloc[-1])
                session = "previous close"
                as_of = None
                try:
                    intra = tk.history(period="1d", interval="1m", prepost=True)
                    if not intra.empty:
                        ic = intra["Close"].dropna()
                        if not ic.empty:
                            price = float(ic.iloc[-1])
                            as_of = ic.index[-1]
                            session = self._session_of(as_of)
                except Exception:
                    pass

                chg = float((price - prev) / prev * 100) if prev else 0.0
                self.prices[real_ticker] = {
                    "price": price,
                    "prev_close": prev,
                    "change_pct": chg,
                }
                self.sessions[real_ticker] = session
                if as_of is not None:
                    self.as_of[real_ticker] = as_of
            except Exception as e:
                self.errors.append(f"{yt}: {e}")

    def get_sector(self, ticker):
        for sector, tickers in SECTORS.items():
            if ticker.upper() in tickers:
                return sector
        return "Other"

    def latest_close_time(self):
        """Truthful freshness label, derived from the data rather than from the wall clock.

        Replaces a hardcoded `if now.hour >= 13` UTC heuristic that labelled every run
        "previous close" before 09:00 ET no matter what the data actually held — the reason
        four consecutive briefings looked identical.
        """
        if not self.sessions:
            return "no prices"
        live = [t for t, s in self.sessions.items() if s != "previous close"]
        if not live:
            return "previous close"
        counts = {}
        for t in live:
            counts[self.sessions[t]] = counts.get(self.sessions[t], 0) + 1
        dominant = max(counts, key=counts.get)
        if not self.as_of:
            return f"{dominant} ({len(live)}/{len(self.sessions)} live)"
        try:
            newest = max(self.as_of.values())
            et = newest.tz_convert("America/New_York") if newest.tzinfo else newest
            return (f"{dominant} — {len(live)}/{len(self.sessions)} live, "
                    f"last bar {et.strftime('%H:%M')} ET")
        except Exception:
            return f"{dominant} ({len(live)}/{len(self.sessions)} live)"

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
                        headlines.append((ticker, title, url, _pub_epoch(content)))
                        per_ticker += 1
                        if per_ticker >= 2:  # max 2 per ticker
                            break
            except Exception as e:
                self.errors.append(f"News fetch {ticker}: {e}")
        return headlines[:max_results]

    def fetch_marketwatch(self):
        """Fetch premium MarketWatch market data + headlines via Camofox browser session.

        Returns dict with 'markets' and 'headlines' keys, or None if unavailable.
        MarketWatch access requires the Camofox browser running on :9377 with the
        Dow Jones SSO session active (ddavislogi@gmail.com)."""
        try:
            import subprocess
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marketwatch_data.py")
            if not os.path.exists(script):
                return None
            r = subprocess.run(
                ["/usr/bin/python3", script, "--all"],
                capture_output=True, text=True, timeout=90,
            )
            if r.returncode != 0:
                return None
            return json.loads(r.stdout)
        except Exception:
            return None

    def generate(self):
        now_str = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        brief = []

        # Header
        # Same bug class as the one fixed in portfolio_analysis.py: a hardcoded DST guess
        # (`-5 if tm_isdst == 0 else -4`) added on top of an ALREADY-LOCAL timestamp, which
        # shifts the date in the small hours. Pin the IANA zone instead.
        et = datetime.now(ZoneInfo("America/New_York"))
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
        headlines = []
        mkts = {}
        news_tickers = list(mover_tickers | set(top_tickers))
        if news_tickers and FIRECRAWL_KEY:
            headlines = self.fetch_news(news_tickers, max_results=5)
            if headlines:
                brief.append("**📰 Recent Headlines:**")
                for ticker, title, url, _pub in headlines:
                    brief.append(f"  [{ticker}] [{title}]({url})")
                brief.append("")

        # MarketWatch premium data (markets snapshot + top headlines)
        mw = self.fetch_marketwatch()
        if mw:
            mkts = mw.get("markets") or {}
            hdls = mw.get("headlines") or []
            if mkts:
                brief.append("**🌐 MarketWatch Snapshot:**")
                for label in ("Dow", "S&P 500", "Nasdaq", "10Y Treasury", "WTI", "Gold", "Silver", "Bitcoin", "VIX"):
                    if label in mkts:
                        brief.append(f"  {label}: {mkts[label]}")
                brief.append("")
            if hdls:
                brief.append("**📈 MarketWatch Headlines:**")
                for h in hdls[:5]:
                    brief.append(f"  • [{h['title']}]({h['url']})")
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

        # Machine-readable summary for the duplicate-suppression decision (should_deliver).
        self.summary = {
            "value": round(total_value, 2),
            "priced": priced_count,
            "movers": sorted(mover_tickers),
            "errors": len([e for e in self.errors if not e.startswith("News fetch")]),
            "headlines": sorted(h[1] for h in headlines),
            "newest_pub": max((h[3] for h in headlines), default=0.0),
            "mw": sorted((k, str(v)) for k, v in mkts.items()),
        }

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

    today_iso = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    deliver, why = should_deliver(b, today_iso)

    if not deliver:
        # Empty stdout: a no_agent cron job delivers nothing. Staying silent IS the feature.
        sys.stderr.write(f"dedup: {why} — staying silent\n")
        return

    s = b.summary
    seen_before = set((_load_state().get("headlines_seen") or []))
    _save_state({
        "date": today_iso,
        "value": s["value"],
        "priced": s["priced"],
        "errors": s["errors"],
        "movers": s["movers"],
        "headlines": s["headlines"],
        "newest_pub": s["newest_pub"],
        "headlines_seen": sorted(seen_before | set(s["headlines"])),
        "mw": s["mw"],
        "delivered_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
    })
    sys.stderr.write(f"deliver: {why}\n")
    print(result)


if __name__ == "__main__":
    main()
