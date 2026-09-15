#!/usr/bin/env python3
"""
portfolio_history.py — append a dated snapshot of the portfolio and regenerate a
running Mermaid chart inside the Obsidian vault.

Writes two artifacts:
  1) ~/Dropbox/Obsidian/Business/02_Areas/Finance/Portfolio History.csv   (append-only time series)
  2) ~/Dropbox/Obsidian/Business/02_Areas/Finance/Portfolio Performance.md (chart, rebuilt each run)

Idempotent per calendar day: re-running on the same date replaces that day's row
rather than appending a duplicate.

Usage:
  python3 portfolio_history.py            # snapshot + rebuild note
  python3 portfolio_history.py --quiet    # no stdout unless something fails
"""

import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from datetime import datetime

VAULT = os.path.expanduser("~/Dropbox/Obsidian/Business/02_Areas/Finance")
CSV_PATH = os.path.join(VAULT, "Portfolio History.csv")
NOTE_PATH = os.path.join(VAULT, "Portfolio Performance.md")
BM_CSV = os.path.join(VAULT, "Benchmark History.csv")
CHART_BLOCK_PATH = os.path.expanduser("~/.hermes/scripts/portfolio_chart_block.md")

# Empower's own class proxies (Index Descriptions panel)
BENCH = [("US Equity", ["VTI"]), ("Foreign Equity", ["VEU"]), ("US Bond", ["AGG"]),
         ("Foreign Bond", ["IGOV"]), ("Alternatives", ["VNQ", "IAU", "DBC"])]
HOLDINGS = os.path.expanduser("~/Dropbox/Obsidian/Business/Holdings.md")
QUIET = "--quiet" in sys.argv

# Empower's own asset-class proxies (from the Index Descriptions panel)
CLASSES = ["US Equity", "Foreign Equity", "US Bond", "Foreign Bond", "Alternatives", "Crypto", "Cash"]

MAP = {
    "US Equity": {"SPY","SCHG","SCHK","ONEQ","FVAL","VTI","VMCIX","VSCIX","ESGV","BRK.B","GOOG","AMZN",
        "NVDA","QQQ","XLK","CLOU","CIBR","ONDS","BBAI","XLV","SOXX","SMH","SDY","DON","DES","VIG","DIVO",
        "VEIRX","F","NAT","GRID","IFRA","ITA"},
    "Foreign Equity": {"VSGX","VEA","VWO","VEMAX","VTSNX","IDV","DEM","DAX","EUAD","SAFRF","BAESF",
        "EADSF","RYCEY","FINMF","RNMBF","TKMSF","TYEKF","ZIM","KTEC","ACHR"},
    "US Bond": {"BND","VCEB","VGIT","VGSH"},
    "Foreign Bond": {"BNDX"},
    "Alternatives": {"VNQ","VGSLX","SIVR","FXE"},
    "Crypto": {"BTC.COIN","XRP.COIN"},
    "Cash": {"SWVXX"},
}
ALIAS = {"BRK.B": "BRK-B", "BTC.COIN": "BTC-USD", "XRP.COIN": "XRP-USD"}
_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE


def log(*a):
    if not QUIET:
        print(*a)


def load_positions():
    text = open(HOLDINGS).read()
    pos = {}
    for m in re.finditer(r'\| +([A-Z][A-Z0-9.]*) +\| +([^|]+?) +\| +([\d,.]+) +\|', text):
        t = m.group(1).strip()
        try:
            s = float(m.group(3).replace(",", ""))
        except Exception:
            continue
        if s > 0:
            pos[t] = pos.get(t, 0.0) + s
    return pos


def yf_close(sym):
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="2d")["Close"].dropna()
        return float(h.iloc[-1]) if len(h) else None
    except Exception:
        return None


def classify(t):
    for c, ts in MAP.items():
        if t.upper() in ts:
            return c
    return "Other"


def snapshot():
    pos = load_positions()
    vals = {c: 0.0 for c in CLASSES}
    vals["Other"] = 0.0
    priced = errors = 0
    for t, sh in pos.items():
        p = yf_close(ALIAS.get(t, t))
        if p is None:
            errors += 1
            continue
        priced += 1
        vals[classify(t)] += sh * p
    return vals, priced, errors


def write_csv(row):
    os.makedirs(VAULT, exist_ok=True)
    cols = ["date", "total"] + [c.lower().replace(" ", "_") for c in CLASSES] + ["other", "priced", "errors"]
    existing = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("date") != row["date"]]
    existing.append(row)
    existing.sort(key=lambda r: r["date"])
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in cols})
    os.replace(tmp, CSV_PATH)          # atomic; survives a Dropbox hiccup mid-write
    return existing, cols


def mermaid_chart(rows, cols):
    dates = [r["date"][5:] for r in rows]           # MM-DD
    totals = [round(float(r["total"])) for r in rows]
    lo, hi = min(totals), max(totals)
    pad = max(2000, int((hi - lo) * 0.15)) or 2000
    x = "[" + ", ".join(f'"{d}"' for d in dates) + "]"
    return (f'```mermaid\nxychart-beta\n'
            f'    title "Portfolio value (USD)"\n'
            f'    x-axis {x}\n'
            f'    y-axis "Value" {lo - pad} --> {hi + pad}\n'
            f'    line {totals}\n```')


def build_note(rows):
    try:
        chart_block = open(CHART_BLOCK_PATH).read().strip()
    except Exception:
        chart_block = "_chart template missing at %s_" % CHART_BLOCK_PATH
    latest = rows[-1]
    first = rows[0]
    total = float(latest["total"])
    chg = total - float(first["total"])
    pct = (chg / float(first["total"]) * 100) if float(first["total"]) else 0

    byc = "\n".join(
        f"| {c} | ${float(latest.get(c.lower().replace(' ','_'), 0) or 0):,.0f} |"
        f" {(float(latest.get(c.lower().replace(' ','_'),0) or 0)/total*100 if total else 0):.1f}% |"
        for c in CLASSES)

    hist = "\n".join(
        f"| {r['date']} | ${float(r['total']):,.0f} | " +
        " | ".join(f"${float(r.get(c.lower().replace(' ','_')) or 0):,.0f}" for c in CLASSES) + " |"
        for r in rows[-15:])

    return f"""---
title: Portfolio Performance
created: {first['date']}
updated: {latest['date']}
type: financial
tags: [finance, portfolio, performance, chart]
confidence: high
---

# Portfolio Performance

_Running snapshot maintained by `portfolio_history.py`. The chart and tables below are
**regenerated on every run** — do not hand-edit them._

**Book value:** **${total:,.0f}** &nbsp;·&nbsp; since {first['date']}: **{chg:+,.0f}** ({pct:+.2f}%)
&nbsp;·&nbsp; snapshots: **{len(rows)}**

## Total value

{mermaid_chart(rows, None)}

## Current allocation (Empower's five classes)

| Asset class | Value | Weight |
|---|---:|---:|
{byc}

## Snapshot log (last 15)

| Date | Total | {' | '.join(CLASSES)} |
|---|---:|---:|---:|---:|---:|---:|
{hist}

## Class vs benchmark — interactive

{chart_block}

---

_Source: `Holdings.md` share counts × live yfinance prices. Asset-class mapping per Empower's
Index Descriptions (US Equity≈VTI, Foreign≈VEU, US Bond≈AGG, Foreign Bond≈IGOV,
Alternatives≈VNQ/IAU/DBC). Crypto and cash are tracked outside the five-class benchmark._
"""


def write_benchmark(weeks=14):
    """Write cumulative index levels (base 100) for Empower's class proxies so the
    interactive chart can compare the user's classes against the right benchmark."""
    import yfinance as yf
    syms = sorted({s for _, ts in BENCH for s in ts})
    try:
        df = yf.download(syms, period=f"{weeks + 1}wk", interval="1wk",
                         progress=False, auto_adjust=True)["Close"]
    except Exception:
        return 0
    if df is None or getattr(df, "empty", True):
        return 0
    df = df.ffill()
    cols = ["date"] + [c.lower().replace(" ", "_") for c, _ in BENCH]
    rows, base = [], None
    for dt, r in df.iterrows():
        vals = {}
        for cname, syms_ in BENCH:
            ps = []
            for s in syms_:
                if s not in df.columns:
                    continue
                try:
                    v = float(r[s])
                except Exception:
                    continue
                if v == v and v > 0:
                    ps.append(v)
            if not ps:
                break
            vals[cname] = sum(ps) / len(ps)
        if len(vals) != len(BENCH):
            continue
        if base is None:
            base = dict(vals)
        rows.append({"date": dt.strftime("%Y-%m-%d"),
                     **{c.lower().replace(" ", "_"): round(vals[c] / base[c] * 100, 3)
                        for c in vals}})
    tmp = BM_CSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, BM_CSV)
    return len(rows)


def read_rows():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH) as f:
        return sorted(csv.DictReader(f), key=lambda r: r["date"])


def backfill(weeks=13):
    """Seed weekly historical rows (share counts × historical closes) so the chart
    has shape on day one instead of a single point. Idempotent per date."""
    import yfinance as yf
    pos = load_positions()
    if not pos:
        return 0
    syms = sorted({ALIAS.get(t, t) for t in pos})
    try:
        df = yf.download(syms, period=f"{weeks + 1}wk", interval="1wk",
                         progress=False, auto_adjust=True)["Close"]
    except Exception:
        return 0
    if df is None or getattr(df, "empty", True):
        return 0
    df = df.ffill()
    added = 0
    for dt, rowp in df.iterrows():
        d = dt.strftime("%Y-%m-%d")
        vals = {c: 0.0 for c in CLASSES}
        vals["Other"] = 0.0
        for t, sh in pos.items():
            s = ALIAS.get(t, t)
            if s not in df.columns:
                continue
            p = rowp.get(s)
            try:
                p = float(p)
            except Exception:
                continue
            if p != p or p <= 0:            # NaN guard
                continue
            vals[classify(t)] += sh * p
        total = sum(vals.values())
        if total <= 0:
            continue
        rec = {"date": d, "total": round(total, 2), "priced": len(pos), "errors": 0}
        for c in CLASSES:
            rec[c.lower().replace(" ", "_")] = round(vals[c], 2)
        rec["other"] = round(vals["Other"], 2)
        write_csv(rec)
        added += 1
    return added


def main():
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    if "--backfill" in sys.argv:
        nb = write_benchmark(14)
        n = backfill(13)
        rows = read_rows()
        if rows:
            tmp = NOTE_PATH + ".tmp"
            open(tmp, "w").write(build_note(rows))
            os.replace(tmp, NOTE_PATH)
        log(f"✅ backfilled {n} weekly rows -> {len(rows)} total snapshots "
            f"-> {NOTE_PATH}  [{time.time()-t0:.1f}s]")
        return

    vals, priced, errors = snapshot()
    total = sum(vals.values())
    if priced == 0:
        log("❌ no positions priced — aborting")
        sys.exit(1)
    row = {"date": today, "total": round(total, 2),
           "priced": priced, "errors": errors}
    for c in CLASSES:
        row[c.lower().replace(" ", "_")] = round(vals[c], 2)
    row["other"] = round(vals["Other"], 2)

    write_benchmark(14)
    rows, cols = write_csv(row)
    note = build_note(rows)
    tmp = NOTE_PATH + ".tmp"
    open(tmp, "w").write(note)
    os.replace(tmp, NOTE_PATH)
    log(f"✅ {today}  total ${total:,.0f}  ({priced} priced, {errors} errors)  "
        f"{len(rows)} snapshots -> {NOTE_PATH}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
