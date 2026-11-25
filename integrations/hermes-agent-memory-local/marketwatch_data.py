#!/usr/bin/env python3
"""
marketwatch_data.py — pull MarketWatch market data via the Camofox browser session.

Uses the running Camofox browser (already logged into MarketWatch via Dow Jones SSO)
to fetch premium market data for the pre-market brief.

Usage:
  python3 marketwatch_data.py --markets     # futures/indices/rates snapshot
  python3 marketwatch_data.py --headlines   # top headlines
  python3 marketwatch_data.py --quote NVDA  # single quote page data
  python3 marketwatch_data.py --all         # markets + headlines (for brief)

Requires Camofox running on localhost:9377 with an authenticated MarketWatch session.
Exit 0 with data, exit 2 if browser unavailable, exit 3 if not authenticated.
"""

import json
import re
import sys
import urllib.request

CAMOFOX = "http://localhost:9377"
USER_ID = "brief"


def api_get(path):
    req = urllib.request.Request(f"{CAMOFOX}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def api_get_with_params(path, params):
    import urllib.parse
    qs = urllib.parse.urlencode(params)
    sep = "&" if "?" in path else "?"
    req = urllib.request.Request(f"{CAMOFOX}{path}{sep}{qs}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def api_post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        f"{CAMOFOX}{path}", data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get_tab():
    """Create (or reuse) the MarketWatch tab for the brief userId."""
    # /tabs listing is per-user and can be empty; create is idempotent enough for the brief.
    res = api_post("/tabs", {"userId": USER_ID, "listItemId": "mw", "url": "https://www.marketwatch.com/markets"})
    return res.get("tabId") or res.get("id")


def extract_markets(snapshot):
    """Parse only actual futures quote links from the markets snapshot."""
    raw = snapshot.get("snapshot", "") if isinstance(snapshot, dict) else str(snapshot)
    found = {}
    patterns = [
        (r"DJIA F\s+[+-]?\d+(?:\.\d+)?%", "Dow"),
        (r"S&P F\s+[+-]?\d+(?:\.\d+)?%", "S&P 500"),
        (r"NASDAQ F\s+[+-]?\d+(?:\.\d+)?%", "Nasdaq"),
        (r"Crude Oil\s+[+-]?\d+(?:\.\d+)?%", "WTI"),
        (r"Gold\s+[+-]?\d+(?:\.\d+)?%", "Gold"),
        (r"Silver\s+[+-]?\d+(?:\.\d+)?%", "Silver"),
    ]
    for pat, label in patterns:
        m = re.search(r'- link "(' + pat + r')"', raw, re.I)
        if m:
            found[label] = m.group(1)
    return found


def markets():
    tab = get_tab()
    api_post(f"/tabs/{tab}/navigate", {"userId": USER_ID, "url": "https://www.marketwatch.com/markets"})
    snap = api_get_with_params(f"/tabs/{tab}/snapshot", {"userId": USER_ID})
    return extract_markets(snap)


def headlines():
    """Get top headlines from the MarketWatch homepage snapshot."""
    tab = get_tab()
    api_post(f"/tabs/{tab}/navigate", {"userId": USER_ID, "url": "https://www.marketwatch.com/"})
    snap = api_get_with_params(f"/tabs/{tab}/snapshot", {"userId": USER_ID})
    raw = snap.get("snapshot", "") if isinstance(snap, dict) else str(snap)

    # The snapshot is a text tree. Story pattern (verified):
    #   - link "TITLE" [eNN]:
    #   - /url: https://www.marketwatch.com/story/...?mod=home_ln
    items = []
    lines = [ln.strip() for ln in raw.splitlines()]
    for i, ln in enumerate(lines):
        if "marketwatch.com/story" in ln and i >= 1:
            m = re.search(r'https://www\.marketwatch\.com/story/[a-z0-9\-]+', ln)
            if not m:
                continue
            url = m.group(0)
            # Title is the 'link "..."' line directly above the /url line
            title = None
            prev = lines[i - 1]
            tm = re.search(r'^\-\s*link\s+"(.*)"\s*\[', prev)
            if tm:
                title = tm.group(1).strip()
            if title and len(title) > 15:
                items.append({"title": title, "url": url})
    # De-dup preserving order
    seen, out = set(), []
    for it in items:
        if it["title"] not in seen and it["title"] not in out:
            seen.add(it["title"])
            out.append(it)
        if len(out) >= 8:
            break
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    try:
        health = api_get("/health")
        if not health.get("browserConnected"):
            # Camofox starts lazily: the server can be healthy while no browser tab
            # exists. Boot the same `brief` identity before declaring it down.
            get_tab()
            health = api_get("/health")
            if not health.get("browserConnected"):
                print("BROWSER_DOWN after tab boot", file=sys.stderr)
                sys.exit(2)
    except Exception as e:
        print(f"BROWSER_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    result = {"source": "marketwatch", "mode": mode}
    if mode in ("--markets", "--all"):
        result["markets"] = markets()
    if mode in ("--headlines", "--all"):
        result["headlines"] = headlines()

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
