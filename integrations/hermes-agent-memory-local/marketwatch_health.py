#!/usr/bin/env python3
"""
marketwatch_health.py — verify the MarketWatch/Camofox session is still authenticated.

Checks the persisted Camofox profile's sso cookie expiry and does a live
paywall probe (fetch an article page, look for paywall markers).

Exit 0 = session OK (prints status).
Exit 2 = session expiring soon / broken / browser unavailable (prints what's needed).

NOTE: a DOWN browser and a BROKEN login are different failures. When Camofox is
not running we say so plainly instead of telling the user to re-login, because a
valid sso cookie + a dead browser previously produced a false "BROKEN" alarm.
"""

import datetime
import json
import os
import re
import sys
import urllib.request
import urllib.parse

PROFILES_DIR = os.path.expanduser("~/.camofox/profiles")
CAMOFOX = "http://localhost:9377"
PROBE_URL = ("https://www.marketwatch.com/story/u-s-stock-futures-flat-as-chances-of-rate-hike"
             "-rise-after-warshs-jackson-hole-comments-1f5287f2")
WARN_DAYS = 7


def sso_expiry():
    """Scan all Camofox profiles for the Dow Jones sso cookie.
    Returns (expiry_dt, days_left) or None."""
    if not os.path.isdir(PROFILES_DIR):
        return None
    best = None
    for d in os.listdir(PROFILES_DIR):
        state = os.path.join(PROFILES_DIR, d, "storage-state.json")
        if not os.path.exists(state):
            continue
        try:
            with open(state) as f:
                data = json.load(f)
        except Exception:
            continue
        for c in data.get("cookies", []):
            if c.get("name") == "sso" and "dowjones" in c.get("domain", ""):
                exp = c.get("expires", 0)
                if exp and exp > 0:
                    dt = datetime.datetime.fromtimestamp(exp, datetime.timezone.utc)
                    days = (dt - datetime.datetime.now(datetime.timezone.utc)).days
                    if best is None or exp > best[0]:
                        best = (exp, dt, days)
    if best:
        return best[1], best[2]
    return None


def browser_health():
    """Return (browser_up: bool, detail: str). Separates 'browser down' from 'auth broken'."""
    try:
        h = json.loads(urllib.request.urlopen(CAMOFOX + "/health", timeout=5).read())
        return bool(h.get("browserConnected")), "reachable"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def live_probe():
    """Navigate a Camofox tab to a paywalled article and check it renders."""
    try:
        def api_post(path, payload=None):
            data = json.dumps(payload or {}).encode()
            req = urllib.request.Request(CAMOFOX + path, data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())

        def api_get(path, params):
            qs = urllib.parse.urlencode(params)
            req = urllib.request.Request(f"{CAMOFOX}{path}?{qs}", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())

        health = json.loads(urllib.request.urlopen(CAMOFOX + "/health", timeout=5).read())
        if not health.get("browserConnected"):
            return False, "browser not connected"

        tab = api_post("/tabs", {"userId": "health", "listItemId": "mw", "url": "https://www.marketwatch.com/"}).get("tabId")
        api_post(f"/tabs/{tab}/navigate", {"userId": "health", "url": PROBE_URL})
        snap = api_get(f"/tabs/{tab}/snapshot", {"userId": "health"})
        raw = snap.get("snapshot", "") if isinstance(snap, dict) else str(snap)
        low = raw.lower()
        paywalled = any(x in low for x in ["subscribe to read", "premium article", "unlock this article",
                                           "paywall", "sign in to read", "your free access"])
        has_content = "stock-index futures" in low or "futures" in low
        return (not paywalled) and has_content, f"paywalled={paywalled} content={has_content}"
    except Exception as e:
        return False, f"error: {e}"


def main():
    exp = sso_expiry()

    if not exp:
        print("⚠️ MarketWatch session: no sso cookie found (never logged in or profile wiped)")
        print("Fix: log in at sso.accounts.dowjones.com via Camofox browser (ddavislogi@gmail.com)")
        sys.exit(2)

    exp_dt, days = exp
    cookie_line = f"sso cookie expires {exp_dt:%Y-%m-%d} ({days}d)"
    browser_up, bdetail = browser_health()

    # Browser/service problem — do NOT tell the user to re-login over this.
    if not browser_up:
        print(f"⚠️ Camofox browser NOT RUNNING ({bdetail}) — cannot verify MarketWatch live.")
        print(f"   {cookie_line}. This is a service problem, not an auth failure;")
        print("   re-login is only needed once the cookie actually expires.")
        print("   Start Camofox on localhost:9377, then re-run this check.")
        sys.exit(2)

    ok, detail = live_probe()

    if ok and days > WARN_DAYS:
        print(f"✅ MarketWatch session OK — {cookie_line}, paywall probe passed")
        sys.exit(0)
    elif ok:
        print(f"⚠️ MarketWatch session OK but {cookie_line}")
        print("Re-login soon: sso.accounts.dowjones.com via Camofox (ddavislogi@gmail.com)")
        sys.exit(2)
    else:
        print(f"❌ MarketWatch session BROKEN — paywall probe: {detail}")
        print("Fix: re-login at sso.accounts.dowjones.com via Camofox (ddavislogi@gmail.com)")
        sys.exit(2)


if __name__ == "__main__":
    main()
