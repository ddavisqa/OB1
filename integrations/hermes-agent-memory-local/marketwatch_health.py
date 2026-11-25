#!/usr/bin/env python3
"""
marketwatch_health.py — verify the MarketWatch/Camofox session is still authenticated.

Verdict comes from the LIVE PROBE, not from a cookie date: the probe navigates a paywalled
article in the SAME Camofox identity the pre-market brief uses and asserts the body renders.
Cookie expiry is reported as context, never as the verdict.

Exit 0 = session OK (prints status; the cron wrapper stays silent-looking when healthy)
Exit 2 = broken / expiring soon / service unreachable (prints what is needed)

Three failure modes that previously produced false alarms, now handled:
  1. A DOWN browser is a service problem, not an auth failure.
  2. Camoufox launches LAZILY — /health reports browserConnected=false until the first tab
     exists, so we boot a tab before judging the service.
  3. A login stored in a DIFFERENT Camofox profile does nothing for the pipeline. This check
     probes the pipeline's own identity (USER_ID), which must match marketwatch_data.py and
     Hermes' browser.camofox.user_id.
"""

import datetime
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

PROFILES_DIR = os.path.expanduser("~/.camofox/profiles")
CAMOFOX = "http://localhost:9377"
# The pipeline's own reading target: markets + headlines on the homepage. Not one story URL —
# a hardcoded article can be edited or 404'd, which previously faked a BROKEN verdict.
PROBE_URL = "https://www.marketwatch.com/"
WARN_DAYS = 7

# The identity the actual pipeline drives (marketwatch_data.py USER_ID) and the identity Hermes'
# browser tool shares (config: browser.camofox.user_id). Certify the profile that gets used.
USER_ID = "brief"
AUTH_COOKIES = ("session", "sso")   # 'session' is the durable one (~179d); 'sso' rotates ~14d


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def profile_for_user(user_id):
    """Locate the Camofox profile directory bound to a userId (via its meta.json)."""
    if not os.path.isdir(PROFILES_DIR):
        return None
    for d in sorted(os.listdir(PROFILES_DIR)):
        meta = os.path.join(PROFILES_DIR, d, "meta.json")
        if not os.path.exists(meta):
            continue
        try:
            with open(meta) as fh:
                if json.load(fh).get("userId") == user_id:
                    return os.path.join(PROFILES_DIR, d)
        except Exception:
            continue
    return None


def cookie_report(profile_dir=None):
    """Newest expiry per auth-cookie name. Scoped to the CERTIFIED profile when given —
    a stale cookie in some other profile must not colour the verdict."""
    out = {}
    if profile_dir is None:
        if not os.path.isdir(PROFILES_DIR):
            return out
        dirs = [os.path.join(PROFILES_DIR, d) for d in sorted(os.listdir(PROFILES_DIR))]
    else:
        dirs = [profile_dir]
    for d in dirs:
        state = os.path.join(d, "storage-state.json")
        if not os.path.exists(state):
            continue
        try:
            with open(state) as fh:
                data = json.load(fh)
        except Exception:
            continue
        for c in data.get("cookies", []):
            if c.get("name") in AUTH_COOKIES and "dowjones" in c.get("domain", ""):
                exp = c.get("expires", 0)
                if exp and exp > 0 and (c["name"] not in out or exp > out[c["name"]]):
                    out[c["name"]] = exp
    return out


def boot_browser():
    """Return (browser_up, detail). Camoufox starts lazily on first tab creation."""
    try:
        h = json.loads(urllib.request.urlopen(CAMOFOX + "/health", timeout=5).read())
        if h.get("browserConnected"):
            return True, "already running"
        payload = json.dumps({"userId": USER_ID, "listItemId": "mw-health",
                              "url": "https://www.marketwatch.com/"}).encode()
        req = urllib.request.Request(CAMOFOX + "/tabs", data=payload,
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
        time.sleep(4)
        h = json.loads(urllib.request.urlopen(CAMOFOX + "/health", timeout=5).read())
        return bool(h.get("browserConnected")), "booted a tab"
    except Exception as e:                                     # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def live_probe():
    """Read the SAME page the pre-market brief reads, in the pipeline identity, and assert we
    got usable content (>=5 story links) with no paywall markers.

    Deliberately not keyed to one story URL: a hardcoded article can be edited, redirected or
    404'd, which made the old probe report BROKEN while the pipeline was working fine.
    """
    def api_post(path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(CAMOFOX + path, data=data,
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def api_get(path, params):
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{CAMOFOX}{path}?{qs}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    try:
        tab = api_post("/tabs", {"userId": USER_ID, "listItemId": "mw-probe",
                                 "url": PROBE_URL}).get("tabId")
        api_post(f"/tabs/{tab}/navigate", {"userId": USER_ID, "url": PROBE_URL})
        snap = api_get(f"/tabs/{tab}/snapshot", {"userId": USER_ID})
        raw = snap.get("snapshot", "") if isinstance(snap, dict) else str(snap)
        low = raw.lower()
        paywalled = any(x in low for x in ["subscribe to read", "premium article", "unlock this article",
                                           "paywall", "sign in to read", "your free access"])
        stories = len(re.findall(r"marketwatch\.com/story", raw, re.I))
        has_markets = "s&p 500" in low or "dow" in low or "nasdaq" in low
        ok = (not paywalled) and stories >= 5 and has_markets
        return ok, f"paywalled={paywalled} story_links={stories} markets_rendered={has_markets}"
    except Exception as e:                                     # noqa: BLE001
        return False, f"error: {e}"



def main():
    browser_up, bdetail = boot_browser()
    if not browser_up:
        print(f"⚠️ Camofox browser UNAVAILABLE ({bdetail}) — cannot verify MarketWatch live.")
        print("   Service problem, not an auth failure. Start Camofox on localhost:9377 ")
        print("   (LaunchAgent com.hermes.camofox), then re-run this check.")
        sys.exit(2)

    ok, detail = live_probe()
    prof = profile_for_user(USER_ID)
    ck = cookie_report(prof)
    scope = os.path.basename(prof)[:12] if prof else "profile not found"
    ck_line = ", ".join(
        f"{n} {(datetime.datetime.fromtimestamp(e, datetime.timezone.utc)):%Y-%m-%d} "
        f"({(datetime.datetime.fromtimestamp(e, datetime.timezone.utc) - _now()).days}d)"
        for n, e in sorted(ck.items())) or "no dowjones auth cookies found in any profile"

    if ok:
        soonest = min(((datetime.datetime.fromtimestamp(e, datetime.timezone.utc) - _now()).days
                       for e in ck.values()), default=999)
        if soonest > WARN_DAYS:
            # Watchdog contract: say NOTHING when healthy (a no_agent cron delivers stdout
            # verbatim, so a success line would ping the user every week). --verbose for tests.
            if "--verbose" in sys.argv:
                print(f"✅ MarketWatch session OK — probe passed in '{USER_ID}' identity "
                      f"({scope}). {ck_line}")
            sys.exit(0)
        print(f"⚠️ MarketWatch session OK but auth cookies expire soon — {ck_line}")
        print("Re-login soon: sso.accounts.dowjones.com via Camofox (ddavislogi@gmail.com)")
        sys.exit(2)

    print(f"❌ MarketWatch session BROKEN — paywall probe in '{USER_ID}' identity ({scope}): {detail}")
    print(f"   cookies in that profile: {ck_line}")
    print("Fix: re-login at sso.accounts.dowjones.com via Camofox (ddavislogi@gmail.com).")
    print(f"   The login must land in the '{USER_ID}' profile — marketwatch_data.py USER_ID, "
          "Hermes' browser.camofox.user_id and this check must all name the same identity.")
    sys.exit(2)


if __name__ == "__main__":
    main()
