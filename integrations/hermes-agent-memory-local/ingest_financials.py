#!/usr/bin/env python3
"""Parse Empower clippings, compute dashboard, write Holdings.md + OB1 seed."""
import json, re, os, sys, time, urllib.request

DROPBOX = os.path.expanduser("~/Dropbox/Obsidian/Business")
CLIPPINGS = f"{DROPBOX}/Clippings"
HOLDINGS_FILE = f"{DROPBOX}/Clippings/Empower Personal Holdings.md"
ASSETS_FILE = f"{DROPBOX}/Clippings/Empower Personal Assets.md"
OUTPUT = f"{DROPBOX}/Holdings.md"
CHROMA = "http://localhost:8000"
COLL_BASE = f"{CHROMA}/api/v2/tenants/default_tenant/databases/default_database/collections"
COLL = "ob1_memory"
OLLAMA = "http://localhost:11434/api/embeddings"

def embed(text):
    d = json.dumps({"model": "nomic-embed-text", "prompt": text[:8000]}).encode()
    req = urllib.request.Request(OLLAMA, data=d, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("embedding", [])

def get_cid():
    try:
        r = json.loads(urllib.request.urlopen(f"{COLL_BASE}?name={COLL}", timeout=5).read())
        for c in (r if isinstance(r, list) else []):
            if c["name"] == COLL:
                return c["id"]
    except: pass
    c = json.loads(urllib.request.urlopen(
        urllib.request.Request(COLL_BASE,
            data=json.dumps({"name": COLL, "metadata": {"hnsw:space": "cosine"}}).encode(),
            method="POST", headers={"Content-Type": "application/json"}), timeout=5).read())
    return c.get("id", "")

def read_clippings():
    """Parse holdings from Empower clippings.
    
    Format: TICKER (blank) Name (blank) Shares (blank) Price (blank) Change% (blank) Change$ (blank) Value (blank)
    """
    holdings = []
    
    htext = open(HOLDINGS_FILE).read()
    body = htext.split("---", 2)[-1].strip() if htext.startswith("---") else htext
    # Remove blank lines to get a compressed view
    lines = [l.strip() for l in body.strip().split("\n") if l.strip()]
    
    # Find header positions
    header_indices = {}
    for idx, line in enumerate(lines):
        if line in ("Holding", "Shares", "Price", "Change", "1 day $", "Value"):
            header_indices[line] = idx
    
    # Data starts after the last header (Value is the last header)
    if "Value" in header_indices:
        data_start = header_indices["Value"] + 1
    else:
        data_start = 6
    
    # Now every 7 items is one holding: Ticker, Name, Shares, Price, Change%, Change$, Value
    data_lines = lines[data_start:]
    
    i = 0
    while i + 6 < len(data_lines):
        ticker = data_lines[i]
        name = data_lines[i+1]
        shares_str = data_lines[i+2].replace(",", "")
        price_str = data_lines[i+3].replace("$", "").replace(",", "")
        # data_lines[i+4] is Change%
        # data_lines[i+5] is Change$
        value_str = data_lines[i+6].replace("$", "").replace(",", "")
        
        try:
            shares = float(shares_str)
        except: shares = 0
        try:
            price = float(price_str)
        except: price = 0
        try:
            value = float(value_str)
        except: value = 0
        
        if value > 0:
            holdings.append({"ticker": ticker, "name": name[:60],
                             "shares": shares, "price": price, "value": value})
        
        i += 7
    
    return holdings

def parse_assets():
    """Parse asset/liability summary."""
    atext = open(ASSETS_FILE).read()
    body = atext.split("---", 2)[-1].strip() if atext.startswith("---") else atext
    
    data = {}
    for m in re.finditer(r'(\w[\w\s&,.(\)\d#]+?)\$([\d,]+)', body):
        label = m.group(1).strip().rstrip("•").strip()
        val = float(m.group(2).replace(",", ""))
        data[label] = val
    
    # Parse key lines
    total = 0
    cash = 0
    investments = 0
    real_estate = 0
    liabilities = 0
    mortgage = 0
    
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("Assets$"):
            total = float(line.replace("Assets$", "").replace(",", ""))
        elif line.startswith("Cash$"):
            cash = float(line.split("$")[1].split("+")[0].replace(",", ""))
        elif line.startswith("Investments$"):
            investments = float(line.split("$")[1].split("+")[0].replace(",", ""))
        elif line.startswith("Liabilities$"):
            liabilities = float(line.split("$")[1].replace(",", ""))
        elif line.startswith("Mortgage$"):
            mortgage = float(line.split("$")[1].split("+")[0].replace(",", ""))
    
    # Real estate from Other section
    other_match = re.search(r'Other\$([\d,]+)', body)
    if other_match:
        real_estate = float(other_match.group(1).replace(",", ""))
    
    return {
        "total_assets": total,
        "cash": cash,
        "investments": investments,
        "real_estate": real_estate,
        "liabilities": liabilities,
        "mortgage": mortgage,
        "net_worth": round(total - liabilities, 2),
    }

def categorize(ticker):
    """Simple sector categorization."""
    t = ticker.upper()
    defense_ae = {"ITA","PPA","XAR","EUAD","SHLD","SAFRF","BAESF","EADSF","RYCEY","FINMF","RNMBF"}
    semi = {"SOXX","SMH","XSD","CHPX","NVDA"}
    tech = {"QQQ","XLK","CLOU","CIBR","KTEC","GOOG","AMZN","ONDS"}
    infra_util = {"GRID","IFRA","GLFOX","GRDU","NAT"}
    dividend = {"VIG","DIVO","VEIRX","DEM","IDV"}
    broad = {"SPY","SCHG","SCHK","ONEQ","FVAL","VSCIX","ESGV"}
    reit = {"VGSLX"}
    crypto = {"BTC.COIN","XRP.COIN"}
    defense = {"ACHR"}
    
    if t in defense_ae: return "Aerospace & Defense"
    if t in semi: return "Semiconductors"
    if t in tech: return "Technology"
    if t in infra_util: return "Infrastructure"
    if t in dividend: return "Dividend Income"
    if t in broad: return "Broad Market"
    if t in reit: return "Real Estate"
    if t in crypto: return "Crypto"
    if t in defense: return "Aerospace & Defense"
    return "Other"

def main():
    holdings = read_clippings()
    assets = parse_assets()
    
    print(f"Holdings parsed: {len(holdings)} positions")
    for h in holdings[:5]:
        print(f"  {h['ticker']:8s} {h['shares']:>8.2f} @ ${h['price']:<8.2f} = ${h['value']:>8.2f}")
    if len(holdings) > 5:
        print(f"  ... and {len(holdings)-5} more")
    print()
    
    total_inv = sum(h['value'] for h in holdings)
    
    # Sector breakdown
    sectors = {}
    for h in holdings:
        cat = categorize(h['ticker'])
        sectors[cat] = sectors.get(cat, 0) + h['value']
    
    print(f"Total investments: ${total_inv:,.2f}")
    print(f"Net worth: ${assets['net_worth']:,.2f}")
    print(f"\nSector breakdown:")
    for s, v in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {s:25s} ${v:>8,.2f} ({v/total_inv*100:.1f}%)")
    
    # Write Holdings.md
    ts = time.strftime("%Y-%m-%d")
    md = f"""---
title: Holdings Dashboard
created: {ts}
updated: {ts}
type: financial
tags: [finance, portfolio, holdings, empower]
confidence: high
---

# Holdings Dashboard

**Source:** Empower Personal Dashboard (clipped {ts})
**Note:** Vanguard 401(k) holdings are tracked within Empower under two accounts ($77,068 + $117,711).

## Summary

| Metric | Value |
|--------|-------|
| Total Assets | ${assets['total_assets']:,.0f} |
| Cash | ${assets['cash']:,.0f} |
| Investments | ${assets['investments']:,.0f} |
| Real Estate | ${assets['real_estate']:,.0f} |
| Liabilities | ${assets['liabilities']:,.0f} |
| Mortgage | ${assets['mortgage']:,.0f} |
| **Net Worth** | **${assets['net_worth']:,.0f}** |

## Sector Allocation

| Sector | Value | % |
|--------|-------|---|
"""
    for s, v in sorted(sectors.items(), key=lambda x: -x[1]):
        pct = v/total_inv*100
        md += f"| {s} | ${v:,.0f} | {pct:.1f}% |\n"
    
    md += f"\n**Total Invested:** ${total_inv:,.0f}\n\n"
    
    # Per-account breakdown from assets
    md += """## Account Breakdown

| Account | Value |
|---------|-------|
"""
    # Parse accounts from assets file
    atext = open(ASSETS_FILE).read()
    abody = atext.split("---", 2)[-1].strip()
    # Extract account lines under Investments section
    in_inv = False
    for line in abody.split("\n"):
        line = line.strip()
        if line.startswith("Investments$"):
            in_inv = True
            continue
        if line.startswith("Other$"):
            in_inv = False
            continue
        if in_inv and "•" in line and "$" in line:
            parts = line.split("$")
            if len(parts) >= 3:
                acct = parts[0].strip().rstrip("•").strip()
                val = parts[1].strip()
                try:
                    val_f = float(val.replace(",", ""))
                    if val_f > 0:
                        md += f"| {acct} | ${val_f:,.0f} |\n"
                except: pass
    
    md += """
## Holdings Detail

| Ticker | Name | Shares | Price | Value |
|--------|------|--------|-------|-------|
"""
    for h in sorted(holdings, key=lambda x: -x['value']):
        md += f"| {h['ticker']} | {h['name'][:40]} | {h['shares']:>8.2f} | ${h['price']:<8.2f} | ${h['value']:>8,.2f} |\n"
    
    md += f"\n_Data sourced from Empower Personal Dashboard on {ts}_\n"
    
    with open(OUTPUT, "w") as f:
        f.write(md)
    print(f"\nWritten: {OUTPUT} ({len(md)} bytes)")
    
    # Seed to OB1
    print("\nSeeding to OB1...")
    cid = get_cid()
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Seed dashboard summary
    summary = f"Financial dashboard as of {ts}: Net worth ${assets['net_worth']:,.0f}, " \
              f"Investments ${total_inv:,.0f}, Cash ${assets['cash']:,.0f}, " \
              f"Real Estate ${assets['real_estate']:,.0f}, Liabilities ${assets['liabilities']:,.0f}"
    vec = embed(summary)
    data = json.dumps({"ids":[f"fin_dashboard_{int(time.time())}"],
        "embeddings":[vec],
        "metadatas":[{"workspace_id":"ddavis","content_type":"financial_dashboard",
                      "timestamp":ts_iso,"use_policy":"evidence","confidence":0.9}],
        "documents":[summary]}).encode()
    req = urllib.request.Request(f"{COLL_BASE}/{cid}/add", data=data, method="POST",
                                 headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=10)
    
    # Seed key positions as individual memories
    for h in sorted(holdings, key=lambda x: -x['value'])[:20]:  # top 20
        doc = f"Position: {h['ticker']} — {h['name']}. {h['shares']} shares @ ${h['price']:.2f} = ${h['value']:,.2f}. Sector: {categorize(h['ticker'])}"
        vec = embed(doc)
        data = json.dumps({"ids":[f"fin_pos_{h['ticker']}_{int(time.time())}"],
            "embeddings":[vec],
            "metadatas":[{"workspace_id":"ddavis","content_type":"position",
                          "ticker":h['ticker'],"value":h['value'],
                          "timestamp":ts_iso,"use_policy":"evidence","confidence":0.9}],
            "documents":[doc]}).encode()
        req = urllib.request.Request(f"{COLL_BASE}/{cid}/add", data=data, method="POST",
                                     headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req, timeout=10)
    
    print(f"Seeded dashboard + {min(len(holdings),20)} positions to OB1")
    print()
    print("=== DONE ===")

if __name__ == "__main__":
    main()
