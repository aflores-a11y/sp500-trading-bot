"""Quick diagnostic: compares IB actual positions vs options_positions.json."""
import json
from ib_insync import IB, util

util.logToConsole(level=0)  # suppress ib_insync noise

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=99)

# Fetch all positions from IB
ib_options = {}
for pos in ib.positions():
    c = pos.contract
    if c.secType == "OPT" and pos.position != 0:
        key = f"{c.symbol}_{c.right.lower()}"
        ib_options[key] = {
            "symbol":   c.symbol,
            "right":    "call" if c.right == "C" else "put",
            "strike":   c.strike,
            "expiry":   c.lastTradeDateOrContractMonth,
            "qty":      pos.position,
            "avg_cost": round(pos.avgCost, 2),
            "conid":    c.conId,
        }

ib.disconnect()

# Load JSON state
with open("results/options_positions.json") as f:
    json_positions = json.load(f)

print("\n=== IB ACTUAL OPTIONS POSITIONS ===")
if ib_options:
    for k, v in ib_options.items():
        print(f"  {v['symbol']:>6} {v['right']:<4} | strike={v['strike']} | exp={v['expiry']} | qty={v['qty']} | avgCost=${v['avg_cost']}")
else:
    print("  (none)")

print("\n=== JSON TRACKED POSITIONS ===")
for k, v in json_positions.items():
    in_ib = "✓ IN IB" if k in ib_options or f"{v['ticker']}_{'c' if v['type']=='call' else 'p'}" in ib_options else "✗ NOT IN IB"
    # Check by conid too
    conid_match = any(p["conid"] == v["ib_conid"] for p in ib_options.values()) if v["ib_conid"] else False
    status = "[OK] IN IB" if conid_match else "[!!] NOT IN IB"
    print(f"  {v['ticker']:>6} {v['type']:<4} | strike={v['strike']} | conid={v['ib_conid']} | {status}")

print("\n=== SUMMARY ===")
json_conids = {v["ib_conid"] for v in json_positions.values() if v["ib_conid"]}
ib_conids   = {v["conid"] for v in ib_options.values()}

phantom = [k for k, v in json_positions.items() if v["ib_conid"] not in ib_conids or not v["ib_conid"]]
missing = [k for k, v in ib_options.items() if v["conid"] not in json_conids]

if phantom:
    print(f"  PHANTOM (in JSON but not IB): {phantom}")
else:
    print("  No phantom positions.")

if missing:
    print(f"  UNTRACKED (in IB but not JSON): {missing}")
else:
    print("  No untracked IB positions.")
