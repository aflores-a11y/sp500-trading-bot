from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=2)

for av in ib.accountValues():
    if av.tag == "NetLiquidation" and av.currency == "USD":
        print(f"Net Liquidation: ${float(av.value):,.2f}")

print()
print("Open Positions:")
found = False
for pos in ib.positions():
    if pos.position != 0:
        found = True
        c = pos.contract
        right = getattr(c, "right", "")
        strike = getattr(c, "strike", "")
        exp = getattr(c, "lastTradeDateOrContractMonth", "")
        print(f"  {c.symbol} {right} strike={strike} exp={exp} qty={pos.position} paid=${pos.avgCost:.2f}")

if not found:
    print("  No open positions")

ib.disconnect()
