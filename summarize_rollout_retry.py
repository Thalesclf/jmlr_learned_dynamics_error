"""Summarize renewed rollout-training validation sweep."""
import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root")
a = p.parse_args()

rows = []
for path in sorted(Path(a.root).glob("H*_lambda_*/results_validation.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    by = {x["method"]: x for x in data}
    if "one_step" not in by or "rollout" not in by:
        continue

    m = re.search(r"H(\d+)_lambda_([0-9p]+)", path.parent.name)
    if not m:
        continue
    H = int(m.group(1))
    lam = float(m.group(2).replace("p", "."))

    base = float(by["one_step"]["selection_score"])
    roll = float(by["rollout"]["selection_score"])
    diff = 100.0 * (roll / base - 1.0)
    rows.append({
        "H": H, "lambda": lam, "one": base, "roll": roll, "diff": diff,
        "step_one": int(by["one_step"]["selected_step"]),
        "step_roll": int(by["rollout"]["selected_step"]),
        "path": str(path.parent),
    })

if not rows:
    raise SystemExit("No complete rollout-sweep results found.")

rows.sort(key=lambda r: (r["roll"], r["H"], r["lambda"]))

print(" H   lambda      one_step_val       rollout_val      %diff       step(one/roll)")
for r in rows:
    print(
        f"{r['H']:>2}   {r['lambda']:<8.4g} "
        f"{r['one']:<17.10g} {r['roll']:<17.10g} "
        f"{r['diff']:>+9.3f}%      {r['step_one']:>2}/{r['step_roll']:<2}"
    )

best = rows[0]
print("\nBEST ROLLOUT CONFIGURATION ON VALIDATION")
print(f"H={best['H']}, lambda={best['lambda']}")
print(f"rollout score={best['roll']:.10g}")
print(f"matched one-step score={best['one']:.10g}")
print(f"change vs matched one-step={best['diff']:+.3f}%")
print(f"rollout selected step={best['step_roll']}")
print(f"folder={best['path']}")

# Also report the best configuration that actually improves over its matched one-step arm.
improving = [r for r in rows if r["roll"] < r["one"]]
if improving:
    b = improving[0]
    print("\nBEST CONFIGURATION THAT BEATS MATCHED ONE-STEP")
    print(f"H={b['H']}, lambda={b['lambda']}, change={b['diff']:+.3f}%")
else:
    print("\nNo tested rollout configuration beats its matched one-step arm on validation.")
