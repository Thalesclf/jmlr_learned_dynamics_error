"""Aggregate matched intervention arms across seeds without treating windows as seeds."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("root", nargs="?", default="results_intervention")
a=p.parse_args()
items=[]
for path in sorted(Path(a.root).glob("*/seed_*/results.json")):
    items.extend(json.loads(path.read_text(encoding="utf-8")))
if not items:
    raise SystemExit("No results.json found. Expected results_intervention/ks/seed_0/results.json etc.")
for system in sorted(set(x["system"] for x in items)):
    print(f"\nSYSTEM: {system}")
    by_seed={}
    for x in items:
        if x["system"] == system:
            by_seed.setdefault(x["seed"],{})[x["method"]]=x
    print(f"Seeds: {sorted(by_seed)}; methods may be incomplete")
    methods=sorted(set(x["method"] for x in items))
    hkeys=[k for k in items[0]["test"] if k.startswith("h") and k.endswith("_rel_rmse")]
    print("Method             n seeds  " + "  ".join(f"{k:>16}" for k in hkeys))
    for method in methods:
        mm=[x for x in items if x["system"]==system and x["method"]==method]
        print(f"{method:<20}{len(mm):<8}" + "  ".join(f"{np.median([x['test'][h] for x in mm]):>16.6g}" for h in hkeys))
    if "one_step" in methods:
        print("Paired % difference in test RMSE relative to one_step (negative is better):")
        for method in methods:
            if method=="one_step": continue
            subset=[s for s,d in by_seed.items() if method in d and "one_step" in d]
            if not subset: continue
            vals=[]
            for h in hkeys:
                improvements=[100*(by_seed[s][method]["test"][h]/by_seed[s]["one_step"]["test"][h]-1) for s in subset]
                vals.append(f"{np.median(improvements):>+10.3f}%")
            print(f"{method:<20} n={len(subset)} " + " ".join(vals))
