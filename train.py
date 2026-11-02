"""Pipeline: generate synthetic data (if no data) -> clean -> evaluate (time-aware) -> fit final -> save."""
import json, sys, joblib
from pathlib import Path
from src.generate_data import generate
from src.data_prep import clean
from src.model import fit, evaluate

raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw_prices_SYNTHETIC.csv"
if not Path(raw).exists():
    generate(raw)
df, rep = clean(raw)
codes = {d: i for i, d in enumerate(sorted(df["district"].unique()))}
df.to_csv("data/clean_prices.csv", index=False)
summary, res = evaluate(df, codes)
res.to_csv("models/backtest_results.csv", index=False)
final = fit(df, codes)
joblib.dump({"model": final, "codes": codes}, "models/rf_model.joblib")
summary["synthetic_data"] = "SYNTHETIC" in raw
json.dump(summary, open("models/metrics.json", "w"), indent=2)
json.dump(rep, open("data/cleaning_report.json", "w"), indent=2, default=str)
print(json.dumps(rep, indent=1, default=str)); 
for m, v in summary["overall"].items(): print(f"{m:15s}", {k: round(x, 2) for k, x in v.items()})
