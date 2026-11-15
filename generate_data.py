"""SYNTHETIC data generator. The output is NOT real market data.
It deliberately contains missing values, duplicates, gaps and outliers
so the data-preparation step has something realistic to clean."""
import numpy as np, pandas as pd
from pathlib import Path

DISTRICTS = {"Pune": 92, "Nashik": 90, "Satara": 88, "Ahmednagar": 89, "Solapur": 86}

def generate(path="data/raw_prices_SYNTHETIC.csv", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-11-01", "2026-10-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    seas = 5*np.cos(2*np.pi*(doy-15)/365) - 4*np.exp(-((doy-225)/20)**2)  # winter high, Shravan dip
    feed = 100 + np.cumsum(rng.normal(0, 0.35, n))                        # random-walk feed index
    season = np.select([dates.month.isin([12,1,2]), dates.month.isin([3,4,5]), dates.month.isin([6,7,8,9])],
                       ["Winter", "Summer", "Monsoon"], "Post-monsoon")
    rows = []
    for k, (d, base) in enumerate(DISTRICTS.items()):
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.85*noise[t-1] + rng.normal(0, 1.1)
        price = base + seas + 0.35*(feed-100) + 1.5*(dates.dayofweek.values == 6) + noise
        age = 28 + (np.arange(n) + 4*k) % 17
        rows.append(pd.DataFrame({"date": dates, "district": d, "live_bird_price": price.round(2),
            "season": season, "flock_age_days": age, "feed_cost_index": (feed + rng.normal(0, .3, n)).round(2)}))
    df = pd.concat(rows, ignore_index=True)
    # inject data-quality problems
    i = rng.choice(len(df), int(.01*len(df)), replace=False); df.loc[i, "live_bird_price"] = np.nan
    i = rng.choice(len(df), int(.005*len(df)), replace=False); df.loc[i, "feed_cost_index"] = np.nan
    i = rng.choice(len(df), 8, replace=False); df.loc[i, "live_bird_price"] *= rng.choice([1.7, .45], 8)
    df = pd.concat([df.drop(rng.choice(len(df), 15, replace=False)), df.sample(20, random_state=1)])
    Path(path).parent.mkdir(exist_ok=True)
    df.sample(frac=1, random_state=3).to_csv(path, index=False)
    return path

if __name__ == "__main__":
    print("Wrote", generate())
