import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .features import add_features, training_frame, FEATURE_COLS

H = 7

def fit(df, codes, cutoff=None):
    """Train on rows whose TARGET date is before `cutoff` (None = use everything)."""
    tr = training_frame(df, codes)
    if cutoff is not None:
        tr = tr[tr["date"] + pd.Timedelta(days=1) < cutoff]
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return m.fit(tr[FEATURE_COLS], tr["target"])

def forecast(model, hist, district, codes, horizon=H):
    """Recursive multi-step forecast: each prediction is fed back as the next lag.
    Feed index is held at its last observed value (not forecast)."""
    h = hist[hist["district"] == district].sort_values("date").tail(40)[
        ["date", "district", "live_bird_price", "feed_cost_index"]].copy()
    out = []
    for _ in range(horizon):
        x = add_features(h, codes).iloc[[-1]][FEATURE_COLS]
        p = float(model.predict(x)[0])
        nd = h["date"].iloc[-1] + pd.Timedelta(days=1)
        h = pd.concat([h, pd.DataFrame([{"date": nd, "district": district, "live_bird_price": p,
                                         "feed_cost_index": h["feed_cost_index"].iloc[-1]}])], ignore_index=True)
        out.append((nd, p))
    return pd.DataFrame(out, columns=["date", "forecast"])

def evaluate(df, codes, test_days=90, step=3):
    """Time-aware split: train strictly before the test window; rolling-origin 7-day forecasts inside it."""
    test_start = df["date"].max() - pd.Timedelta(days=test_days - 1)
    model = fit(df, codes, cutoff=test_start)
    rows = []
    origins = pd.date_range(test_start - pd.Timedelta(days=1), df["date"].max() - pd.Timedelta(days=H), freq=f"{step}D")
    for d in df["district"].unique():
        full = df[df["district"] == d].set_index("date")["live_bird_price"]
        for o in origins:
            fc = forecast(model, df[df["date"] <= o], d, codes)
            for i, r in fc.iterrows():
                h = i + 1; sn = o + pd.Timedelta(days=h - 7)
                rows.append(dict(district=d, origin=o, h=h, actual=full[r["date"]], rf=r["forecast"],
                                 naive=full[o], seasonal_naive=full[sn]))
    res = pd.DataFrame(rows)
    def metrics(g):
        out = {}
        for m in ["naive", "seasonal_naive", "rf"]:
            e = g[m] - g["actual"]
            out[m] = dict(MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e**2).mean())),
                          MAPE=float((e.abs()/g["actual"]).mean()*100))
        return out
    summary = {"test_start": str(test_start.date()), "test_end": str(df["date"].max().date()),
               "n_forecasts": len(res), "overall": metrics(res),
               "by_horizon": {int(h): metrics(g) for h, g in res.groupby("h")}}
    return summary, res
