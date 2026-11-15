import numpy as np, pandas as pd

FEATURE_COLS = ["lag0", "lag1", "lag2", "lag6", "roll7", "roll14", "std7", "mom6",
                "feed", "feed_chg7", "month_sin", "month_cos", "dow", "district_code"]

def add_features(df, codes):
    """Features on row t use only data up to date t (no look-ahead).
    Calendar features describe the TARGET date (t+1), which is known in advance.
    lag6 = price 6 days before t = same weekday as the target date."""
    df = df.sort_values(["district", "date"]).copy()
    g = df.groupby("district")["live_bird_price"]
    for k in (0, 1, 2, 6):
        df[f"lag{k}"] = g.shift(k)
    df["roll7"] = g.transform(lambda s: s.rolling(7).mean())
    df["roll14"] = g.transform(lambda s: s.rolling(14).mean())
    df["std7"] = g.transform(lambda s: s.rolling(7).std())
    df["mom6"] = df["lag0"] - df["lag6"]
    df["feed"] = df["feed_cost_index"]
    df["feed_chg7"] = df.groupby("district")["feed_cost_index"].diff(7)
    tgt = df["date"] + pd.Timedelta(days=1)
    df["month_sin"] = np.sin(2*np.pi*tgt.dt.month/12); df["month_cos"] = np.cos(2*np.pi*tgt.dt.month/12)
    df["dow"] = tgt.dt.dayofweek
    df["district_code"] = df["district"].map(codes)
    return df

def training_frame(df, codes):
    f = add_features(df, codes)
    f["target"] = f.groupby("district")["live_bird_price"].shift(-1)   # next-day price
    return f.dropna(subset=FEATURE_COLS + ["target"])
