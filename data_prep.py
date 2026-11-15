import pandas as pd, numpy as np

def clean(raw_path):
    """Return (clean_df, report). Daily, one row per district-date, short gaps filled."""
    df = pd.read_csv(raw_path)
    rep = {"rows_raw": len(df)}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["live_bird_price", "flock_age_days", "feed_cost_index"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rep["bad_dates_dropped"] = int(df["date"].isna().sum()); df = df.dropna(subset=["date"])
    rep["missing_before"] = df[["live_bird_price", "feed_cost_index"]].isna().sum().to_dict()
    n0 = len(df); df = df.drop_duplicates(["district", "date"]); rep["duplicates_dropped"] = n0 - len(df)
    out = []
    for d, g in df.sort_values("date").groupby("district"):
        g = g.set_index("date").reindex(pd.date_range(g["date"].min(), g["date"].max(), freq="D"))
        rep.setdefault("gap_days_filled", {})[d] = int(g["district"].isna().sum())
        g["district"] = d
        # outliers: > 4 robust sigmas from centred 15-day rolling median -> NaN, then re-filled below
        med = g["live_bird_price"].rolling(15, center=True, min_periods=5).median()
        mad = (g["live_bird_price"] - med).abs().rolling(15, center=True, min_periods=5).median()
        bad = (g["live_bird_price"] - med).abs() > 4 * 1.4826 * mad.clip(lower=1.0)
        rep.setdefault("outliers_removed", {})[d] = int(bad.sum())
        g.loc[bad, "live_bird_price"] = np.nan
        for c in ["live_bird_price", "feed_cost_index"]:          # short gaps (<=3 d) only
            g[c] = g[c].interpolate(limit=3, limit_direction="both")
        g["flock_age_days"] = g["flock_age_days"].ffill()
        g["season"] = g["season"].ffill()
        out.append(g.rename_axis("date").reset_index())
    df = pd.concat(out, ignore_index=True)
    rep["missing_after"] = df[["live_bird_price", "feed_cost_index"]].isna().sum().to_dict()
    df = df.dropna(subset=["live_bird_price", "feed_cost_index"])
    rep["rows_clean"] = len(df)
    return df, rep
