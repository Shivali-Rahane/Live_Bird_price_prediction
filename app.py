import json, joblib, pandas as pd, plotly.graph_objects as go, streamlit as st
from src.model import forecast
from src.harvest import recommend

st.set_page_config(page_title="Live Bird Price Outlook", page_icon="🐔", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#0f5132,#2e8b57 60%,#6cc08b);color:#fff;padding:26px 30px;border-radius:18px;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:4px 0 0;opacity:.9}
.kpi{background:#fff;border:1px solid #e6ece8;border-radius:16px;padding:16px 20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.kpi .l{color:#6b7a72;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:1.8rem;font-weight:700;color:#12372a}.kpi .s{font-size:.9rem}
.up{color:#1a8f4c}.down{color:#c0392b}.flat{color:#6b7a72}
.day{background:#f5faf7;border-radius:12px;padding:10px 6px;text-align:center;border:1px solid #e2ede6}
.day b{display:block;font-size:1.05rem;color:#12372a}.day small{color:#6b7a72}
.rec{border-radius:16px;padding:18px 22px;border-left:8px solid var(--c);background:var(--b)}
.rec h3{margin:0 0 4px;color:var(--c)}
.foot{color:#8a968f;font-size:.8rem;margin-top:24px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    return pd.read_csv("data/clean_prices.csv", parse_dates=["date"]), json.load(open("models/metrics.json"))
@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.joblib")

df, metrics = load(); art = load_model(); model, codes = art["model"], art["codes"]
rmse_h = pd.Series([metrics["by_horizon"][str(h)]["rf"]["RMSE"] for h in range(1, 8)])

@st.cache_data
def get_fc(district):
    fc = forecast(model, df, district, codes)
    fc["low"] = fc["forecast"] - 1.28 * rmse_h; fc["high"] = fc["forecast"] + 1.28 * rmse_h
    return fc

def direction(c): return ("Rising", "up", "▲") if c > .02 else ("Falling", "down", "▼") if c < -.02 else ("Stable", "flat", "●")

with st.sidebar:
    st.header("⚙️ Your farm")
    district = st.selectbox("District", sorted(df["district"].unique()))
    age = st.number_input("Current flock age (days)", 20, 60, 36)
    history_days = st.slider("History shown (days)", 14, 120, 45)
    st.divider(); st.caption("Model: Random Forest, 7-day recursive forecast.")

d = df[df["district"] == district].sort_values("date")
last_date, last_price = d["date"].iloc[-1], d["live_bird_price"].iloc[-1]
fc = get_fc(district)
chg = fc["forecast"].iloc[-1] / last_price - 1
label, cls, arrow = direction(chg)

st.markdown(f'<div class="hero"><h1>🐔 Live Bird Price Outlook</h1><p>{district} · 7-day forecast · data up to {last_date:%d %b %Y}</p></div>', unsafe_allow_html=True)

k = st.columns(4)
cards = [("Latest price", f"₹{last_price:.1f}", f"per kg · {last_date:%d %b}", "flat"),
         ("Day-7 forecast", f"₹{fc['forecast'].iloc[-1]:.1f}", f"{arrow} {chg:+.1%} vs today", cls),
         ("Expected direction", label, f"range ₹{fc['low'].min():.0f}–₹{fc['high'].max():.0f}", cls),
         ("Flock age", f"{int(age)} d", f"{max(45-int(age),0)} days to 45-day limit", "flat")]
for col, (l, v, s, c) in zip(k, cards):
    col.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s {c}">{s}</div></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🗺️ All districts", "📊 Model details"])

with tab1:
    h = d.tail(history_days)
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["live_bird_price"], name="Actual", line=dict(color="#1b6ca8", width=3))
    fig.add_scatter(x=fc["date"], y=fc["high"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["date"], y=fc["low"], fill="tonexty", fillcolor="rgba(46,139,87,.18)", line=dict(width=0), name="Approx. 80% range")
    fx = pd.concat([h.tail(1)[["date"]].assign(forecast=last_price), fc[["date", "forecast"]]])
    fig.add_scatter(x=fx["date"], y=fx["forecast"], name="Forecast", mode="lines+markers", line=dict(color="#2e8b57", width=3, dash="dash"))
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper", line=dict(color="#999", dash="dot"))
    fig.update_layout(height=400, margin=dict(t=10, b=10), yaxis_title="₹ per kg (live)", hovermode="x unified",
                      template="plotly_white", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(7); prev = last_price
    for col, (_, r) in zip(cols, fc.iterrows()):
        dd = r["forecast"] - prev; prev = r["forecast"]
        col.markdown(f'<div class="day"><small>{r["date"]:%a %d}</small><b>₹{r["forecast"]:.1f}</b>'
                     f'<small class="{"up" if dd > 0 else "down"}">{"▲" if dd > 0 else "▼"} {abs(dd):.1f}</small></div>', unsafe_allow_html=True)

    st.write("")
    action, date, why = recommend(int(age), last_price, fc)
    pal = {"WAIT": ("#6b7a72", "#f1f4f2"), "HOLD FOR BETTER PRICE": ("#1a8f4c", "#eaf7ef"), "HARVEST IN WINDOW": ("#1b6ca8", "#eaf3fa"),
           "HARVEST EARLY": ("#c0392b", "#fdeeec"), "HARVEST NOW": ("#c0392b", "#fdeeec")}
    c, b = pal.get(action, ("#6b7a72", "#f1f4f2"))
    when = f" · around {date:%a %d %b}" if date is not None else ""
    st.markdown(f'<div class="rec" style="--c:{c};--b:{b}"><h3>🚜 {action}{when}</h3><div>{why}</div></div>', unsafe_allow_html=True)
   

with tab2:
    rows = []
    for dist in sorted(df["district"].unique()):
        f = get_fc(dist); lp = df[df["district"] == dist].sort_values("date")["live_bird_price"].iloc[-1]
        rows.append({"District": dist, "Latest ₹/kg": round(lp, 1), "Day-7 ₹/kg": round(f["forecast"].iloc[-1], 1),
                     "Change %": round((f["forecast"].iloc[-1] / lp - 1) * 100, 1), "Direction": direction(f["forecast"].iloc[-1] / lp - 1)[0]})
    t = pd.DataFrame(rows)
    bar = go.Figure(go.Bar(x=t["District"], y=t["Change %"], marker_color=["#1a8f4c" if v > 0 else "#c0392b" for v in t["Change %"]]))
    bar.update_layout(height=320, template="plotly_white", yaxis_title="Expected 7-day change (%)", margin=dict(t=10))
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(t, hide_index=True, use_container_width=True)

with tab3:
    o = metrics["overall"]
    m = st.columns(3)
    for col, (n, key) in zip(m, [("Random Forest", "rf"), ("Naive baseline", "naive"), ("Same weekday last week", "seasonal_naive")]):
        col.markdown(f'<div class="kpi"><div class="l">{n}</div><div class="v">₹{o[key]["MAE"]:.2f}</div><div class="s flat">MAE · MAPE {o[key]["MAPE"]:.1f}%</div></div>', unsafe_allow_html=True)
    st.write(f"Backtest {metrics['test_start']} → {metrics['test_end']}, all districts, 1–7 day horizons. MAE is the average miss in ₹/kg. "
             "The band on the chart is approximate (±1.28 × backtest RMSE per horizon).")
    st.dataframe(fc.assign(date=fc["date"].dt.strftime("%a %d %b")).round(2).rename(columns={"forecast": "forecast ₹/kg"}), hide_index=True)

st.markdown('<div class="foot">Prototype built on synthetic data for demonstration; not real market prices.</div>', unsafe_allow_html=True)
