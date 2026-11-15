MIN_AGE, TARGET_AGE, MAX_AGE = 32, 35, 45   # illustrative thresholds - tune with farm/business guidance
THRESH = 0.02                                # 2% move = "meaningful"

def recommend(flock_age, last_price, fc):
    """Indicative rule-based output -> (action, date or None, reason). fc: DataFrame[date, forecast]."""
    chg_end = fc["forecast"].iloc[-1] / last_price - 1
    allowed = fc.head(max(MAX_AGE - flock_age, 0))
    if flock_age < MIN_AGE:
        return "WAIT", None, f"Flock is {flock_age} days old, below the {MIN_AGE}-day minimum. Re-check closer to harvest age."
    if flock_age >= MAX_AGE or allowed.empty:
        return "HARVEST NOW", fc["date"].iloc[0], f"Flock age ({flock_age} d) is at/over {MAX_AGE} days; holding cost and mortality risk usually outweigh price gains."
    ab = allowed.loc[allowed["forecast"].idxmax()]
    chg_ab = ab["forecast"] / last_price - 1
    if chg_end <= -THRESH:
        return "HARVEST EARLY", fc["date"].iloc[0], f"Prices are forecast to fall ~{abs(chg_end):.1%} over 7 days; consider selling at the first practical date."
    if chg_ab >= THRESH:
        return "HOLD FOR BETTER PRICE", ab["date"], f"Forecast peaks ~{chg_ab:.1%} above today around {ab['date']:%d %b}, and the flock stays within the {MAX_AGE}-day limit."
    if flock_age >= TARGET_AGE:
        return "HARVEST IN WINDOW", fc["date"].iloc[0], "Prices look broadly flat and the flock has reached target age; harvest as logistics allow."
    return "WAIT", None, f"Prices flat and flock is {flock_age} d (target {TARGET_AGE}+). Continue growing."
