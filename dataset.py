"""Meta-label dataset — ported from trendline_break_dataset.py (MIT).

Long breakouts above the projected resistance line, exited by a triple barrier
(take profit / stop loss / hold period) measured in log-ATR units. Five features,
one binary meta-label: did the primary trade make money?

ATR and ADX are implemented in-house with Wilder smoothing (SMA-seeded RMA), the
same convention pandas_ta uses by default, so the project needs no TA dependency.
"""
import numpy as np
import pandas as pd

FEATURES = ["resist_s", "tl_err", "max_dist", "vol", "adx"]


def rma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    n = int(max(1, n))
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def true_range(high, low, close):
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    prev = np.concatenate(([c[0]], c[:-1]))
    return np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))


def atr(high, low, close, n):
    return rma(true_range(high, low, close), n)


def adx(high, low, close, n):
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100.0 * rma(plus_dm, n) / tr
        mdi = 100.0 * rma(minus_dm, n) / tr
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(np.nan_to_num(dx, nan=0.0), n)


def build_trades(ohlc: pd.DataFrame, lines: dict, lookback: int,
                 hold_period: int = 12, tp_mult: float = 3.0, sl_mult: float = 3.0,
                 atr_lookback: int = 168):
    """Trade list + feature matrix + meta-labels. All prices in log space."""
    atr_lookback = int(max(atr_lookback, lookback))
    hold_period = int(max(1, hold_period))

    close = np.log(ohlc["close"].to_numpy(dtype=float))
    atr_arr = atr(np.log(ohlc["high"]), np.log(ohlc["low"]), np.log(ohlc["close"]), atr_lookback)
    adx_arr = adx(ohlc["high"], ohlc["low"], ohlc["close"], lookback)

    if "volume" in ohlc.columns:
        v = ohlc["volume"].to_numpy(dtype=float)
        med = pd.Series(v).rolling(atr_lookback).median().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            vol_arr = np.where(med > 0, v / med, 1.0)
        has_volume = True
    else:
        vol_arr = np.ones(len(close))
        has_volume = False

    r_tl = lines["resist"]
    r_slope, r_int = lines["r_slope"], lines["r_int"]

    rows = []
    in_trade = False
    tp = sl = None
    hp_i = None
    cur = None

    start = max(atr_lookback, lookback)
    for i in range(start, len(close)):
        if in_trade:
            reason = None
            if close[i] >= tp:
                reason = "take profit"
            elif close[i] <= sl:
                reason = "stop loss"
            elif i >= hp_i:
                reason = "hold period"
            if reason:
                cur["exit_i"] = i
                cur["exit_p"] = close[i]
                cur["exit_reason"] = reason
                rows.append(cur)
                in_trade = False
                cur = None

        if (not in_trade) and np.isfinite(r_tl[i]) and close[i] > r_tl[i]:
            a = atr_arr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            slope, intercept = r_slope[i], r_int[i]
            window = close[i - lookback:i]
            line_vals = intercept + np.arange(lookback) * slope
            diff = line_vals - window
            tp = close[i] + a * tp_mult
            sl = close[i] - a * sl_mult
            hp_i = i + hold_period
            cur = dict(
                entry_i=i, entry_p=close[i], atr=a, sl=sl, tp=tp, hp_i=hp_i,
                slope=slope, intercept=intercept,
                resist_s=slope / a,
                tl_err=float(diff.sum()) / lookback / a,
                max_dist=float(diff.max()) / a,
                vol=float(vol_arr[i]),
                adx=float(adx_arr[i]),
            )
            in_trade = True

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, pd.DataFrame(columns=FEATURES), pd.Series(dtype=int), has_volume

    trades["return"] = trades["exit_p"] - trades["entry_p"]
    trades["bars_held"] = trades["exit_i"] - trades["entry_i"]
    ok = np.isfinite(trades[FEATURES].to_numpy()).all(axis=1)
    trades = trades.loc[ok].reset_index(drop=True)

    data_x = trades[FEATURES].copy()
    data_y = (trades["return"] > 0).astype(int)
    return trades, data_x, data_y, has_volume


def baseline_signal(n_bars: int, trades: pd.DataFrame):
    """Every primary trade taken, no ML filter (the repo's 'dumb_sig')."""
    sig = np.zeros(n_bars)
    for t in trades.itertuples():
        sig[int(t.entry_i):int(t.exit_i)] = 1.0
    return sig
