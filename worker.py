"""Background job runner — streams progress into a plain dict, same pattern as
the MCPT terminal: the UI never blocks and can stop mid-run."""
import time
import traceback

import numpy as np
import pandas as pd

from trendlines import trendline_pass
from dataset import build_trades, baseline_signal, FEATURES
from meta_model import (walkforward_model, curve_metrics, trade_stats,
                        threshold_sweep, confusion)

TF_LABEL = {"1h": "Hourly", "1D": "Daily", "1M": "Monthly", "1Y": "Yearly"}
TF_UNIT = {"1h": "hour", "1D": "day", "1M": "month", "1Y": "year"}
TF_PER_YEAR = {"1h": 8760, "1D": 365, "1M": 12, "1Y": 1}
TIMEFRAMES = ("1h", "1D", "1M", "1Y")

TF_DEFAULTS = {
    "1h": dict(lookback=72, line_step=4, hold=12, atr=168, train=17520, step=8760),
    "1D": dict(lookback=30, line_step=1, hold=5, atr=60, train=365, step=90),
    "1M": dict(lookback=12, line_step=1, hold=3, atr=24, train=36, step=6),
    "1Y": dict(lookback=3, line_step=1, hold=1, atr=3, train=5, step=1),
}


def tf_defaults(tf):
    return TF_DEFAULTS.get(tf, TF_DEFAULTS["1D"])


def bars_per_year(tf):
    return TF_PER_YEAR.get(tf, 1)


def resample_ohlc(df, tf):
    if tf == "1h":
        return df
    rule = {"1D": "1D", "1M": "1ME", "1Y": "1YE"}[tf]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    try:
        return df.resample(rule).agg(agg).dropna()
    except ValueError:                      # pandas < 2.2 aliases
        rule = {"1M": "1M", "1Y": "1Y"}.get(tf, rule)
        return df.resample(rule).agg(agg).dropna()


def fit_windows(n_bars, train_bars, step_bars):
    """Auto-fit, no minimum bar count. Always leaves out-of-sample bars."""
    n = int(n_bars)
    if n < 12:
        raise ValueError(f"Only {n} bar(s) at this bar size — pick a finer bar size or a wider range.")
    step = int(max(1, min(int(step_bars), max(1, n // 4))))
    train = int(max(10, min(int(train_bars), n - max(2 * step, n // 4))))
    train = int(min(train, n - 2))
    return train, step


def run_job(job, ohlc, cfg):
    try:
        t0 = time.time()
        n = len(ohlc)
        close = np.log(ohlc["close"].to_numpy(dtype=float))

        job["phase"] = "Fitting trendlines…"
        lines = trendline_pass(
            close, cfg["lookback"], cfg["line_step"],
            progress=lambda f: job.update(progress=0.45 * f),
            stop=job["stop"],
        )
        if job["stop"].is_set():
            job["status"] = "stopped"; return
        job["lines"] = lines
        job["lookback_used"] = lines["lookback"]

        job["phase"] = "Building meta-label dataset…"
        job["progress"] = 0.5
        trades, data_x, data_y, has_vol = build_trades(
            ohlc, lines, lines["lookback"], cfg["hold"], cfg["tp"], cfg["sl"], cfg["atr"])
        job["has_volume"] = has_vol
        if trades.empty:
            raise ValueError("No breakouts were produced — shorten the lookback or pick a finer bar size.")

        train, step = fit_windows(n, cfg["train"], cfg["step"])
        job["train_bars"], job["step_bars"] = train, step

        job["phase"] = "Walk-forward training…"
        sig, prob_sig, trades, imp, log, train = walkforward_model(
            close, trades, data_x, data_y, train, step,
            threshold=cfg["threshold"], n_estimators=cfg["trees"],
            max_depth=cfg["depth"], seed=69420,
            progress=lambda f: job.update(progress=0.5 + 0.48 * f),
            stop=job["stop"],
        )
        if job["stop"].is_set():
            job["status"] = "stopped"; return

        base = baseline_signal(n, trades)
        bh = np.ones(n)
        start = train
        idx = ohlc.index[start:][::max(1, (n - start) // 1400)]
        ds = max(1, (n - start) // 1400)

        m_meta = curve_metrics(ohlc["close"].to_numpy(), sig, start)
        m_all = curve_metrics(ohlc["close"].to_numpy(), base, start)
        m_bh = curve_metrics(ohlc["close"].to_numpy(), bh, start)

        scored = trades[np.isfinite(trades["model_prob"])]
        taken = scored[scored["model_prob"] > cfg["threshold"]]

        job["trades"] = trades
        job["importances"] = dict(zip(FEATURES, [float(v) for v in imp]))
        job["train_log"] = log
        job["sweep"] = threshold_sweep(scored)
        job["confusion"] = confusion(scored, cfg["threshold"])
        job["curves"] = dict(x=idx, meta=m_meta["equity"][::ds],
                             all=m_all["equity"][::ds], bh=m_bh["equity"][::ds])
        job["signal"] = sig[start:][::ds]
        job["metrics"] = dict(
            meta=m_meta, all=m_all, bh=m_bh,
            meta_trades=trade_stats(taken), all_trades=trade_stats(scored),
            n_scored=int(len(scored)), n_taken=int(len(taken)),
            oos_start=start, bars=n,
        )
        job["window_desc"] = (f"train {train:,} · step {step:,} · OOS {n - start:,} bars · "
                              f"{len(log)} refits")
        job["progress"] = 1.0
        job["status"] = "done"
        job["elapsed"] = time.time() - t0
    except Exception:
        job["error"] = traceback.format_exc(limit=4)
        job["status"] = "error"
