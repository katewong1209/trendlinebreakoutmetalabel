"""Seeded synthetic BTC/ETH hourly OHLCV feed used when no file is uploaded.

GARCH(1,1) volatility clustering, fat-tailed t(4) innovations, slow AR(1) trend
regimes, rare jumps, correlated cross-market shocks, realistic wicks, and volume
that responds to absolute returns. Deterministic for a given seed.
"""
import numpy as np
import pandas as pd


def _garch(z, omega, alpha, beta, uncond):
    n = len(z)
    s2 = np.empty(n)
    s2[0] = uncond
    for i in range(1, n):
        s2[i] = omega + alpha * (z[i - 1] ** 2) * s2[i - 1] + beta * s2[i - 1]
    return np.sqrt(s2)


def _ohlc(index, rets, sigma, s0, gaps, wick_u, wick_d, base_vol, rng):
    close = s0 * np.exp(np.cumsum(rets))
    prev = np.concatenate(([s0], close[:-1]))
    op = prev * np.exp(gaps)
    hi = np.maximum(op, close) * np.exp(wick_u * sigma)
    lo = np.minimum(op, close) * np.exp(-wick_d * sigma)
    shock = np.abs(rets) / max(np.std(rets), 1e-9)
    volume = base_vol * np.exp(0.55 * shock + rng.normal(0, 0.35, len(rets)))
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close,
                         "volume": np.maximum(volume, 1.0)}, index=index)


def make_synthetic_pair(seed: int = 7):
    index = pd.date_range("2016-01-01", "2021-01-01", freq="h", inclusive="left")
    index = index.astype("datetime64[s]")
    n = len(index)
    rng = np.random.default_rng(seed)

    z_b = rng.standard_t(4, n) / np.sqrt(2.0)
    rho = 0.62
    z_e = rho * z_b + np.sqrt(1 - rho ** 2) * (rng.standard_t(4, n) / np.sqrt(2.0))

    ub = 0.006 ** 2
    a, b = 0.08, 0.90
    sig_b = _garch(z_b, ub * (1 - a - b), a, b, ub)
    ue = (0.006 * 1.35) ** 2
    sig_e = _garch(z_e, ue * (1 - a - b), a, b, ue)

    trend = np.empty(n)
    t = 0.0
    shocks = rng.normal(0, 0.02, n)
    for i in range(n):
        t = 0.9995 * t + shocks[i]
        trend[i] = t

    jump_b = (rng.random(n) < 0.0015) * rng.normal(0, 0.035, n)
    jump_e = (rng.random(n) < 0.0018) * rng.normal(0, 0.045, n)
    rb = 0.000012 + 0.00045 * trend + sig_b * z_b + jump_b
    re = 0.000010 + 0.00060 * trend + sig_e * z_e + jump_e

    gb = rng.normal(0, 0.0004, n)
    ge = rng.normal(0, 0.0005, n)
    wub, wdb = np.abs(rng.normal(0, 0.55, n)), np.abs(rng.normal(0, 0.55, n))
    wue, wde = np.abs(rng.normal(0, 0.60, n)), np.abs(rng.normal(0, 0.60, n))

    btc = _ohlc(index, rb, sig_b, 430.0, gb, wub, wdb, 1200.0, rng)
    eth = _ohlc(index, re, sig_e, 8.5, ge, wue, wde, 9000.0, rng)
    return btc, eth
