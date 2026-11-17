"""Trendline automation

Method, unchanged:
  1. least-squares line through the window,
  2. pivot = point furthest above (resistance) / below (support) that line,
  3. rotate the line around the pivot by numerical gradient descent with step
     halving, minimising squared distance subject to the line never cutting
     through price.

Two documented deviations for interactive use: degenerate windows return the
least-squares line instead of raising, and `step` lets the fit be reused for a
few bars (projected forward) so hourly feeds stay responsive.
"""
import numpy as np


def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray):
    """Squared error of the pivot-anchored line, or -1 if it cuts through price."""
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y
    if support and diffs.max() > 1e-5:
        return -1.0
    if (not support) and diffs.min() < -1e-5:
        return -1.0
    return float((diffs ** 2.0).sum())


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray):
    """Gradient descent on the slope with halving steps. None if degenerate."""
    span = float(y.max() - y.min())
    if not np.isfinite(span) or span <= 0:
        return None
    slope_unit = span / len(y)
    min_step = 0.0001
    curr_step = 1.0
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    if best_err < 0.0:
        return None

    get_derivative = True
    derivative = 0.0
    guard = 0
    while curr_step > min_step and guard < 600:
        guard += 1
        if get_derivative:
            test_err = check_trend_line(support, pivot, best_slope + slope_unit * min_step, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                test_err = check_trend_line(support, pivot, best_slope - slope_unit * min_step, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                return None
            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err, best_slope = test_err, test_slope
            get_derivative = True

    return best_slope, -best_slope * pivot + y[pivot]


def fit_trendlines_single(data: np.ndarray):
    """(support_coefs, resistance_coefs) for one window of log closes."""
    y = np.asarray(data, dtype=float)
    x = np.arange(len(y))
    coefs = np.polyfit(x, y, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = int((y - line_points).argmax())
    lower_pivot = int((y - line_points).argmin())
    sup = optimize_slope(True, lower_pivot, coefs[0], y)
    res = optimize_slope(False, upper_pivot, coefs[0], y)
    fallback = (float(coefs[0]), float(coefs[1]))
    return (sup or fallback), (res or fallback)


def fit_trendlines_high_low(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    """Same fit, pivots taken from the wicks (the repo's OHLC variant)."""
    c = np.asarray(close, dtype=float)
    x = np.arange(len(c))
    coefs = np.polyfit(x, c, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = int((np.asarray(high) - line_points).argmax())
    lower_pivot = int((np.asarray(low) - line_points).argmin())
    sup = optimize_slope(True, lower_pivot, coefs[0], np.asarray(low, dtype=float))
    res = optimize_slope(False, upper_pivot, coefs[0], np.asarray(high, dtype=float))
    fallback = (float(coefs[0]), float(coefs[1]))
    return (sup or fallback), (res or fallback)


def trendline_pass(close: np.ndarray, lookback: int, step: int = 1,
                   progress=None, stop=None):
    """Rolling trendlines projected onto the current bar.

    The window never includes the current candle: at bar i the line is fitted on
    close[i-lookback:i] and evaluated at index `lookback`, exactly as the source.
    Returns per-bar line values, the carried coefficients and the raw signal.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    lookback = int(max(3, min(lookback, max(3, n - 1))))
    step = int(max(1, step))

    s_tl = np.full(n, np.nan)
    r_tl = np.full(n, np.nan)
    s_slope = np.full(n, np.nan)
    r_slope = np.full(n, np.nan)
    r_int = np.full(n, np.nan)
    sig = np.zeros(n)

    s_coefs = r_coefs = None
    origin = lookback
    for i in range(lookback, n):
        if s_coefs is None or (i - origin) % step == 0:
            s_coefs, r_coefs = fit_trendlines_single(c[i - lookback:i])
            origin = i
        k = lookback + (i - origin)
        s_val = s_coefs[1] + k * s_coefs[0]
        r_val = r_coefs[1] + k * r_coefs[0]
        s_tl[i], r_tl[i] = s_val, r_val
        s_slope[i], r_slope[i], r_int[i] = s_coefs[0], r_coefs[0], r_coefs[1]

        if c[i] > r_val:
            sig[i] = 1.0
        elif c[i] < s_val:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1]

        if progress is not None and (i % 256 == 0):
            progress((i - lookback) / max(1, n - lookback))
        if stop is not None and (i % 256 == 0) and stop.is_set():
            break

    return dict(support=s_tl, resist=r_tl, s_slope=s_slope, r_slope=r_slope,
                r_int=r_int, signal=sig, lookback=lookback, step=step)
