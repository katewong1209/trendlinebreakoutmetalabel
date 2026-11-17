"""Walk-forward meta-labeling — ported from walkforward.py (MIT).

A RandomForestClassifier is re-fitted every `step_size` bars using only trades
that both opened and closed inside the trailing `train_size` window. At each new
breakout the model's P(profit) gates the entry; the position is then held to the
triple barrier regardless of later probabilities. Skipped breakouts still occupy
the slot, exactly as in the source.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def walkforward_model(close, trades, data_x, data_y, train_size, step_size,
                      threshold=0.5, n_estimators=300, max_depth=3, seed=69420,
                      min_train=12, progress=None, stop=None):
    n = len(close)
    signal = np.zeros(n)
    prob_signal = np.zeros(n)
    probs = np.full(len(trades), np.nan)

    train_size = int(max(20, min(train_size, max(20, n - 2))))
    step_size = int(max(1, min(step_size, max(1, n // 2))))

    next_train = train_size
    trade_i = 0
    in_trade = False
    tp = sl = None
    hp_i = None
    model = None
    imp_sum = np.zeros(data_x.shape[1])
    imp_n = 0
    log = []

    entry_idx = trades["entry_i"].to_numpy() if len(trades) else np.array([])

    for i in range(n):
        if i == next_train:
            start_i = i - train_size
            mask = (trades["entry_i"] > start_i) & (trades["exit_i"] < i)
            idx = trades.index[mask]
            y = data_y.loc[idx].to_numpy()
            if len(idx) >= min_train and len(np.unique(y)) == 2:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators), max_depth=max_depth,
                    random_state=seed, n_jobs=-1,
                )
                model.fit(data_x.loc[idx].to_numpy(), y)
                imp_sum += model.feature_importances_
                imp_n += 1
                log.append(dict(bar=i, cases=int(len(idx)),
                                win_rate=float(y.mean())))
            else:
                log.append(dict(bar=i, cases=int(len(idx)), win_rate=float("nan")))
            next_train += step_size

        if in_trade:
            if close[i] >= tp or close[i] <= sl or i >= hp_i:
                signal[i] = 0.0
                prob_signal[i] = 0.0
                in_trade = False
            else:
                signal[i] = signal[i - 1]
                prob_signal[i] = prob_signal[i - 1]

        if trade_i < len(trades) and i == entry_idx[trade_i]:
            if model is not None:
                x = data_x.iloc[trade_i].to_numpy().reshape(1, -1)
                prob = float(model.predict_proba(x)[0][1])
                probs[trade_i] = prob
                prob_signal[i] = prob
                if prob > threshold:
                    signal[i] = 1.0
                t = trades.iloc[trade_i]
                tp, sl, hp_i = t["tp"], t["sl"], t["hp_i"]
                in_trade = True
            trade_i += 1

        if progress is not None and i % 512 == 0:
            progress(i / max(1, n))
        if stop is not None and i % 512 == 0 and stop.is_set():
            break

    out = trades.copy()
    out["model_prob"] = probs
    importances = imp_sum / imp_n if imp_n else np.zeros(data_x.shape[1])
    return signal, prob_signal, out, importances, log, train_size


def profit_factor(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    if neg > 0:
        return float(pos / neg)
    return float("inf") if pos > 0 else float("nan")


def curve_metrics(close, signal, start=0):
    r = np.diff(np.log(np.asarray(close, float)), append=np.nan)
    rets = (signal * r)[start:]
    eq = np.nancumsum(rets)
    dd = eq - np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    sd = np.nanstd(rets)
    return dict(pf=profit_factor(rets), total=float(np.nansum(rets)),
                exposure=float(np.nanmean(signal[start:] > 0)) if len(signal[start:]) else 0.0,
                sharpe=float(np.nanmean(rets) / sd) if sd > 0 else float("nan"),
                mdd=float(np.expm1(dd.min())) if len(dd) else 0.0,
                equity=eq)


def trade_stats(trades):
    if trades.empty:
        return dict(n=0, win=float("nan"), avg=float("nan"), pf=float("nan"))
    r = trades["return"].to_numpy()
    return dict(n=int(len(r)), win=float((r > 0).mean()),
                avg=float(r.mean()), pf=profit_factor(r))


def threshold_sweep(trades, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.81, 0.05), 2)
    scored = trades[np.isfinite(trades["model_prob"])]
    rows = []
    for t in grid:
        sel = scored[scored["model_prob"] > t]
        s = trade_stats(sel)
        rows.append(dict(threshold=float(t), trades=s["n"], win_rate=s["win"],
                         avg=s["avg"], pf=s["pf"]))
    return pd.DataFrame(rows)


def confusion(trades, threshold=0.5):
    scored = trades[np.isfinite(trades["model_prob"])]
    if scored.empty:
        return None
    y = (scored["return"] > 0).to_numpy()
    p = (scored["model_prob"] > threshold).to_numpy()
    tp = int((p & y).sum()); fp = int((p & ~y).sum())
    fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec + rec) and prec + rec > 0 else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / max(1, tp + fp + fn + tn), n=int(len(scored)))
