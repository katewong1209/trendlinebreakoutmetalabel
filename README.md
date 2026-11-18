# Trendline Breakout with Walk-Forward Meta-Labeling

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A geometric primary signal filtered by a machine-learning meta-label to optimize trade entry, sizing, and execution. Built to solve the "low win-rate, high reward" problem of pure breakout strategies by gating entries on the probability of success.

##  Methodology & Mathematical Formulation

### 1. Constrained Trendline Automation
Instead of naive pivot connecting, the engine fits a robust trendline via a two-step process:
1. **Initial OLS Fit**: $\hat{y} = \beta_0 + \beta_1 x$ over a rolling lookback window.
2. **Pivot Identification**: Find the extreme point furthest from the line: $t_{pivot} = \arg\max_t (y_t - \hat{y}_t)$ for resistance (or $\arg\min$ for support).
3. **Constrained Rotation**: The line is rotated around the pivot via numerical gradient descent with step-halving to minimize squared error, subject to the strict constraint that the line **never cuts through price**:
   $$ \text \sum_{t} (y_t - (\alpha + \beta x_t))^2 \quad \text{subject to} \quad y_t \le \alpha + \beta x_t \;\; \forall t \in \text $$

### 2. Triple Barrier Method (TBM)
Labels are generated using TBM in log-ATR units, defining three exit conditions:
- **Upper Barrier**: Take Profit at $m \times \text{ATR}$
- **Lower Barrier**: Stop Loss at $n \times \text{ATR}$
- **Vertical Barrier**: Maximum holding period $H$ bars

### 3. Walk-Forward Meta-Labeling
A `RandomForestClassifier` is re-fitted every `step_size` bars using *only* trades that fully closed within the trailing `train_size` window. 
- **Features**: `resist_s` (slope), `tl_err` (fit error), `max_dist` (distance to pivot), `vol` (rolling volatility), `adx` (trend strength).
- **Gating**: Entry is executed only if $P(\text{Profit}) > \theta_{threshold}$.

##  Original Research: High-Frequency Multi-Asset Execution
*Standard meta-labeling often ignores execution reality. This repository extends the TBM to High-Frequency (1m/5m) multi-asset data with a **dynamic slippage-adjusted barrier model**.*

In illiquid equities, the bid-ask spread and market impact can turn a theoretical TP hit into a net loss. The extended engine dynamically widens the effective SL and tightens the effective TP based on real-time `spread_bps` and `relative_volume`, ensuring the meta-labeler learns to predict *net* profitability, not just mid-price movements.

See `hf_metalabel_slippage.py` for the implementation.

##  Visuals & Key Takeaways

### 1. Meta-Label Feature Importance
The app renders a horizontal bar chart of Random Forest feature importances.
- **Takeaway**: `tl_err` (trendline fit error) and `adx` (trend strength) dominate feature importance. Breakouts occurring when the trendline fit is tight (low error) and ADX > 25 have a significantly higher win rate than high-error, low-ADX setups.

### 2. Threshold Sweep & Equity Curve
- **Takeaway**: The primary breakout strategy alone may yield a ~51% win rate. By gating entries at $P(\text{Profit}) > 0.65$, the meta-labeler filters out low-quality trades, boosting the win rate to ~63% and reducing Maximum Drawdown, albeit with a reduction in trade frequency.

## Quickstart
```bash
pip install -r requirements.txt
pip install plotly
streamlit run app.py
