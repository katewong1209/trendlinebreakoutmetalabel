# Trendline Breakout · Meta-Label ML

Built by **Kate Wong**
A geometric primary signal filtered by a machine-learning meta-label to optimize trade entry, sizing, and execution. Built to solve the "low win-rate, high reward" problem of pure breakout strategies by gating entries on the probability of success.

## 📐 Methodology & Mathematical Formulation
### 1. Constrained Trendline Automation
Instead of naive pivot connecting, the engine fits a robust trendline via a two-step process:
1. **Initial OLS Fit**: $\hat{y} = \beta_0 + \beta_1 x$ over a rolling lookback window.
2. **Pivot Identification**: Find the extreme point furthest from the line: $t_{pivot} = \arg\max_t (y_t - \hat{y}_t)$ for resistance (or $\arg\min$ for support).
3. **Constrained Rotation**: The line is rotated around the pivot via numerical gradient descent with step-halving to minimize squared error, subject to the strict constraint that the line **never cuts through price**:
   $$ \text{Minimize} \sum_{t} (y_t - (\alpha + \beta x_t))^2 \quad \text{subject to} \quad y_t \le \alpha + \beta x_t \;\; \forall t \in \text{window} $$

### 2. Triple Barrier Method (TBM)
Labels are generated using TBM in log-ATR units, defining three exit conditions:
- **Upper Barrier**: Take Profit at $m \times \text{ATR}$
- **Lower Barrier**: Stop Loss at $n \times \text{ATR}$
- **Vertical Barrier**: Maximum holding period $H$ bars

### 3. Walk-Forward Meta-Labeling
A `RandomForestClassifier` is re-fitted every `step_size` bars using *only* trades that fully closed within the trailing `train_size` window. 
- **Features**: `resist_s` (slope), `tl_err` (fit error), `max_dist` (distance to pivot), `vol` (rolling volatility), `adx` (trend strength).
- **Gating**: Entry is executed only if $P(\text{Profit}) > \theta_{threshold}$.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
