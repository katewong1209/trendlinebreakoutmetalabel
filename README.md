# Trendline Breakout · Meta-Label ML

Built by **Kate Wong**
A geometric primary signal filtered by a machine-learning meta-label:

1. **Trendline automation** — least-squares line, extreme pivot, slope rotated by
   numerical gradient descent with step halving under a no-cut constraint.
2. **Primary breakout** — close beyond the resistance line projected from a window
   that excludes the current bar.
3. **Triple barrier** — take profit / stop loss / hold period in log-ATR units.
4. **Meta-label** — did that trade pay? Five features: `resist_s`, `tl_err`,
   `max_dist`, `vol`, `adx`.
5. **Walk-forward forest** — `RandomForestClassifier` re-fitted on trailing closed
   trades only; entry gated on `P(profit) > threshold`.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
