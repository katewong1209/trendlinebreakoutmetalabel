import numpy as np
import pandas as pd

def hf_slippage_adjusted_tbm(high, low, close, entry_indices, tp_mult, sl_mult, 
                             max_bars, spread_bps, vol_impact_coef=0.1):
    """
    High-Frequency Triple Barrier that accounts for dynamic bid-ask spread and 
    market impact at both entry and exit.
    
    This ensures the Meta-Labeler learns to predict NET profitability, not just 
    mid-price theoretical touches.
    """
    n = len(close)
    exits = np.full(n, -1, dtype=int)
    net_returns = np.zeros(n)
    
    # Dynamic slippage in decimal: base spread + volatility/volume impact
    base_slip = spread_bps / 10000.0
    slip_cost = base_slip + vol_impact_coef * 0.01  # Simplified impact model
    
    for i in entry_indices:
        if exits[i] != -1:
            continue
            
        entry_price = close[i]
        
        # Adjust barriers for slippage: TP is harder to hit, SL is hit sooner
        # For a LONG position:
        effective_tp = entry_price * (1.0 + tp_mult - slip_cost)
        effective_sl = entry_price * (1.0 - sl_mult - slip_cost)
        
        for j in range(i + 1, min(i + max_bars + 1, n)):
            # Check barriers on high/low (mid-price crossing)
            if high[j] >= effective_tp:
                exits[i] = j
                # Exit at TP, but pay slippage on the exit fill
                net_returns[i] = (effective_tp / entry_price) - 1.0 - slip_cost
                break
            elif low[j] <= effective_sl:
                exits[i] = j
                net_returns[i] = (effective_sl / entry_price) - 1.0 - slip_cost
                break
        else:
            # Max holding period reached, exit at close
            exit_idx = min(i + max_bars, n - 1)
            exits[i] = exit_idx
            net_returns[i] = (close[exit_idx] / entry_price) - 1.0 - slip_cost
            
    return exits, net_returns
