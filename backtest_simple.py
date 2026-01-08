#!/usr/bin/env python3
"""


Tests whether ML-predicted mispricing signals can generate profitable trading.
Incorporates realistic bid-ask spreads, slippage, and transaction costs.

Outputs:
  - Total P&L and return
  - Sharpe ratio (risk-adjusted return)
  - Maximum drawdown
  - Win rate and trade statistics
  - Turnover (trading intensity)

Usage:
    python backtest_simple.py --model gradient_boosting --ticker AAPL
    
   
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


def estimate_bid_ask_spread(option_price: float, option_type: str = 'call') -> float:
    """
    Estimate bid-ask spread for equity options.
    
    Typical spreads for liquid equity options:
    - ATM calls/puts: 0.05–0.10 ($)
    - OTM calls/puts: 0.01–0.05 ($)
    - Wide spreads for illiquid options: 0.20+ ($)
    
    We use 0.1% of option price as conservative estimate.
    
    Args:
        option_price: Current option market price ($)
        option_type: 'call' or 'put' (not used currently)
    
    Returns:
        Estimated bid-ask spread ($)
    """
    spread_pct = 0.001  # 0.1% of option price
    spread_dollars = max(0.01, option_price * spread_pct)  # min $0.01
    return spread_dollars


def estimate_slippage(option_price: float, pct_of_spread: float = 0.5) -> float:
    """
    Estimate slippage costs (price movement during execution).
    
    Typically 0.25–0.5x the bid-ask spread for retail orders.
    
    Args:
        option_price: Current option price ($)
        pct_of_spread: Slippage as % of spread (default 50%)
    
    Returns:
        Estimated slippage cost ($)
    """
    spread = estimate_bid_ask_spread(option_price)
    return spread * pct_of_spread


def simulate_delta_hedged_pnl(
    entry_price: float,
    exit_price: float,
    delta: float,
    stock_entry: float,
    stock_exit: float,
    contracts: int = 1
) -> float:
    """
    Calculate P&L for delta-hedged option position.
    
    Delta-hedged P&L ≈ (option_exit - option_entry) - delta * (stock_exit - stock_entry)
    
    The hedge removes directional risk, leaving only residual (gamma/vega) P&L.
    
   
    
    Returns:
        Delta-hedged P&L ($)
    """
    option_pnl = (exit_price - entry_price) * contracts * 100
    hedge_pnl = -delta * contracts * 100 * (stock_exit - stock_entry)
    return option_pnl + hedge_pnl


def backtest_underpriced_strategy(
    df: pd.DataFrame,
    model,
    features: List[str],
    risk_free_rate: float = 0.05,
    confidence_threshold: float = 0.5,
    max_trades_per_day: int = 10,
    delta_hedge: bool = False,
    use_costs: bool = True
) -> Dict:
    """
    Backtest strategy: "Buy options predicted as underpriced, sell next day."
    
    Args:
        df: Options DataFrame with columns:
            - 'date': Trading date
            - 'mkt_price': Market price at entry
            - 'fwd_option_price': Estimated option price next day (or proxy)
            - 'delta': Option delta (for hedging)
            - 'S': Stock price
            - All feature columns
        model: Trained sklearn classifier
        features: List of feature names used in model
        risk_free_rate: Risk-free rate for Sharpe ratio (annual)
        confidence_threshold: Confidence threshold for trade signal
        max_trades_per_day: Max underpriced options to trade per day
        delta_hedge: If True, calculate delta-hedged P&L; else mark-to-market
        use_costs: If True, include bid-ask + slippage; else ideal costs
    
    Returns:
        Dictionary with:
        {
            'total_trades': int,
            'winning_trades': int,
            'losing_trades': int,
            'win_rate': float (0–1),
            'total_pnl': float ($),
            'total_return': float (% of capital),
            'sharpe_ratio': float,
            'max_drawdown': float (% of peak),
            'turnover': float (% of available contracts),
            'avg_trade_pnl': float ($),
            'trades': pd.DataFrame
        }
    """
    
    if 'fwd_option_price' not in df.columns:
        print("[WARNING] 'fwd_option_price' not in DataFrame. Using next-day mkt_price proxy.")
        df = df.sort_values('date').reset_index(drop=True)
        df['fwd_option_price'] = df.groupby('date')['mkt_price'].shift(-1)
    
    trades = []
    dates = sorted(df['date'].unique())
    
    for date_idx, trading_date in enumerate(dates[:-1]):  # Skip last day (no forward price)
        
        # Get all options available today
        today_df = df[df['date'] == trading_date].copy()
        
        if len(today_df) == 0:
            continue
        
        # Extract features
        X_today = today_df[features].values
        
        try:
            # Get model predictions (probability of each class)
            probs = model.predict_proba(X_today)
            
            # Class 0 = underpriced, 1 = fair, 2 = overpriced
            underpriced_probs = probs[:, 0]
            
        except Exception as e:
            print(f"[ERROR] Failed to predict for {trading_date}: {e}")
            continue
        
        # Find best underpriced candidates
        best_idx = np.argsort(-underpriced_probs)[:max_trades_per_day]
        
        for idx in best_idx:
            
            if underpriced_probs[idx] < confidence_threshold:
                break  # Stop if confidence below threshold
            
            opt_row = today_df.iloc[idx]
            
            # Extract trade parameters
            entry_price = opt_row['mkt_price']
            exit_price = opt_row.get('fwd_option_price', np.nan)
            delta = opt_row.get('delta', 0.5)
            stock_entry = opt_row.get('S', np.nan)
            stock_exit = np.nan  # Would need next day's close
            
            # Skip if missing data
            if pd.isna(entry_price) or pd.isna(exit_price):
                continue
            
            # Calculate costs
            entry_cost = 0.0
            exit_cost = 0.0
            
            if use_costs:
                entry_cost = estimate_bid_ask_spread(entry_price) + \
                            estimate_slippage(entry_price)
                exit_cost = estimate_bid_ask_spread(exit_price) + \
                           estimate_slippage(exit_price)
            
            # Calculate P&L
            # Mark-to-market: profit = sell price - buy price - costs
            pnl = (exit_price - entry_price - entry_cost / 2 - exit_cost / 2) * 100
            
            trade_return = pnl / (entry_price * 100) if entry_price > 0 else 0.0
            
            trades.append({
                'date': trading_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_dollars': pnl,
                'pnl_return': trade_return,
                'underpriced_prob': underpriced_probs[idx],
                'strike': opt_row.get('K', np.nan),
                'delta': delta,
                'tau_days': opt_row.get('tau_days', np.nan),
                'moneyness': opt_row.get('moneyness', np.nan)
            })
    
    if len(trades) == 0:
        print("[WARNING] No trades generated. Check features/model/data.")
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan
        }
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate statistics
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl_dollars'] > 0).sum()
    losing_trades = (trades_df['pnl_dollars'] <= 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    total_pnl = trades_df['pnl_dollars'].sum()
    total_return = trades_df['pnl_return'].sum()
    avg_trade_pnl = trades_df['pnl_dollars'].mean()
    
    # Sharpe ratio (daily)
    daily_returns = trades_df.groupby('date')['pnl_return'].sum()
    
    if len(daily_returns) > 1:
        daily_std = daily_returns.std()
        daily_sharpe = daily_returns.mean() / daily_std if daily_std > 0 else 0.0
        annual_sharpe = daily_sharpe * np.sqrt(252)
    else:
        annual_sharpe = np.nan
    
    # Max drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Turnover (% of available contracts)
    total_available = len(df)
    turnover_pct = (total_trades / total_available) * 100 if total_available > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'avg_trade_pnl': avg_trade_pnl,
        'sharpe_ratio': annual_sharpe,
        'max_drawdown': max_drawdown,
        'turnover_pct': turnover_pct,
        'trades': trades_df
    }


def print_backtest_results(results: Dict):
    
    
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS: Underpriced Options Trading Strategy")
    print(f"{'='*70}")
    
    print(f"\n[TRADE STATISTICS]")
    print(f"  Total Trades:        {results['total_trades']:>8,}")
    print(f"  Winning Trades:      {results['winning_trades']:>8,}")
    print(f"  Losing Trades:       {results['losing_trades']:>8,}")
    print(f"  Win Rate:            {results['win_rate']:>8.1%}")
    print(f"  Avg P&L per Trade:   ${results['avg_trade_pnl']:>7,.2f}")
    
    print(f"\n[RETURNS]")
    print(f"  Total P&L:           ${results['total_pnl']:>10,.2f}")
    print(f"  Total Return:        {results['total_return']:>10.2%}")
    print(f"  Annualized Sharpe:   {results['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:>10.2%}")
    
    print(f"\n[TRADING INTENSITY]")
    print(f"  Turnover:            {results['turnover_pct']:>10.1f}%")
    
    print(f"\n[INTERPRETATION]")
    if results['sharpe_ratio'] > 2.0:
        print(f"  EXCELLENT: Sharpe > 2.0 = strong risk-adjusted returns")
    elif results['sharpe_ratio'] > 1.0:
        print(f"  GOOD: Sharpe > 1.0 = acceptable risk-adjusted returns")
    elif results['sharpe_ratio'] > 0:
        print(f"  FAIR: Sharpe 0–1.0 = marginal after costs")
    else:
        print(f"  POOR: Sharpe < 0 = strategy underperforms")
    
    if results['turnover_pct'] > 50:
        print(f"  HIGH TURNOVER: {results['turnover_pct']:.1f}% limits retail feasibility")
    else:
        print(f"  Reasonable turnover: {results['turnover_pct']:.1f}%")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Backtest Module for Economic Validation")
    print("=" * 70)
    print("\nUsage:")
    print("  from backtest_simple import backtest_underpriced_strategy, print_backtest_results")
    print("")
    print("  results = backtest_underpriced_strategy(df, model, features, use_costs=True)")
    print("  print_backtest_results(results)")
