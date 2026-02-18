"""
Logique métier pour le pricer vanilla (options européennes/américaines).

Extrait la logique pure des pages pour faciliter tests et maintenance.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core import BlackScholes
from core.solvers import IVsolver
from data import DataCleaner
from config.options import MAX_SPREAD_PCT


def calibrate_market_from_put_call_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot_market: float,
    T: float,
    rate_init: float,
    div_yield_init: float,
    div_yield_forecast: float = 0.0,
    max_spread_pct: Optional[float] = None,
    n_strikes_near_atm: int = 10,
) -> Tuple[float, float, float]:
    """
    Calibre r, q, S par régression sur la parité Put-Call: C - P = S*e^(-qT) - K*e^(-rT).

    Returns:
        (spot_implied, rate_implied, div_yield_implied)
    """
    if max_spread_pct is None:
        max_spread_pct = MAX_SPREAD_PCT
    q_init = div_yield_init if div_yield_init > 0.001 else max(div_yield_init, div_yield_forecast)
    try:
        calls_c = DataCleaner.clean_option_chain(calls.copy(), min_bid=0.01)
        puts_c = DataCleaner.clean_option_chain(puts.copy(), min_bid=0.01)
        calls_c = calls_c[["strike", "bid", "ask"]].rename(
            columns={"bid": "call_bid", "ask": "call_ask"}
        )
        puts_c = puts_c[["strike", "bid", "ask"]].rename(
            columns={"bid": "put_bid", "ask": "put_ask"}
        )
        merged = pd.merge(calls_c, puts_c, on="strike")
        merged["call_mid"] = (merged["call_bid"] + merged["call_ask"]) / 2
        merged["put_mid"] = (merged["put_bid"] + merged["put_ask"]) / 2
        merged = merged[(merged["call_mid"] > 0) & (merged["put_mid"] > 0)]
        merged["call_spread_pct"] = (merged["call_ask"] - merged["call_bid"]) / merged["call_mid"]
        merged["put_spread_pct"] = (merged["put_ask"] - merged["put_bid"]) / merged["put_mid"]
        merged["cp_market"] = merged["call_mid"] - merged["put_mid"]
        merged["dist_to_atm"] = (merged["strike"] - spot_market).abs()
        merged = merged[
            (merged["call_spread_pct"] <= max_spread_pct) &
            (merged["put_spread_pct"] <= max_spread_pct)
        ]
        merged = merged.nsmallest(n_strikes_near_atm, "dist_to_atm")
        if len(merged) < 5:
            return spot_market, rate_init, q_init

        K_arr = merged["strike"].values.astype(float)
        cp_arr = merged["cp_market"].values.astype(float)

        def objective_qs(x: np.ndarray) -> float:
            q, S = x[0], x[1]
            theory = S * np.exp(-q * T) - K_arr * np.exp(-rate_init * T)
            return float(np.sum((cp_arr - theory) ** 2))

        x0_qs = [q_init, spot_market]
        bounds_qs = [
            (0.0, 0.20),
            (spot_market * 0.5, spot_market * 2.0),
        ]
        res = minimize(objective_qs, x0_qs, method="L-BFGS-B", bounds=bounds_qs)
        if res.success:
            q_impl, S_impl = res.x
            if 0 <= q_impl <= 0.20 and spot_market * 0.5 <= S_impl <= spot_market * 2:
                return float(S_impl), float(rate_init), float(q_impl)

        def objective_full(x: np.ndarray) -> float:
            r, q, S = x[0], x[1], x[2]
            theory = S * np.exp(-q * T) - K_arr * np.exp(-r * T)
            ssr = np.sum((cp_arr - theory) ** 2)
            penalty = 50000.0 * (r - rate_init) ** 2
            return float(ssr + penalty)

        x0 = [rate_init, q_init, spot_market]
        bounds = [
            (max(0.001, rate_init - 0.02), min(0.20, rate_init + 0.02)),
            (0.0, 0.20),
            (spot_market * 0.5, spot_market * 2.0),
        ]
        res = minimize(objective_full, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return spot_market, rate_init, q_init
        r_impl, q_impl, S_impl = res.x
        if not (0.001 <= r_impl <= 0.20 and 0 <= q_impl <= 0.20 and spot_market * 0.5 <= S_impl <= spot_market * 2):
            return spot_market, rate_init, q_init
        return float(S_impl), float(r_impl), float(q_impl)
    except (KeyError, ValueError, TypeError, ImportError):
        return spot_market, rate_init, q_init


def compute_american_price(
    spot: float,
    K: float,
    T: float,
    rate: float,
    iv: float,
    opt_type: str,
    div_yield: float,
    dividends: list,
    n_steps: int = 1000,
) -> float:
    """Prix américain via BinomialTree avec dividendes discrets."""
    from models.trees import BinomialTree
    euro_price = BlackScholes.get_price(spot, K, T, rate, iv, opt_type, div_yield)
    tree = BinomialTree(spot, K, T, rate, iv, opt_type, n_steps=n_steps, dividends=dividends, q=div_yield)
    return tree.price_american_cv(euro_price)


def find_iv_from_market_price(
    market_price: float,
    spot: float,
    K: float,
    T: float,
    rate: float,
    opt_type: str,
    div_yield: float = 0.0,
) -> Optional[float]:
    """IV implicite depuis le prix de marché."""
    return IVsolver.find_implied_vol(market_price, spot, K, T, rate, opt_type, div_yield)


def compute_put_call_parity_theory(
    spot: float, K: float, T: float, rate: float, div_yield: float
) -> float:
    """C - P théorique (parité put-call)."""
    return spot * np.exp(-div_yield * T) - K * np.exp(-rate * T)
