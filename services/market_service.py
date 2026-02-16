from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from data import DataConnector


def load_market_snapshot(ticker: str) -> Tuple[float, List[str], Dict, float]:
    """
    Load core live market snapshot for a ticker.

    Returns:
        (spot, expirations, market_data, dividend_yield)
    """
    spot = DataConnector.get_spot_price(ticker)
    expirations = DataConnector.get_expirations(ticker)
    if not expirations:
        raise ValueError(f"No options data for {ticker}")
    market_data = DataConnector.get_market_data(ticker)
    div_yield = market_data.get("dividend_yield", 0.0) or 0.0
    return float(spot), expirations, market_data, float(div_yield)


def build_expiration_options(expirations: List[str], max_items: int = 20) -> List[Dict]:
    """
    Build UI-ready expiration options with calendar and business days.
    """
    exp_options: List[Dict] = []
    today_np = np.datetime64(datetime.now().date())
    for exp in expirations[:max_items]:
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            days = (exp_dt - datetime.now()).days
            if days > 0:
                exp_np = np.datetime64(exp_dt.date())
                biz_days = int(np.busday_count(today_np, exp_np))
                biz_days = max(1, biz_days)
                exp_options.append({
                    "date": exp, "days": days, "biz_days": biz_days,
                    "label": f"{exp} ({days}d)"
                })
        except Exception:
            continue
    return exp_options


def require_hist_vol_market_only(ticker: str, window_days: int) -> float:
    """
    Enforce strict market-only HV policy (no synthetic fallback).

    Args:
        ticker: Symbole Yahoo Finance
        window_days: Fenêtre en jours ouvrés pour le calcul de la vol historique
    """
    hist_vol = DataConnector.get_historical_volatility(ticker, window=window_days)
    if hist_vol is None or not np.isfinite(hist_vol) or hist_vol <= 0:
        raise ValueError("Historical volatility unavailable from Yahoo.")
    return float(hist_vol)
