"""Implied volatility solver (Newton-Raphson + Brent)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from core.black_scholes import BlackScholes


# Raisons explicites d'échec du solveur IV
IV_FAIL_INVALID_INPUT = "invalid_input"
IV_FAIL_OUT_OF_BOUNDS = "out_of_bounds"
IV_FAIL_BRENT_NO_ROOT = "brent_no_root"
IV_FAIL_BRENT_ERROR = "brent_error"
IV_OK = "ok"


class IVsolver:
    """
    Implied Volatility Solver.
    
    Newton-Raphson en priorité, Brent en secours si NR échoue.
    """

    @staticmethod
    def find_implied_vol_with_reason(
        market_price: float, S: float, K: float, T: float, r: float,
        option_type: str = "call", q: float = 0.0
    ) -> Tuple[Optional[float], str]:
        """
        Trouve la volatilite implicite avec une raison explicite en cas d'echec.

        Returns:
            (iv, reason): iv = volatilite implicite ou None; reason = IV_OK ou code d'echec
        """
        if not np.isfinite(market_price) or market_price <= 0 or T <= 0:
            return None, IV_FAIL_INVALID_INPUT
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(r)):
            return None, IV_FAIL_INVALID_INPUT
        if S <= 0 or K <= 0:
            return None, IV_FAIL_INVALID_INPUT

        if option_type.lower() == "call":
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            upper_bound = S * np.exp(-q * T)
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            upper_bound = K * np.exp(-r * T)

        if market_price < intrinsic - 0.01 or market_price > upper_bound + 0.01:
            return None, IV_FAIL_OUT_OF_BOUNDS

        sigma_init = IVsolver._brenner_subrahmanyam_guess(market_price, S, K, T, r, option_type, q)
        iv = IVsolver._newton_raphson(market_price, S, K, T, r, option_type, q, sigma_init=sigma_init)
        if iv is not None:
            return iv, IV_OK

        iv, brent_reason = IVsolver._brent_with_reason(market_price, S, K, T, r, option_type, q)
        return iv, brent_reason

    @staticmethod
    def find_implied_vol(
        market_price: float, S: float, K: float, T: float, r: float,
        option_type: str = "call", q: float = 0.0
    ) -> Optional[float]:
        """
        Trouve la volatilite implicite depuis un prix de marche.
        
        Args:
            market_price: Prix observe sur le marche
            S: Spot
            K: Strike
            T: Time to maturity (annees)
            r: Taux sans risque
            option_type: "call" ou "put"
            q: Dividend yield
        
        Returns:
            float: Implied volatility, ou None si echec
        """
        iv, _ = IVsolver.find_implied_vol_with_reason(
            market_price, S, K, T, r, option_type, q
        )
        return iv

    @staticmethod
    def _brenner_subrahmanyam_guess(market_price, S, K, T, r, option_type, q) -> float:
        """
        Guess initial adaptatif (Brenner-Subrahmanyam 1988).
        Pour ATM : σ ≈ sqrt(2π/T) * (price/S). Sinon fallback 0.3.
        """
        if T <= 0 or S <= 0:
            return 0.3
        moneyness = abs(np.log(S / K)) if K > 0 else 1.0
        # ATM : |ln(S/K)| < 0.05
        if moneyness < 0.05:
            guess = np.sqrt(2 * np.pi / T) * (market_price / S)
            guess = float(np.clip(guess, 0.01, 2.0))
            return guess if np.isfinite(guess) else 0.3
        return 0.3

    @staticmethod
    def _newton_raphson(market_price, S, K, T, r, option_type, q,
                        sigma_init=0.3, tol=1e-8, max_iter=100):
        """Newton-Raphson avec vega comme derivee."""
        sigma = sigma_init

        for _ in range(max_iter):
            if sigma <= 0.001 or sigma > 10:
                return None

            price = BlackScholes.get_price(S, K, T, r, sigma, option_type, q)
            
            # Vega (non normalise)
            d1, _ = BlackScholes._d1d2(S, K, T, r, sigma, q)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-12:
                return None

            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            sigma -= diff / vega

        return None

    @staticmethod
    def _brent(market_price, S, K, T, r, option_type, q,
               sigma_min=0.001, sigma_max=5.0):
        """Methode de Brent (bisection amelioree), toujours converge."""
        iv, _ = IVsolver._brent_with_reason(
            market_price, S, K, T, r, option_type, q, sigma_min, sigma_max
        )
        return iv

    @staticmethod
    def _brent_with_reason(market_price, S, K, T, r, option_type, q,
                           sigma_min=0.001, sigma_max=5.0) -> Tuple[Optional[float], str]:
        """Brent avec raison explicite en cas d'echec."""
        def objective(sigma):
            return BlackScholes.get_price(S, K, T, r, sigma, option_type, q) - market_price

        try:
            f_min = objective(sigma_min)
            f_max = objective(sigma_max)

            if f_min * f_max > 0:
                return None, IV_FAIL_BRENT_NO_ROOT

            iv = brentq(objective, sigma_min, sigma_max, xtol=1e-8, maxiter=200)
            if 0 < iv < 5.0:
                return iv, IV_OK
            return None, IV_FAIL_BRENT_NO_ROOT

        except (ValueError, RuntimeError):
            return None, IV_FAIL_BRENT_ERROR
