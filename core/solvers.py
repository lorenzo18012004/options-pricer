import numpy as np
from scipy.optimize import brentq
from core.black_scholes import BlackScholes


class IVsolver:
    """
    Implied Volatility Solver.
    
    Methode principale : Newton-Raphson (rapide, quadratique).
    Fallback : Brent (robuste, garanti de converger si IV existe).
    """

    @staticmethod
    def find_implied_vol(market_price, S, K, T, r, option_type="call", q=0.0):
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
        if market_price <= 0 or T <= 0:
            return None

        # Verifier que le prix est dans les bornes theoriques
        if option_type.lower() == "call":
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
            upper_bound = S * np.exp(-q * T)
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
            upper_bound = K * np.exp(-r * T)

        if market_price < intrinsic - 0.01 or market_price > upper_bound + 0.01:
            return None

        # 1. Newton-Raphson (rapide)
        iv = IVsolver._newton_raphson(market_price, S, K, T, r, option_type, q)
        if iv is not None:
            return iv

        # 2. Fallback Brent (robuste)
        iv = IVsolver._brent(market_price, S, K, T, r, option_type, q)
        return iv

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
            vega = S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)

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
        def objective(sigma):
            return BlackScholes.get_price(S, K, T, r, sigma, option_type, q) - market_price

        try:
            # Verifier que la racine est dans l'intervalle
            f_min = objective(sigma_min)
            f_max = objective(sigma_max)

            if f_min * f_max > 0:
                return None

            iv = brentq(objective, sigma_min, sigma_max, xtol=1e-8, maxiter=200)
            return iv if 0 < iv < 5.0 else None

        except (ValueError, RuntimeError):
            return None


def norm_pdf(x):
    """Standard normal PDF (plus rapide que scipy pour un scalaire)."""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
