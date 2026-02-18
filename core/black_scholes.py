"""Black-Scholes-Merton pricing model for European options."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    Modele de Black-Scholes-Merton pour le pricing d'options europeennes.
    
    Supporte les dividendes continus (dividend yield q).
    Greeks 1er ordre : Delta, Gamma, Vega, Theta, Rho
    Greeks 2nd ordre : Vanna, Volga (Vomma), Charm, Speed, Color, Zomma
    
    Formules :
        d1 = [ln(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        Call = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        Put  = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    """

    @staticmethod
    def _d1d2(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> tuple:
        """Calcule d1 et d2."""
        if not (np.isfinite(S) and np.isfinite(K)):
            raise ValueError("S and K must be finite")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be > 0")
        if T <= 0:
            raise ValueError("T must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if sigma < 1e-10:
            raise ValueError("sigma too small, numerical instability")
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # =================================================================
    # PRICING
    # =================================================================

    @staticmethod
    def get_price(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call", q: float = 0.0
    ) -> float:
        """Prix d'une option europeenne (Black-Scholes-Merton)."""
        if not (np.isfinite(S) and np.isfinite(K)):
            raise ValueError("S and K must be finite")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be > 0")
        if not (np.isfinite(sigma) and np.isfinite(T) and np.isfinite(r)):
            raise ValueError("sigma, T, r must be finite")
        if sigma <= 0 and T > 0:
            raise ValueError("sigma must be > 0 for T > 0")
        if T <= 0:
            if option_type.lower() == "call":
                return max(S - K, 0)
            return max(K - S, 0)

        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)

        if option_type.lower() == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    @staticmethod
    def digital_call(S, K, T, r, sigma, q=0.0) -> float:
        """Digital call (cash-or-nothing) : paie 1 si S>K à l'échéance. Valeur = e^(-rT)*N(d2)."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        _, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        return np.exp(-r * T) * norm.cdf(d2)

    # =================================================================
    # GREEKS 1ER ORDRE
    # =================================================================

    @staticmethod
    def get_delta(S, K, T, r, sigma, option_type="call", q=0.0):
        """Delta : dV/dS. Call: e^(-qT)*N(d1), Put: e^(-qT)*(N(d1)-1)."""
        if T <= 0:
            if option_type.lower() == "call":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0

        d1, _ = BlackScholes._d1d2(S, K, T, r, sigma, q)
        if option_type.lower() == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def get_gamma(S, K, T, r, sigma, q=0.0):
        """Gamma : d2V/dS2. Identique call/put."""
        if T <= 0:
            return 0.0
        d1, _ = BlackScholes._d1d2(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def get_vega(S, K, T, r, sigma, q=0.0):
        """Vega : dV/dsigma. Normalise pour 1% de vol (/100). Identique call/put."""
        if T <= 0:
            return 0.0
        d1, _ = BlackScholes._d1d2(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def get_theta(S, K, T, r, sigma, option_type="call", q=0.0):
        """Theta : -dV/dT. Normalise pour 1 jour (/365)."""
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == "call":
            theta = (term1
                     + q * S * np.exp(-q * T) * norm.cdf(d1)
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (term1
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
        return theta / 365

    @staticmethod
    def get_rho(S, K, T, r, sigma, option_type="call", q=0.0):
        """Rho : dV/dr. Normalise pour 1% de taux (/100)."""
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        if option_type.lower() == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    # =================================================================
    # GREEKS 2ND ORDRE
    # =================================================================

    @staticmethod
    def get_vanna(S, K, T, r, sigma, q=0.0):
        """
        Vanna : d2V/dSdsigma = dDelta/dsigma = dVega/dS.
        Mesure la sensibilite du delta a la vol (et du vega au spot).
        Crucial pour le vol hedging et la gestion du skew.
        Vanna = -e^(-qT) * phi(d1) * d2 / sigma
        """
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

    @staticmethod
    def get_volga(S, K, T, r, sigma, q=0.0):
        """
        Volga (Vomma) : d2V/dsigma2 = dVega/dsigma.
        Mesure la convexite du prix par rapport a la vol.
        Volga = Vega * d1 * d2 / sigma
        (ici Vega non-normalise)
        """
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        vega_raw = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega_raw * d1 * d2 / sigma / 100  # normalise pour 1% de vol

    @staticmethod
    def get_charm(S, K, T, r, sigma, option_type="call", q=0.0):
        """
        Charm : dDelta/dT (delta decay).
        Mesure comment le delta change avec le passage du temps.
        Essentiel pour le delta hedging dynamique : indique combien
        de delta on perd/gagne chaque jour sans rebalancer.
        Normalise pour 1 jour (/365).
        """
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)

        charm_common = np.exp(-q * T) * norm.pdf(d1) * (
            2 * (r - q) * T - d2 * sigma * np.sqrt(T)
        ) / (2 * T * sigma * np.sqrt(T))

        if option_type.lower() == "call":
            charm = -q * np.exp(-q * T) * norm.cdf(d1) + charm_common
        else:
            charm = q * np.exp(-q * T) * norm.cdf(-d1) + charm_common

        return charm / 365  # par jour

    @staticmethod
    def get_speed(S, K, T, r, sigma, q=0.0):
        """
        Speed : d3V/dS3 = dGamma/dS.
        Mesure la sensibilite du gamma au spot.
        Speed = -Gamma/S * (1 + d1/(sigma*sqrt(T)))
        """
        if T <= 0:
            return 0.0
        d1, _ = BlackScholes._d1d2(S, K, T, r, sigma, q)
        gamma = BlackScholes.get_gamma(S, K, T, r, sigma, q)
        return -gamma / S * (1 + d1 / (sigma * np.sqrt(T)))

    @staticmethod
    def get_zomma(S, K, T, r, sigma, q=0.0):
        """
        Zomma : d3V/dS2dsigma = dGamma/dsigma.
        Mesure la sensibilite du gamma a la vol.
        Zomma = Gamma * (d1*d2 - 1) / sigma
        """
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        gamma = BlackScholes.get_gamma(S, K, T, r, sigma, q)
        return gamma * (d1 * d2 - 1) / sigma

    @staticmethod
    def get_color(S, K, T, r, sigma, q=0.0):
        """
        Color : d3V/dS2dT = dGamma/dT.
        Mesure la sensibilite du gamma au temps.
        Normalise pour 1 jour (/365).
        """
        if T <= 0:
            return 0.0
        d1, d2 = BlackScholes._d1d2(S, K, T, r, sigma, q)
        color = -np.exp(-q * T) * norm.pdf(d1) / (2 * S * T * sigma * np.sqrt(T)) * (
            2 * q * T + 1 + d1 * (2 * (r - q) * T - d2 * sigma * np.sqrt(T))
            / (sigma * np.sqrt(T))
        )
        return color / 365

    # =================================================================
    # AGGREGATION
    # =================================================================

    @staticmethod
    def get_all_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
        """Retourne tous les Grecs (1er et 2nd ordre) dans un dictionnaire."""
        return {
            # Prix
            'price': BlackScholes.get_price(S, K, T, r, sigma, option_type, q),
            # 1er ordre
            'delta': BlackScholes.get_delta(S, K, T, r, sigma, option_type, q),
            'gamma': BlackScholes.get_gamma(S, K, T, r, sigma, q),
            'vega': BlackScholes.get_vega(S, K, T, r, sigma, q),
            'theta': BlackScholes.get_theta(S, K, T, r, sigma, option_type, q),
            'rho': BlackScholes.get_rho(S, K, T, r, sigma, option_type, q),
            # 2nd ordre
            'vanna': BlackScholes.get_vanna(S, K, T, r, sigma, q),
            'volga': BlackScholes.get_volga(S, K, T, r, sigma, q),
            'charm': BlackScholes.get_charm(S, K, T, r, sigma, option_type, q),
            'speed': BlackScholes.get_speed(S, K, T, r, sigma, q),
            'zomma': BlackScholes.get_zomma(S, K, T, r, sigma, q),
            'color': BlackScholes.get_color(S, K, T, r, sigma, q),
        }

    # =================================================================
    # P&L ATTRIBUTION
    # =================================================================

    @staticmethod
    def pnl_attribution(S_old, S_new, sigma_old, sigma_new, K, T, r,
                        option_type="call", q=0.0, days_passed=1):
        """
        Decompose le P&L d'une option en composantes Greeks.
        
        C'est ce que chaque desk de trading fait chaque matin :
        analyser d'ou vient le P&L de la veille.
        
        Args:
            S_old, S_new: Spot ancien et nouveau
            sigma_old, sigma_new: Vol implicite ancienne et nouvelle
            K: Strike
            T: Temps restant AVANT le move (en annees)
            r: Taux sans risque
            option_type: "call" ou "put"
            q: Dividend yield
            days_passed: Nombre de jours ecoules
        
        Returns:
            dict: decomposition P&L par composante
        """
        dS = S_new - S_old
        dsigma = sigma_new - sigma_old
        dT = days_passed / 365.0

        # Greeks au point initial
        delta = BlackScholes.get_delta(S_old, K, T, r, sigma_old, option_type, q)
        gamma = BlackScholes.get_gamma(S_old, K, T, r, sigma_old, q)
        vega_raw = BlackScholes.get_vega(S_old, K, T, r, sigma_old, q) * 100  # de-normalise
        theta_daily = BlackScholes.get_theta(S_old, K, T, r, sigma_old, option_type, q)
        vanna = BlackScholes.get_vanna(S_old, K, T, r, sigma_old, q)

        # P&L par composante
        delta_pnl = delta * dS
        gamma_pnl = 0.5 * gamma * dS ** 2
        vega_pnl = vega_raw * dsigma
        theta_pnl = theta_daily * days_passed  # theta_daily < 0 pour long, days > 0 => perte
        vanna_pnl = vanna * dS * dsigma

        # P&L reel (prix nouveau - prix ancien)
        price_old = BlackScholes.get_price(S_old, K, T, r, sigma_old, option_type, q)
        price_new = BlackScholes.get_price(S_new, K, T - dT, r, sigma_new, option_type, q)
        actual_pnl = price_new - price_old

        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl
        unexplained = actual_pnl - explained

        return {
            'actual_pnl': actual_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'vanna_pnl': vanna_pnl,
            'explained_pnl': explained,
            'unexplained_pnl': unexplained,
            'explanation_ratio': (explained / actual_pnl * 100) if abs(actual_pnl) > 1e-10 else None,
        }
