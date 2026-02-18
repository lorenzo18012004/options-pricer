"""
Protocol (interface) pour les connecteurs de données.

Permet le typage et l'interchangeabilité entre DataConnector et SyntheticDataConnector.
"""

from typing import Dict, List, Optional, Tuple, Union, Protocol

import numpy as np
import pandas as pd


class DataConnectorProtocol(Protocol):
    """Interface commune pour les connecteurs de données (Yahoo, Synthétique, Fallback)."""

    @staticmethod
    def get_spot_price(ticker_symbol: str, force_refresh: bool = False) -> float:
        """Prix spot actuel."""
        ...

    @staticmethod
    def get_expirations(ticker_symbol: str) -> List[str]:
        """Dates d'expiration disponibles."""
        ...

    @staticmethod
    def get_market_data(ticker_symbol: str) -> Dict:
        """Données de marché (spot, vol, dividend_yield, etc.)."""
        ...

    @staticmethod
    def get_option_chain(ticker_symbol: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chaîne d'options (calls, puts)."""
        ...

    @staticmethod
    def get_option_chain_with_synced_spot(
        ticker_symbol: str, expiration_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Chaîne + spot synchronisé."""
        ...

    @staticmethod
    def get_dividend_yield_forecast(ticker_symbol: str, spot: float) -> float:
        """Dividend yield estimé."""
        ...

    @staticmethod
    def get_dividends_schedule(
        ticker_symbol: str, expiration_date: str
    ) -> List[Tuple[float, float]]:
        """Calendrier dividendes discrets (t, amount)."""
        ...

    @staticmethod
    def get_risk_free_rate(time_to_maturity_years: float) -> float:
        """Taux sans risque pour une maturité."""
        ...

    @staticmethod
    def get_historical_volatility(
        ticker_symbol: str,
        window: Optional[int] = None,
        maturity_days: Optional[int] = None,
    ) -> Optional[float]:
        """Volatilité historique."""
        ...

    @staticmethod
    def get_treasury_par_curve() -> Tuple[np.ndarray, np.ndarray]:
        """Courbe Treasury (maturities, rates)."""
        ...
