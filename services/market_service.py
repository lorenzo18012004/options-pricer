from datetime import datetime
from typing import Dict, List, Tuple, Union, Type

import numpy as np
import streamlit as st

from data import DataConnector, SyntheticDataConnector
from data.protocols import DataConnectorProtocol


# Exceptions qui déclenchent le fallback Yahoo -> Synthétique
try:
    from data.exceptions import NetworkError, DataError
    _FALLBACK_EXCEPTIONS = (ValueError, ConnectionError, TimeoutError, OSError, NetworkError, DataError)
except ImportError:
    _FALLBACK_EXCEPTIONS = (ValueError, ConnectionError, TimeoutError, OSError)


class FallbackDataConnector:
    """
    Wrapper : essaie DataConnector (Yahoo) puis SyntheticDataConnector en secours.
    Une fois le fallback activé, utilise Synthetic pour tous les appels suivants.
    Peut être instancié avec un connecteur synthétique injecté (découplage de Streamlit).
    """

    def __init__(self, synthetic_connector=None, state_store: dict = None):
        self._use_fallback = False
        self._synthetic = synthetic_connector or SyntheticDataConnector
        self._state = state_store

    def _get_connector(self):
        return self._synthetic if self._use_fallback else DataConnector

    def _try_or_fallback(self, method_name: str, *args, **kwargs):
        if self._use_fallback:
            return getattr(self._synthetic, method_name)(*args, **kwargs)
        try:
            return getattr(DataConnector, method_name)(*args, **kwargs)
        except _FALLBACK_EXCEPTIONS:
            self._use_fallback = True
            store = self._state if self._state is not None else st.session_state
            if "data_connector_fallback" not in store:
                store["data_connector_fallback"] = True
                st.warning("**Yahoo Finance indisponible** — Données synthétiques utilisées en secours.")
            return getattr(self._synthetic, method_name)(*args, **kwargs)

    @staticmethod
    def _get_instance():
        c = st.session_state.get("_fallback_connector")
        if c is None:
            c = FallbackDataConnector(state_store=st.session_state)
            st.session_state["_fallback_connector"] = c
        return c

    @staticmethod
    def get_spot_price(ticker_symbol: str, force_refresh: bool = False) -> float:
        return FallbackDataConnector._get_instance()._try_or_fallback("get_spot_price", ticker_symbol, force_refresh)

    @staticmethod
    def get_expirations(ticker_symbol: str) -> List[str]:
        return FallbackDataConnector._get_instance()._try_or_fallback("get_expirations", ticker_symbol)

    @staticmethod
    def get_market_data(ticker_symbol: str) -> Dict:
        return FallbackDataConnector._get_instance()._try_or_fallback("get_market_data", ticker_symbol)

    @staticmethod
    def get_option_chain(ticker_symbol: str, expiration_date: str):
        return FallbackDataConnector._get_instance()._try_or_fallback("get_option_chain", ticker_symbol, expiration_date)

    @staticmethod
    def get_option_chain_with_synced_spot(ticker_symbol: str, expiration_date: str):
        return FallbackDataConnector._get_instance()._try_or_fallback("get_option_chain_with_synced_spot", ticker_symbol, expiration_date)

    @staticmethod
    def get_dividend_yield_forecast(ticker_symbol: str, spot: float) -> float:
        return FallbackDataConnector._get_instance()._try_or_fallback("get_dividend_yield_forecast", ticker_symbol, spot)

    @staticmethod
    def get_dividends_schedule(ticker_symbol: str, expiration_date: str):
        return FallbackDataConnector._get_instance()._try_or_fallback("get_dividends_schedule", ticker_symbol, expiration_date)

    @staticmethod
    def get_risk_free_rate(time_to_maturity_years: float) -> float:
        return FallbackDataConnector._get_instance()._try_or_fallback("get_risk_free_rate", time_to_maturity_years)

    @staticmethod
    def get_historical_volatility(ticker_symbol: str, window: int = None, maturity_days: int = None):
        return FallbackDataConnector._get_instance()._try_or_fallback("get_historical_volatility", ticker_symbol, window=window, maturity_days=maturity_days)

    @staticmethod
    def get_treasury_par_curve():
        return FallbackDataConnector._get_instance()._try_or_fallback("get_treasury_par_curve")


def get_data_connector(use_synthetic: bool = False):
    """
    Retourne le connecteur de données.
    Si use_synthetic=False : FallbackDataConnector (Yahoo avec fallback Synthétique).
    Si use_synthetic=True : SyntheticDataConnector uniquement.
    """
    if use_synthetic or st.session_state.get("data_source") == "Synthétique":
        # Réinitialiser l'état fallback quand on utilise Synthétique
        st.session_state.pop("data_connector_fallback", None)
        st.session_state.pop("_fallback_connector", None)
        return SyntheticDataConnector
    return FallbackDataConnector


def load_market_snapshot(
    ticker: str,
    connector: Union[Type[DataConnectorProtocol], Type[DataConnector], Type[SyntheticDataConnector], Type[FallbackDataConnector]] = None,
) -> Tuple[float, List[str], Dict, float]:
    """
    Load core market snapshot for a ticker.

    Args:
        ticker: Symbole de l'actif
        connector: DataConnector, SyntheticDataConnector ou FallbackDataConnector

    Returns:
        (spot, expirations, market_data, dividend_yield)
    """
    from core.request_context import set_request_id
    set_request_id()

    if connector is None:
        use_syn = st.session_state.get("data_source") == "Synthétique"
        connector = get_data_connector(use_syn)
    spot = connector.get_spot_price(ticker)
    expirations = connector.get_expirations(ticker)
    if not expirations:
        raise ValueError(f"No options data for {ticker}")
    market_data = connector.get_market_data(ticker)
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
        except (ValueError, TypeError, KeyError):
            continue
    return exp_options


def require_hist_vol_market_only(
    ticker: str,
    window_days: int,
    connector: Union[Type[DataConnectorProtocol], Type[DataConnector], Type[SyntheticDataConnector]] = None,
) -> float:
    """
    Récupère la volatilité historique (échoue si indisponible).

    Args:
        ticker: Symbole
        window_days: Fenêtre en jours ouvrés
        connector: DataConnector ou SyntheticDataConnector (défaut: DataConnector)
    """
    if connector is None:
        connector = DataConnector
    hist_vol = connector.get_historical_volatility(ticker, window=window_days)
    if hist_vol is None or not np.isfinite(hist_vol) or hist_vol <= 0:
        raise ValueError("Historical volatility unavailable.")
    return float(hist_vol)
