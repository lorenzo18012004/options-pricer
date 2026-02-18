"""
Données synthétiques pour le Pricer.

Source prioritaire : data/synthetic_data.xlsx (généré par scripts/generate_synthetic_excel.py)
Fallback : génération en code si Excel absent.

L'Excel contient un smile de volatilité réaliste (U-shape + skew) et une term structure.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

from core.black_scholes import BlackScholes

logger = logging.getLogger(__name__)

_EXCEL_PATH = Path(__file__).parent / "synthetic_data.xlsx"
_cache: Dict[str, Any] = {}


def _load_excel() -> bool:
    """Charge l'Excel en cache. Retourne True si chargé."""
    if not _EXCEL_PATH.exists():
        return False
    try:
        _cache["spot"] = pd.read_excel(_EXCEL_PATH, sheet_name="spot")
        _cache["options"] = pd.read_excel(_EXCEL_PATH, sheet_name="options")
        _cache["expirations"] = pd.read_excel(_EXCEL_PATH, sheet_name="expirations")
        _cache["treasury"] = pd.read_excel(_EXCEL_PATH, sheet_name="treasury")
        try:
            _cache["hist_vol"] = pd.read_excel(_EXCEL_PATH, sheet_name="hist_vol")
        except (ValueError, KeyError, ImportError):
            _cache["hist_vol"] = None
        try:
            _cache["dividends"] = pd.read_excel(_EXCEL_PATH, sheet_name="dividends")
        except (ValueError, KeyError, ImportError):
            _cache["dividends"] = None
        return True
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        logger.warning(f"Impossible de charger synthetic_data.xlsx: {e}")
        return False


def _use_excel() -> bool:
    if "excel_loaded" not in _cache:
        _cache["excel_loaded"] = _load_excel()
    return _cache.get("excel_loaded", False)


# Fallback : profils si pas d'Excel
_SYNTHETIC_PROFILES = {
    "AAPL": (175.0, 0.22, 0.005),
    "MSFT": (420.0, 0.20, 0.007),
    "TSLA": (250.0, 0.45, 0.0),
    "SPY": (580.0, 0.15, 0.012),
    "DEFAULT": (100.0, 0.25, 0.02),
}


def _get_profile(ticker: str) -> Tuple[float, float, float]:
    t = ticker.upper().strip()
    if _use_excel():
        df = _cache["spot"]
        row = df[df["ticker"].str.upper() == t]
        if not row.empty:
            r = row.iloc[0]
            return (float(r["spot"]), float(r["base_vol"]), float(r["div_yield"]))
    return _SYNTHETIC_PROFILES.get(t, _SYNTHETIC_PROFILES["DEFAULT"])


class SyntheticDataConnector:
    """
    Connecteur de données synthétiques.
    Lit depuis data/synthetic_data.xlsx si présent, sinon génère en code.
    """

    @staticmethod
    def get_spot_price(ticker_symbol: str, force_refresh: bool = False) -> float:
        spot, _, _ = _get_profile(ticker_symbol)
        return float(spot)

    @staticmethod
    def get_option_chain_with_synced_spot(
        ticker_symbol: str, expiration_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        calls, puts = SyntheticDataConnector.get_option_chain(ticker_symbol, expiration_date)
        spot = SyntheticDataConnector.get_spot_price(ticker_symbol, force_refresh=True)
        return calls, puts, spot

    @staticmethod
    def get_market_data(ticker_symbol: str) -> Dict[str, Any]:
        ticker_symbol = ticker_symbol.upper().strip()
        spot, base_vol, div_yield = _get_profile(ticker_symbol)
        return {
            "ticker": ticker_symbol,
            "spot": float(spot),
            "volatility": base_vol,
            "dividend_yield": div_yield,
            "name": f"{ticker_symbol} (Synthetic)",
            "market_cap": None,
            "sector": None,
            "currency": "USD",
        }

    @staticmethod
    def get_dividend_yield_forecast(ticker_symbol: str, spot: float) -> float:
        _, _, div_yield = _get_profile(ticker_symbol)
        return float(div_yield)

    @staticmethod
    def get_dividends_schedule(ticker_symbol: str, expiration_date: str) -> List[Tuple[float, float]]:
        """
        Calendrier des dividendes discrets (ex_date, amount) avant expiration.
        Utilisé pour le pricing des options américaines (BinomialTree).

        Source : UNIQUEMENT data/synthetic_data.xlsx (feuille "dividends").
        Ne jamais utiliser Yahoo Finance ici.

        Returns:
            Liste de (t, amount) avec t = temps en années depuis aujourd'hui
        """
        ticker_symbol = ticker_symbol.upper().strip()
        if not _use_excel() or _cache.get("dividends") is None:
            return []
        df = _cache["dividends"]
        if "ticker" not in df.columns or "ex_date" not in df.columns or "amount" not in df.columns:
            return []
        sub = df[df["ticker"].str.upper() == ticker_symbol]
        if sub.empty:
            return []
        today = datetime.now().date()
        try:
            exp_dt = datetime.strptime(str(expiration_date)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return []
        result = []
        for _, row in sub.iterrows():
            try:
                ex_d = pd.Timestamp(row["ex_date"]).date()
            except (ValueError, TypeError, KeyError):
                continue
            if today < ex_d <= exp_dt:
                t_years = (ex_d - today).days / 365.0
                amount = float(row.get("amount", 0) or 0)
                if amount > 0 and t_years > 0:
                    result.append((t_years, amount))
        return sorted(result, key=lambda x: x[0])

    @staticmethod
    def _fallback_expirations() -> List[str]:
        """Generate future expiration dates (Fridays)."""
        today = datetime.now().date()
        exps = []
        for days in [7, 14, 21, 28, 35, 42, 49, 63, 91, 120, 182, 365]:
            d = today + timedelta(days=days)
            days_until_friday = (4 - d.weekday()) % 7
            d = d + timedelta(days=days_until_friday)
            if d > today:
                exps.append(d.strftime("%Y-%m-%d"))
        return sorted(set(exps))[:18]

    @staticmethod
    def get_expirations(ticker_symbol: str) -> List[str]:
        if _use_excel():
            df = _cache["expirations"]
            ticker_symbol = ticker_symbol.upper().strip()
            rows = df[df["ticker"].str.upper() == ticker_symbol]
            if not rows.empty:
                today = datetime.now().date()
                exps = []
                for x in rows["expiration"].unique():
                    if pd.isna(x):
                        continue
                    try:
                        exp_d = pd.Timestamp(x).date()
                        if exp_d > today:
                            exps.append(exp_d.strftime("%Y-%m-%d"))
                    except (ValueError, TypeError):
                        continue
                if exps:
                    return sorted(exps)
        return SyntheticDataConnector._fallback_expirations()

    @staticmethod
    def get_option_chain(ticker_symbol: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ticker_symbol = ticker_symbol.upper().strip()
        exp_str = str(expiration_date)[:10]

        if _use_excel():
            df = _cache["options"]
            df_exp = df["expiration"].apply(
                lambda x: pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) else ""
            )
            mask = (df["ticker"].str.upper() == ticker_symbol) & (df_exp == exp_str)
            sub = df[mask]
            if not sub.empty:
                calls = sub[sub["type"] == "call"].copy()
                puts = sub[sub["type"] == "put"].copy()
                for c in ["bid", "ask", "iv", "volume", "openInterest", "strike"]:
                    if c in calls.columns:
                        calls[c] = pd.to_numeric(calls[c], errors="coerce")
                    if c in puts.columns:
                        puts[c] = pd.to_numeric(puts[c], errors="coerce")
                if "iv" in calls.columns:
                    calls["impliedVolatility"] = calls["iv"]
                    puts["impliedVolatility"] = puts["iv"]
                if "openInterest" not in calls.columns and "oi" in calls.columns:
                    calls["openInterest"] = calls["oi"]
                    puts["openInterest"] = puts["oi"]
                calls["contractSymbol"] = calls.apply(
                    lambda r: f"{ticker_symbol}_{exp_str}_{r['strike']}_C", axis=1
                )
                puts["contractSymbol"] = puts.apply(
                    lambda r: f"{ticker_symbol}_{exp_str}_{r['strike']}_P", axis=1
                )
                calls["lastTradeDate"] = datetime.now().strftime("%Y-%m-%d")
                puts["lastTradeDate"] = datetime.now().strftime("%Y-%m-%d")
                calls["lastPrice"] = (calls["bid"] + calls["ask"]) / 2
                puts["lastPrice"] = (puts["bid"] + puts["ask"]) / 2
                return calls.sort_values("strike").reset_index(drop=True), puts.sort_values("strike").reset_index(drop=True)

        # Fallback: in-memory generation (when Excel has no data for this expiration)
        spot, base_vol, div_yield = _get_profile(ticker_symbol)
        r = 0.04
        try:
            exp_dt = datetime.strptime(str(expiration_date)[:10], "%Y-%m-%d")
            T = max((exp_dt - datetime.now()).days / 365.0, 1e-4)
        except (ValueError, TypeError):
            T = 0.25

        def _svi_iv(K, atm_vol, T):
            """SVI Gatheral : w(k)=a+b[ρ(k-m)+√((k-m)²+σ²)], IV=√(w/T)."""
            k = np.log(K / spot)
            w_atm = (atm_vol ** 2) * T
            a, b = 0.5 * w_atm, 0.15 * np.sqrt(w_atm)
            rho, m, sig = -0.35, 0.02, 0.25
            x = k - m
            w = a + b * (rho * x + np.sqrt(x * x + sig * sig))
            return float(np.clip(np.sqrt(max(w / T, 1e-8)), 0.10, 0.60))

        atm_vol = base_vol * (1.0 + 0.35 * np.exp(-5.0 * T))
        strikes = np.round(np.linspace(spot * 0.75, spot * 1.25, 45), 2)

        def make_chain(opt_type: str) -> pd.DataFrame:
            rows = []
            for K in strikes:
                iv = _svi_iv(K, atm_vol, T)
                price = BlackScholes.get_price(spot, K, T, r, iv, opt_type, div_yield)
                bid = max(0.01, price * 0.98)
                ask = price * 1.02
                rows.append({
                    "strike": K, "bid": bid, "ask": ask, "lastPrice": price,
                    "volume": 500, "openInterest": 1000, "impliedVolatility": iv,
                    "contractSymbol": f"{ticker_symbol}_{expiration_date}_{K}_{opt_type[0]}",
                    "lastTradeDate": datetime.now().strftime("%Y-%m-%d"),
                })
            return pd.DataFrame(rows)

        return make_chain("call"), make_chain("put")

    @staticmethod
    def get_treasury_par_curve() -> Tuple[np.ndarray, np.ndarray]:
        if _use_excel():
            df = _cache["treasury"]
            mats = df["maturity"].values.astype(float)
            rates = df["rate"].values.astype(float)
            order = np.argsort(mats)
            return mats[order], rates[order]
        mats = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
        rates = np.array([0.035, 0.038, 0.040, 0.042, 0.043, 0.044, 0.0445, 0.045, 0.046])
        return mats, rates

    @staticmethod
    def get_risk_free_rate(time_to_maturity_years: float) -> float:
        from core.curves import YieldCurve
        mats, rates = SyntheticDataConnector.get_treasury_par_curve()
        curve = YieldCurve.bootstrap_from_par_yields(
            par_yields=rates.tolist(), maturities=mats.tolist(), payment_freq=2
        )
        return float(curve.get_zero_rate(time_to_maturity_years))

    @staticmethod
    def get_historical_prices(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
        spot, base_vol, _ = _get_profile(ticker_symbol)
        n_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504}.get(period, 252)
        rng = np.random.RandomState(hash(ticker_symbol.upper()) % 2**32)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
        returns = rng.standard_normal(n_days - 1) * base_vol / np.sqrt(252)
        prices = spot * np.exp(np.concatenate([[0], np.cumsum(returns)]))
        open_p = np.roll(prices, 1)
        open_p[0] = spot
        return pd.DataFrame({
            "Close": prices, "Open": open_p, "High": prices * 1.01, "Low": prices * 0.99,
            "Volume": rng.randint(1_000_000, 10_000_000, n_days),
        }, index=dates)

    @staticmethod
    def get_historical_volatility(
        ticker_symbol: str, window: int = None, maturity_days: int = None
    ) -> Optional[float]:
        if window is None:
            window = 30 if maturity_days is None else min(252, max(10, maturity_days))
        ticker_symbol = ticker_symbol.upper().strip()
        # Priorité : Excel hist_vol si disponible
        if _use_excel() and _cache.get("hist_vol") is not None:
            df = _cache["hist_vol"]
            sub = df[(df["ticker"].str.upper() == ticker_symbol) & (df["window_days"] == window)]
            if not sub.empty:
                return float(sub.iloc[0]["hist_vol"])
            # Interpolation sur la fenêtre la plus proche
            sub = df[df["ticker"].str.upper() == ticker_symbol].sort_values("window_days")
            if not sub.empty:
                w = sub["window_days"].values
                v = sub["hist_vol"].values
                idx = np.searchsorted(w, window)
                if idx == 0:
                    return float(v[0])
                if idx >= len(w):
                    return float(v[-1])
                # Interpolation linéaire
                w0, w1 = w[idx - 1], w[idx]
                v0, v1 = v[idx - 1], v[idx]
                return float(v0 + (v1 - v0) * (window - w0) / (w1 - w0))
        try:
            hist = SyntheticDataConnector.get_historical_prices(
                ticker_symbol, period="1y" if window <= 252 else "2y"
            )
            if len(hist) < window + 1:
                _, base_vol, _ = _get_profile(ticker_symbol)
                return base_vol
            returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            vol = float(returns.tail(window).std() * np.sqrt(252))
            return vol if np.isfinite(vol) and vol > 0.01 else None
        except (KeyError, ValueError, IndexError, TypeError):
            _, base_vol, _ = _get_profile(ticker_symbol)
            return base_vol
