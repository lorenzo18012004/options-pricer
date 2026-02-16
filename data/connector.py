"""
Data Connector via Yahoo Finance (yfinance)
Donnees de marche en temps reel : prix, options, volatilite, taux

Fournit :
- Prix spot live
- Vraies chaines d'options (expirations, strikes, bid/ask, IV)
- Volatilite historique calculee depuis les prix reels
- Taux sans risque (US Treasury via ^IRX, ^FVX, ^TNX, ^TYX)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import time
from threading import Lock

import yfinance as yf

logger = logging.getLogger(__name__)

# Cache simple pour eviter de surcharger Yahoo Finance
_cache: Dict[str, dict] = {}
_cache_ttl = 30  # secondes
_cache_lock = Lock()


def _get_cached(key: str) -> Optional[Any]:
    """Recupere une valeur du cache si elle n'est pas expiree."""
    with _cache_lock:
        if key in _cache:
            entry = _cache[key]
            if time.time() - entry['time'] < _cache_ttl:
                return entry['value']
    return None


def _set_cached(key: str, value: Any):
    """Stocke une valeur dans le cache."""
    with _cache_lock:
        _cache[key] = {'value': value, 'time': time.time()}


class DataConnector:
    """
    Connecteur de donnees Yahoo Finance pour le Pricer.

    Donnees live :
    - Prix spot (prix actuel du marche)
    - Chaines d'options reelles (expirations, strikes, bid/ask, IV, volume, OI)
    - Volatilite historique (calculee depuis l'historique des prix)
    - Taux sans risque (US Treasury yields)

    Usage:
        spot = DataConnector.get_spot_price("AAPL")
        data = DataConnector.get_market_data("AAPL")
        calls, puts = DataConnector.get_option_chain("AAPL", "2025-03-21")
    """

    @classmethod
    def initialize(cls, **kwargs):
        """Pas d'initialisation necessaire pour yfinance."""
        logger.info("DataConnector yfinance pret (aucune cle API requise)")

    @classmethod
    def _ensure_initialized(cls):
        """Rien a faire - yfinance fonctionne directement."""
        pass

    # =========================================================================
    # Donnees de marche
    # =========================================================================

    @staticmethod
    def get_spot_price(ticker_symbol: str, force_refresh: bool = False) -> float:
        """
        Recupere le prix spot actuel.

        Args:
            ticker_symbol: Symbole Yahoo Finance (ex: "AAPL", "MSFT", "SPY")
            force_refresh: Si True, ignore le cache (pour synchroniser avec les options)

        Returns:
            float: Prix spot actuel
        """
        ticker_symbol = ticker_symbol.upper().strip()

        if not force_refresh:
            cached = _get_cached(f"spot_{ticker_symbol}")
            if cached is not None:
                return cached

        try:
            ticker = yf.Ticker(ticker_symbol)

            # Essayer fast_info d'abord (plus rapide)
            price = None
            try:
                price = ticker.fast_info.get('lastPrice')
                if price is None or price == 0:
                    price = ticker.fast_info.get('last_price')
            except Exception:
                pass

            # Fallback : historique recent
            if price is None or price == 0:
                hist = ticker.history(period="5d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])

            if price is None or price == 0:
                raise ValueError(f"No price data for '{ticker_symbol}'")

            price = float(price)
            _set_cached(f"spot_{ticker_symbol}", price)
            return price

        except Exception as e:
            logger.error(f"Erreur prix pour {ticker_symbol}: {e}")
            raise ValueError(f"Unable to retrieve price for '{ticker_symbol}': {e}")

    @staticmethod
    def get_option_chain_with_synced_spot(
        ticker_symbol: str, expiration_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Recupere la chaine d'options puis le spot (meme ordre que les quotes)
        pour minimiser le decalage temporel spot vs options.
        """
        calls, puts = DataConnector.get_option_chain(ticker_symbol, expiration_date)
        spot = DataConnector.get_spot_price(ticker_symbol, force_refresh=True)
        return calls, puts, spot

    @staticmethod
    def get_market_data(ticker_symbol: str) -> Dict[str, Any]:
        """
        Recupere toutes les donnees de marche en un appel.

        Returns:
            dict: ticker, spot, volatility, dividend_yield, market_cap, name
        """
        ticker_symbol = ticker_symbol.upper().strip()

        cached = _get_cached(f"market_{ticker_symbol}")
        if cached is not None:
            return cached

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            # Prix spot
            spot = info.get('currentPrice') or info.get('regularMarketPrice')
            if not spot:
                spot = DataConnector.get_spot_price(ticker_symbol)

            # Volatilite historique 30 jours
            vol = None
            try:
                hist = ticker.history(period="3mo")
                if len(hist) >= 20:
                    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                    vol = float(returns.tail(30).std() * np.sqrt(252))
            except Exception as e:
                logger.warning(f"Calcul vol echoue pour {ticker_symbol}: {e}")

            # Dividend yield - yfinance retourne en decimal (0.005 = 0.5%)
            # Mais certaines versions retournent en % (0.5 = 0.5%)
            # On verifie et corrige si necessaire
            div_yield = info.get('dividendYield')
            if div_yield is not None:
                div_yield = float(div_yield)
                # Calcul de sens : si div_yield > 0.15 (15%), c'est surement en %
                # car aucune action n'a un yield > 15%
                if div_yield > 0.15 and spot:
                    # Verifier avec le dividende annuel
                    annual_div = info.get('dividendRate', 0) or 0
                    if annual_div > 0 and spot > 0:
                        div_yield = annual_div / spot  # recalculer proprement
                    else:
                        div_yield = div_yield / 100  # fallback: diviser par 100

            market_data = {
                'ticker': ticker_symbol,
                'spot': float(spot) if spot else None,
                'volatility': vol,
                'dividend_yield': div_yield if div_yield else 0.0,
                'name': info.get('shortName', ticker_symbol),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'currency': info.get('currency', 'USD'),
            }

            _set_cached(f"market_{ticker_symbol}", market_data)

            logger.info(
                f"{ticker_symbol}: spot=${market_data['spot']:.2f}, "
                f"vol={market_data['volatility']:.1%}" if market_data['volatility'] else
                f"{ticker_symbol}: spot=${market_data['spot']:.2f}"
            )

            return market_data

        except Exception as e:
            logger.error(f"Erreur market data pour {ticker_symbol}: {e}")
            raise ValueError(f"Unable to retrieve data for '{ticker_symbol}': {e}")

    @staticmethod
    def get_dividend_yield_forecast(ticker_symbol: str, spot: float) -> float:
        """
        Estime le dividend yield (forecast) a partir de l'historique Yahoo.
        Utilise dividendYield/dividendRate si dispo, sinon somme des 4 derniers
        dividendes / spot = yield annuel implicite.
        """
        ticker_symbol = ticker_symbol.upper().strip()
        if spot <= 0:
            return 0.0
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            div_yield = info.get('dividendYield')
            if div_yield is not None:
                div_yield = float(div_yield)
                if 0 < div_yield <= 0.20:
                    return div_yield
                if div_yield > 0.15:
                    annual_div = info.get('dividendRate', 0) or 0
                    if annual_div > 0:
                        return float(annual_div) / spot
                    return div_yield / 100.0
            annual_div = info.get('dividendRate', 0) or 0
            if annual_div > 0:
                return float(annual_div) / spot
            divs = ticker.dividends
            if divs is not None and len(divs) >= 1:
                last_4 = divs.tail(4)
                annual_est = float(last_4.sum())
                if annual_est > 0:
                    return annual_est / spot
        except Exception as e:
            logger.debug(f"Dividend forecast failed for {ticker_symbol}: {e}")
        return 0.0

    # =========================================================================
    # Options
    # =========================================================================

    @staticmethod
    def get_expirations(ticker_symbol: str) -> List[str]:
        """
        Recupere les vraies dates d'expiration disponibles sur le marche.

        Args:
            ticker_symbol: Symbole (ex: "AAPL")

        Returns:
            Liste de dates d'expiration au format "YYYY-MM-DD"
        """
        ticker_symbol = ticker_symbol.upper().strip()

        cached = _get_cached(f"exp_{ticker_symbol}")
        if cached is not None:
            return cached

        try:
            ticker = yf.Ticker(ticker_symbol)
            expirations = list(ticker.options)  # Tuple de dates "YYYY-MM-DD"

            # Retry si vide (Yahoo bloque parfois les IP datacenter au 1er essai)
            for attempt in range(2):
                if expirations:
                    break
                logger.warning(f"Retry {attempt + 1}/2 pour {ticker_symbol} (options vides)")
                time.sleep(1.5)
                ticker = yf.Ticker(ticker_symbol)
                expirations = list(ticker.options)

            if not expirations:
                logger.warning(f"Aucune expiration trouvee pour {ticker_symbol}")
                return []

            _set_cached(f"exp_{ticker_symbol}", expirations)
            logger.info(f"{ticker_symbol}: {len(expirations)} expirations disponibles")
            return expirations

        except Exception as e:
            logger.error(f"Erreur expirations pour {ticker_symbol}: {e}")
            return []

    @staticmethod
    def get_option_chain(ticker_symbol: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Recupere la vraie chaine d'options du marche.

        Args:
            ticker_symbol: Symbole (ex: "AAPL")
            expiration_date: Date d'expiration "YYYY-MM-DD"

        Returns:
            Tuple (calls DataFrame, puts DataFrame) avec colonnes :
            strike, bid, ask, lastPrice, volume, openInterest, impliedVolatility
        """
        ticker_symbol = ticker_symbol.upper().strip()

        cache_key = f"chain_{ticker_symbol}_{expiration_date}"
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ticker = yf.Ticker(ticker_symbol)
            chain = ticker.option_chain(expiration_date)

            # Formater les DataFrames
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Renommer les colonnes pour compatibilite avec le dashboard
            rename_map = {
                'contractSymbol': 'contractSymbol',
                'lastTradeDate': 'lastTradeDate',
                'strike': 'strike',
                'lastPrice': 'lastPrice',
                'bid': 'bid',
                'ask': 'ask',
                'change': 'change',
                'percentChange': 'percentChange',
                'volume': 'volume',
                'openInterest': 'openInterest',
                'impliedVolatility': 'impliedVolatility',
                'inTheMoney': 'inTheMoney',
            }

            # S'assurer que les colonnes essentielles existent
            for df in [calls, puts]:
                if 'volume' in df.columns:
                    df['volume'] = df['volume'].fillna(0).astype(int)
                if 'openInterest' in df.columns:
                    df['openInterest'] = df['openInterest'].fillna(0).astype(int)
                if 'bid' in df.columns:
                    df['bid'] = df['bid'].fillna(0)
                if 'ask' in df.columns:
                    df['ask'] = df['ask'].fillna(0)

            # Trier par strike
            if not calls.empty:
                calls = calls.sort_values('strike').reset_index(drop=True)
            if not puts.empty:
                puts = puts.sort_values('strike').reset_index(drop=True)

            _set_cached(cache_key, (calls, puts))

            logger.info(
                f"{ticker_symbol} {expiration_date}: "
                f"{len(calls)} calls, {len(puts)} puts"
            )

            return calls, puts

        except Exception as e:
            logger.error(f"Erreur option chain pour {ticker_symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # =========================================================================
    # Taux sans risque
    # =========================================================================

    @staticmethod
    def get_treasury_par_curve() -> Tuple[np.ndarray, np.ndarray]:
        """
        Recupere la courbe de par yields Treasury US depuis Yahoo.

        Returns:
            (maturities, par_yields) avec taux en decimal
        """
        cached = _get_cached("treasury_par_curve")
        if cached is not None:
            return cached

        treasury_map = {
            0.25: "^IRX",   # 13-week T-Bill
            5.0: "^FVX",    # 5Y
            10.0: "^TNX",   # 10Y
            30.0: "^TYX",   # 30Y
        }

        mats = []
        rates = []
        for mat, ticker_symbol in sorted(treasury_map.items()):
            try:
                t = yf.Ticker(ticker_symbol)
                hist = t.history(period="5d")
                if hist.empty:
                    continue
                rate = float(hist["Close"].iloc[-1]) / 100.0
                if np.isfinite(rate):
                    mats.append(float(mat))
                    rates.append(rate)
                    _set_cached(f"treasury_{ticker_symbol}", rate)
            except Exception as e:
                logger.warning(f"Erreur taux {ticker_symbol}: {e}")
                continue

        if len(mats) < 2:
            raise ValueError(
                "Yahoo Treasury curve unavailable: insufficient live rate points "
                "(need at least 2 maturities)."
            )

        mats = np.asarray(mats, dtype=float)
        rates = np.asarray(rates, dtype=float)
        order = np.argsort(mats)
        mats = mats[order]
        rates = rates[order]

        _set_cached("treasury_par_curve", (mats, rates))
        return mats, rates

    @staticmethod
    def get_risk_free_rate(time_to_maturity_years: float) -> float:
        """
        Recupere le taux sans risque depuis la courbe Treasury US live
        en construisant une zero-curve par bootstrap sur les par yields.

        Tickers Yahoo Finance :
        - ^IRX : 13-week Treasury Bill (3 mois)
        - ^FVX : 5-year Treasury Note
        - ^TNX : 10-year Treasury Note
        - ^TYX : 30-year Treasury Bond

        Le taux retourne est un zero-rate interpolable pour la maturite demandee.
        """
        cached = _get_cached(f"rate_curve_{time_to_maturity_years:.4f}")
        if cached is not None:
            return cached

        curve_maturities, curve_rates = DataConnector.get_treasury_par_curve()

        # Build a bootstrapped zero curve from market par yields, then read r(T).
        from core.curves import YieldCurve
        curve = YieldCurve.bootstrap_from_par_yields(
            par_yields=curve_rates,
            maturities=curve_maturities,
            payment_freq=2,
        )
        rate = float(curve.get_zero_rate(time_to_maturity_years))

        _set_cached(f"rate_curve_{time_to_maturity_years:.4f}", rate)
        logger.info(f"Taux interpole ({time_to_maturity_years:.2f}Y): {rate:.4f}")
        return rate

    # =========================================================================
    # Historique et volatilite
    # =========================================================================

    @staticmethod
    def get_historical_prices(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Recupere l'historique des prix.

        Args:
            ticker_symbol: Symbole
            period: Periode ("1mo", "3mo", "6mo", "1y", "2y", "5y")

        Returns:
            DataFrame avec colonnes: Date, Open, High, Low, Close, Volume
        """
        ticker_symbol = ticker_symbol.upper().strip()

        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                logger.warning(f"Pas d'historique pour {ticker_symbol}")
                return pd.DataFrame()

            return hist

        except Exception as e:
            logger.error(f"Erreur historique pour {ticker_symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_historical_volatility(ticker_symbol: str, window: int = None,
                                  maturity_days: int = None) -> Optional[float]:
        """
        Calcule la volatilite historique annualisee.
        
        La fenetre est automatiquement matchee a la maturite de l'option :
            Maturite < 14j  -> fenetre 10j
            Maturite 14-45j -> fenetre 20j
            Maturite 45-90j -> fenetre 60j
            Maturite 90-180j -> fenetre 120j
            Maturite > 180j -> fenetre 252j

        Args:
            ticker_symbol: Symbole
            window: Fenetre en jours ouvres (override manuel)
            maturity_days: Jours a maturite de l'option (pour auto-matching)

        Returns:
            float: Volatilite annualisee (decimal), ou None
        """
        # Auto-matching fenetre HV <-> maturite
        if window is None:
            if maturity_days is None:
                window = 30  # defaut
            elif maturity_days < 14:
                window = 10
            elif maturity_days < 45:
                window = 20
            elif maturity_days < 90:
                window = 60
            elif maturity_days < 180:
                window = 120
            else:
                window = 252

        # Choisir la periode d'historique en fonction de la fenetre
        if window <= 30:
            period = "3mo"
        elif window <= 120:
            period = "6mo"
        else:
            period = "2y"

        try:
            hist = DataConnector.get_historical_prices(ticker_symbol, period=period)

            if len(hist) < window + 1:
                # Pas assez de donnees, essayer avec plus d'historique
                hist = DataConnector.get_historical_prices(ticker_symbol, period="2y")
                if len(hist) < window + 1:
                    return None

            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            vol = float(returns.tail(window).std() * np.sqrt(252))

            logger.info(
                f"HV {ticker_symbol}: window={window}d "
                f"(maturity={maturity_days}d) -> {vol:.1%}"
            )

            return vol

        except Exception as e:
            logger.error(f"Erreur calcul vol pour {ticker_symbol}: {e}")
            return None

    # =========================================================================
    # Utilitaires
    # =========================================================================

    @staticmethod
    def test_connection() -> dict:
        """
        Teste la connexion Yahoo Finance.

        Returns:
            dict avec status et details
        """
        try:
            price = DataConnector.get_spot_price("AAPL")

            return {
                'status': 'OK',
                'message': f'Yahoo Finance OK ! AAPL = ${price:.2f}',
                'test_price': price,
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Erreur Yahoo Finance : {str(e)}',
                'error': str(e),
            }
