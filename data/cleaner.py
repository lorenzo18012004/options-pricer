import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from config.options import (
    MAX_SPREAD_PCT,
    MAX_SPREAD_PCT_RELAXED,
    MIN_MID_PRICE,
    MIN_OPEN_INTEREST,
    MIN_VOLUME,
    MONEYNESS_MIN,
    MONEYNESS_MAX,
    MONEYNESS_SURFACE_MIN,
    MONEYNESS_SURFACE_MAX,
)


class DataCleaner:
    """
    Nettoyage et préparation des données de marché.
    
    Gestion des outliers, interpolation des données manquantes,
    et calcul de mid-prices pour les options.
    """
    
    @staticmethod
    def clean_option_chain(df, min_bid=0.01, max_spread_pct=None):
        """
        Nettoie une chaîne d'options en supprimant les données aberrantes.
        
        Args:
            df (DataFrame): Chaîne d'options brute
            min_bid (float): Bid minimum acceptable
            max_spread_pct (float): Spread bid-ask maximum en % du mid
        
        Returns:
            DataFrame: Chaîne nettoyée
        """
        if max_spread_pct is None:
            max_spread_pct = MAX_SPREAD_PCT_RELAXED
        df = df.copy()
        
        # Supprimer les lignes avec bid ou ask nuls ou négatifs
        df = df[(df['bid'] > min_bid) & (df['ask'] > min_bid)]
        
        # Calculer le mid price
        df['mid'] = (df['bid'] + df['ask']) / 2
        
        # Calculer le spread en %
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
        
        # Supprimer les spreads trop larges
        df = df[df['spread_pct'] <= max_spread_pct]
        
        # Supprimer les lignes avec volume = 0 et open interest = 0
        # fillna(0) pour éviter que NaN > 0 soit False et supprime des lignes valides
        if 'volume' in df.columns and 'openInterest' in df.columns:
            vol = df['volume'].fillna(0)
            oi = df['openInterest'].fillna(0)
            df = df[(vol > 0) | (oi > 0)]
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def interpolate_volatility_surface(strikes, ivs, target_strikes):
        """
        Interpole la surface de volatilité pour des strikes manquants.
        
        Utilise une interpolation cubique avec extrapolation flat aux extrémités.
        
        Args:
            strikes (array): Strikes disponibles
            ivs (array): Volatilités implicites correspondantes
            target_strikes (array): Strikes où on veut interpoler
        
        Returns:
            array: Volatilités implicites interpolées
        """
        if len(strikes) < 2:
            raise ValueError("Besoin d'au moins 2 points pour interpoler")
        
        # Tri par strike croissant
        sorted_idx = np.argsort(strikes)
        strikes_sorted = np.array(strikes)[sorted_idx]
        ivs_sorted = np.array(ivs)[sorted_idx]
        
        # Interpolation cubique avec fill_value pour extrapolation flat
        interpolator = interp1d(
            strikes_sorted, 
            ivs_sorted,
            kind='cubic',
            bounds_error=False,
            fill_value=(ivs_sorted[0], ivs_sorted[-1])  # Flat extrapolation
        )
        
        return interpolator(target_strikes)
    
    @staticmethod
    def detect_outliers_iqr(data, multiplier=1.5):
        """
        Détecte les outliers avec la méthode IQR (InterQuartile Range).
        
        Args:
            data (array): Données à analyser
            multiplier (float): Multiplicateur de l'IQR (1.5 standard)
        
        Returns:
            array (bool): Masque des outliers (True = outlier)
        """
        data = np.array(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    @staticmethod
    def smooth_volatility_smile(strikes, ivs, window=5):
        """
        Lisse un smile de volatilité avec une moyenne mobile.
        
        Utile pour réduire le bruit des données de marché.
        
        Args:
            strikes (array): Strikes
            ivs (array): Volatilités implicites
            window (int): Taille de la fenêtre de lissage
        
        Returns:
            tuple: (strikes, ivs_smoothed)
        """
        df = pd.DataFrame({'strike': strikes, 'iv': ivs})
        df = df.sort_values('strike')
        
        # Moyenne mobile sur les IVs
        df['iv_smoothed'] = df['iv'].rolling(window=window, center=True, min_periods=1).mean()
        
        return df['strike'].values, df['iv_smoothed'].values
    
    @staticmethod
    def calculate_moneyness(spot, strikes):
        """
        Calcule le moneyness (K/S) pour chaque strike.
        
        Utile pour normaliser les smiles de volatilité.
        
        Args:
            spot (float): Prix spot
            strikes (array): Liste des strikes
        
        Returns:
            array: Moneyness pour chaque strike
        """
        return np.array(strikes) / spot
    
    @staticmethod
    def filter_surface_quality(
        df: pd.DataFrame,
        min_oi: int = None,
        min_mid: float = None,
        max_spread_pct: float = None,
        min_volume: int = None,
    ) -> pd.DataFrame:
        """
        Filtre strict pour la surface IV : exclut options illiquides ou aberrantes.
        - Volume=0 ET Open Interest < min_oi (sauf si min_volume > 0)
        - Si min_volume > 0 : exige Volume > min_volume
        - Spread > mid * max_spread_pct (ex: 25%)
        - Mid price < min_mid (ex: 0.05$)
        """
        if min_oi is None:
            min_oi = MIN_OPEN_INTEREST
        if min_mid is None:
            min_mid = MIN_MID_PRICE
        if max_spread_pct is None:
            max_spread_pct = MAX_SPREAD_PCT
        if min_volume is None:
            min_volume = 0
        df = df.copy()
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        if "spread_pct" not in df.columns:
            df["spread_pct"] = np.where(
                df["mid"] > 0,
                (df["ask"] - df["bid"]) / df["mid"],
                np.nan
            )
        df = df[df["mid"] >= min_mid]
        df = df[df["spread_pct"].fillna(1).between(0, max_spread_pct)]
        if "volume" in df.columns and "openInterest" in df.columns:
            if min_volume > 0:
                df = df[df["volume"].fillna(0) > min_volume]
            else:
                df = df[(df["volume"] > 0) | (df["openInterest"] >= min_oi)]
        return df.reset_index(drop=True)

    @staticmethod
    def filter_by_moneyness(df, spot, min_moneyness=None, max_moneyness=None):
        """
        Filtre les options par moneyness pour se concentrer sur les options
        liquides (proche de ATM).
        
        Args:
            df (DataFrame): Chaîne d'options
            spot (float): Prix spot
            min_moneyness (float): Moneyness minimum (0.7 = 30% OTM put)
            max_moneyness (float): Moneyness maximum (1.3 = 30% OTM call)
        
        Returns:
            DataFrame: Chaîne filtrée
        """
        if min_moneyness is None:
            min_moneyness = MONEYNESS_MIN
        if max_moneyness is None:
            max_moneyness = MONEYNESS_MAX
        df = df.copy()
        df['moneyness'] = df['strike'] / spot
        df = df[(df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)]
        return df.reset_index(drop=True)
    
    @staticmethod
    def add_greeks_to_chain(df, spot, r, T, sigma_func=None, q=0.0):
        """
        Ajoute les colonnes de Grecs à une chaîne d'options.
        
        Args:
            df (DataFrame): Chaîne d'options avec colonnes 'strike', 'type'
            spot (float): Prix spot
            r (float): Taux sans risque
            T (float): Temps à maturité
            sigma_func (callable): Fonction qui retourne la vol pour un strike donné
                                   Si None, utilise 'impliedVolatility' de la df
            q (float): Dividend yield continu (decimal)
        
        Returns:
            DataFrame: Chaîne avec colonnes de Grecs ajoutées
        """
        from core.black_scholes import BlackScholes
        
        df = df.copy()
        
        greeks_list = []
        for _, row in df.iterrows():
            K = row['strike']
            option_type = row.get('type', 'call')
            
            # Déterminer la volatilité
            if sigma_func:
                sigma = sigma_func(K)
            elif 'impliedVolatility' in df.columns:
                sigma = row['impliedVolatility']
            else:
                raise ValueError("Unable to determine volatility")
            
            # Calculer tous les Grecs avec dividend yield
            greeks = BlackScholes.get_all_greeks(spot, K, T, r, sigma, option_type, q)
            greeks_list.append(greeks)
        
        # Ajouter les colonnes au DataFrame
        greeks_df = pd.DataFrame(greeks_list)
        return pd.concat([df, greeks_df], axis=1)
