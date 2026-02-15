import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import CubicSpline
import warnings

class VolatilitySurface:
    """
    Surface de Volatilité Implicite.
    
    Utilisée pour calibrer et interpoler la volatilité implicite
    en fonction du strike et de la maturité.
    
    Implémente le modèle SVI (Stochastic Volatility Inspired)
    et des méthodes d'interpolation par splines.
    """
    
    def __init__(self, strikes, maturities, implied_vols):
        """
        Args:
            strikes (array): Array des strikes
            maturities (array): Array des maturités (en années)
            implied_vols (array): Array des volatilités implicites
        """
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.implied_vols = np.array(implied_vols)
        
        self.calibrated = False
    
    def calibrate_svi(self, strike_slice, iv_slice, spot):
        """
        Calibre le modèle SVI pour une maturité donnée.
        
        SVI parameterization (raw):
        σ²(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
        
        où k = log(K/F) est le log-moneyness.
        
        Args:
            strike_slice (array): Strikes pour cette maturité
            iv_slice (array): IVs pour cette maturité
            spot (float): Prix spot (pour calculer le moneyness)
        
        Returns:
            dict: Paramètres SVI calibrés
        """
        # Conversion en log-moneyness
        k = np.log(strike_slice / spot)
        variance = iv_slice ** 2
        
        # Fonction SVI
        def svi_func(k_val, a, b, rho, m, sigma):
            """SVI raw parameterization."""
            return a + b * (rho * (k_val - m) + np.sqrt((k_val - m)**2 + sigma**2))
        
        # Initialisation des paramètres
        # a ≈ variance ATM, b > 0, -1 < rho < 1, m ≈ 0, sigma > 0
        initial_guess = [
            np.mean(variance),  # a
            0.1,                # b
            0.0,                # rho
            0.0,                # m
            0.1                 # sigma
        ]
        
        # Bounds pour garantir l'arbitrage-free
        bounds = (
            [0, 0, -0.999, -1, 0.001],  # lower bounds
            [np.inf, np.inf, 0.999, 1, np.inf]  # upper bounds
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, _ = curve_fit(
                    svi_func, k, variance,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000
                )
            
            a, b, rho, m, sigma = params
            
            return {
                'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma,
                'k': k, 'fitted_variance': svi_func(k, *params)
            }
        
        except Exception as e:
            print(f"Calibration SVI échouée: {e}")
            return None
    
    def get_iv_from_svi(self, strike, spot, svi_params):
        """
        Retourne la volatilité implicite pour un strike donné avec SVI.
        
        Args:
            strike (float): Strike
            spot (float): Prix spot
            svi_params (dict): Paramètres SVI calibrés
        
        Returns:
            float: Volatilité implicite
        """
        k = np.log(strike / spot)
        a = svi_params['a']
        b = svi_params['b']
        rho = svi_params['rho']
        m = svi_params['m']
        sigma = svi_params['sigma']
        
        variance = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        # Garantir que variance > 0
        variance = max(variance, 1e-8)
        
        return np.sqrt(variance)
    
    def fit_spline(self, strike_slice, iv_slice):
        """
        Interpolation par Cubic Spline (alternative à SVI).
        
        Plus simple mais moins contraint (peut créer des arbitrages).
        
        Args:
            strike_slice (array): Strikes
            iv_slice (array): IVs
        
        Returns:
            CubicSpline: Fonction d'interpolation
        """
        # Tri par strike croissant
        sorted_idx = np.argsort(strike_slice)
        strikes_sorted = strike_slice[sorted_idx]
        ivs_sorted = iv_slice[sorted_idx]
        
        spline = CubicSpline(strikes_sorted, ivs_sorted, extrapolate=False)
        
        return spline
    
    def detect_arbitrage(self, strikes_test, ivs_test, spot, T, r):
        """
        Détecte les opportunités d'arbitrage dans le smile.
        
        Conditions d'absence d'arbitrage:
        1. Butterfly spread condition (convexité du call price)
        2. Calendar spread condition (monotonie en temps)
        
        Args:
            strikes_test (array): Strikes à tester
            ivs_test (array): IVs correspondantes
            spot (float): Prix spot
            T (float): Maturité
            r (float): Taux sans risque
        
        Returns:
            dict: Résultats de la détection
        """
        from core.black_scholes import BlackScholes
        
        # Calcul des prix d'options
        prices = [BlackScholes.get_price(spot, K, T, r, iv, "call") 
                  for K, iv in zip(strikes_test, ivs_test)]
        
        # Vérification de la convexité (butterfly spread)
        arbitrage_detected = False
        violations = []
        
        for i in range(1, len(strikes_test) - 1):
            K1, K2, K3 = strikes_test[i-1], strikes_test[i], strikes_test[i+1]
            C1, C2, C3 = prices[i-1], prices[i], prices[i+1]
            
            # Butterfly condition: (C1 - C2) / (K2 - K1) >= (C2 - C3) / (K3 - K2)
            slope1 = (C1 - C2) / (K2 - K1) if (K2 - K1) > 0 else 0
            slope2 = (C2 - C3) / (K3 - K2) if (K3 - K2) > 0 else 0
            
            if slope1 < slope2:
                arbitrage_detected = True
                violations.append({
                    'type': 'butterfly',
                    'strikes': (K1, K2, K3),
                    'slope1': slope1,
                    'slope2': slope2
                })
        
        return {
            'arbitrage': arbitrage_detected,
            'violations': violations
        }
    
    def plot_smile(self, spot, maturity_idx=0):
        """
        Génère les données pour plotter le volatility smile.
        
        Args:
            spot (float): Prix spot
            maturity_idx (int): Index de la maturité à plotter
        
        Returns:
            dict: {'strikes': array, 'ivs': array, 'moneyness': array}
        """
        # Filtrer pour une maturité spécifique
        mask = self.maturities == np.unique(self.maturities)[maturity_idx]
        strikes_slice = self.strikes[mask]
        ivs_slice = self.implied_vols[mask]
        
        # Calculer le moneyness
        moneyness = strikes_slice / spot
        
        # Tri par strike
        sorted_idx = np.argsort(strikes_slice)
        
        return {
            'strikes': strikes_slice[sorted_idx],
            'ivs': ivs_slice[sorted_idx] * 100,  # En pourcentage
            'moneyness': moneyness[sorted_idx]
        }


class VolatilitySkew:
    """
    Analyse du Skew de Volatilité pour les indices.
    
    Le skew capture la "fear of crash" : les puts OTM sont plus chers
    que les calls OTM sur les indices equity.
    """
    
    @staticmethod
    def calculate_skew(atm_iv, otm_put_25delta_iv, otm_call_25delta_iv):
        """
        Calcule le Risk Reversal (mesure du skew).
        
        RR = IV(25Δ Call) - IV(25Δ Put)
        
        Négatif pour les indices equity (put premium).
        
        Args:
            atm_iv (float): IV ATM
            otm_put_25delta_iv (float): IV du put 25-delta
            otm_call_25delta_iv (float): IV du call 25-delta
        
        Returns:
            dict: Métriques de skew
        """
        risk_reversal = otm_call_25delta_iv - otm_put_25delta_iv
        
        # Butterfly (mesure de la convexité)
        butterfly = (otm_put_25delta_iv + otm_call_25delta_iv) / 2 - atm_iv
        
        return {
            'risk_reversal': risk_reversal,
            'butterfly': butterfly,
            'atm_iv': atm_iv
        }
    
    @staticmethod
    def explain_skew(strike, spot, iv, atm_iv):
        """
        Explique la contribution du skew au prix d'une option.
        
        Args:
            strike (float): Strike de l'option
            spot (float): Prix spot
            iv (float): IV de l'option
            atm_iv (float): IV ATM
        
        Returns:
            dict: Décomposition du prix
        """
        from core.black_scholes import BlackScholes
        
        # Prix avec la IV réelle (incluant le skew)
        price_with_skew = BlackScholes.get_price(spot, strike, 1.0, 0.05, iv, "put")
        
        # Prix si on utilisait l'IV ATM (sans skew)
        price_flat_vol = BlackScholes.get_price(spot, strike, 1.0, 0.05, atm_iv, "put")
        
        skew_premium = price_with_skew - price_flat_vol
        
        return {
            'price_with_skew': price_with_skew,
            'price_flat_vol': price_flat_vol,
            'skew_premium': skew_premium,
            'skew_premium_pct': (skew_premium / price_flat_vol) * 100 if price_flat_vol > 0 else 0
        }
