import numpy as np
from core.curves import YieldCurve, InterestRateSwap

class VanillaSwap:
    """
    Interest Rate Swap Vanille : Échange de flux fixe contre flux flottant.
    
    Produit de base du marché des taux.
    Utilisé pour:
    - Gérer le risque de taux
    - Transformer dette fixe en flottante (ou vice-versa)
    - Spéculer sur l'évolution des taux
    """
    
    def __init__(self, notional, fixed_rate, payment_freq, maturity, 
                 curve, position="payer"):
        """
        Args:
            notional (float): Montant notionnel (en millions)
            fixed_rate (float): Taux fixe annualisé (ex: 0.03 pour 3%)
            payment_freq (float): Fréquence de paiement (1=annuel, 2=semestriel, 4=trimestriel)
            maturity (float): Maturité en années
            curve (YieldCurve): Courbe de taux pour l'actualisation
            position (str): "payer" (paye fixe, reçoit flottant) ou "receiver" (inverse)
        """
        self.swap = InterestRateSwap(notional, fixed_rate, payment_freq, maturity, curve)
        self.position = position.lower()
        
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.payment_freq = payment_freq
        self.maturity = maturity
        self.curve = curve
    
    def npv(self):
        """
        Net Present Value du swap.
        
        Pour un payer swap: NPV = PV(Floating) - PV(Fixed)
        Pour un receiver swap: NPV = PV(Fixed) - PV(Floating)
        
        Returns:
            float: NPV en dollars
        """
        swap_npv = self.swap.get_npv()
        
        if self.position == "payer":
            return swap_npv
        else:
            return -swap_npv
    
    def dv01(self):
        """
        DV01 : Sensibilité à un mouvement de 1bp de la courbe.
        
        Returns:
            float: DV01 en dollars
        """
        return self.swap.get_dv01()
    
    def par_rate(self):
        """
        Taux swap par : taux fixe qui rend le NPV = 0.
        
        Returns:
            float: Taux par
        """
        return self.swap.get_par_rate()
    
    def summary(self):
        """Résumé complet du swap."""
        return {
            'notional': f"${self.notional:,.0f}",
            'fixed_rate': f"{self.fixed_rate * 100:.2f}%",
            'maturity': f"{self.maturity} years",
            'payment_freq': f"{int(self.payment_freq)}x per year",
            'position': self.position.capitalize(),
            'npv': self.npv(),
            'dv01': self.dv01(),
            'par_rate': f"{self.par_rate() * 100:.2f}%"
        }
    
    def pnl_scenario(self, rate_shifts):
        """
        Calcule le P&L pour différents scénarios de shift de taux.
        
        Args:
            rate_shifts (list): Liste de shifts en bp (ex: [-50, -25, 0, 25, 50])
        
        Returns:
            dict: P&L pour chaque scénario
        """
        base_npv = self.npv()
        scenarios = {}
        
        for shift_bp in rate_shifts:
            shift = shift_bp / 10000  # Conversion bp -> décimal
            
            # Shifter la courbe
            shifted_rates = self.curve.rates + shift
            shifted_curve = YieldCurve(self.curve.maturities, shifted_rates)
            
            # Créer un nouveau swap avec la courbe shiftée
            shifted_swap = VanillaSwap(
                self.notional, self.fixed_rate, self.payment_freq,
                self.maturity, shifted_curve, self.position
            )
            
            shifted_npv = shifted_swap.npv()
            pnl = shifted_npv - base_npv
            
            scenarios[f"{shift_bp:+d}bp"] = {
                'npv': shifted_npv,
                'pnl': pnl
            }
        
        return scenarios


class SwapCurveBuilder:
    """
    Construction de la courbe de taux à partir des instruments de marché.
    
    Bootstrapping process:
    1. Dépôts court terme (1M, 3M, 6M)
    2. Futures (pour 1-2 ans)
    3. Swaps (pour 2-30 ans)
    """
    
    @staticmethod
    def build_from_market_data(deposits, futures, swaps):
        """
        Construit la courbe de taux complète.
        
        Args:
            deposits (dict): {maturity_years: rate}
            futures (dict): {maturity_years: rate}
            swaps (dict): {maturity_years: rate}
        
        Returns:
            YieldCurve: Courbe de taux bootstrappée
        """
        # Combiner tous les instruments
        all_instruments = {}
        all_instruments.update(deposits)
        all_instruments.update(futures)
        all_instruments.update(swaps)
        
        # Trier par maturité
        maturities = sorted(all_instruments.keys())
        rates = [all_instruments[mat] for mat in maturities]
        
        # Bootstrapping (par yields -> zero curve)
        curve = YieldCurve.bootstrap_from_par_yields(
            rates, maturities, payment_freq=2
        )
        
        return curve
    
    @staticmethod
    def create_sample_curve(base_rate=0.03, curve_type="flat"):
        """
        Crée une courbe de taux synthétique pour testing.
        
        Args:
            base_rate (float): Taux de base
            curve_type (str): "flat", "upward", "inverted", "humped"
        
        Returns:
            YieldCurve: Courbe synthétique
        """
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        if curve_type == "flat":
            rates = np.full_like(maturities, base_rate)
        
        elif curve_type == "upward":
            # Courbe croissante normale
            rates = base_rate + 0.01 * np.log(1 + maturities)
        
        elif curve_type == "inverted":
            # Courbe inversée (récession)
            rates = base_rate - 0.005 * np.log(1 + maturities)
        
        elif curve_type == "humped":
            # Courbe en bosse (court terme élevé)
            rates = base_rate + 0.02 * np.exp(-0.2 * maturities)
        
        else:
            raise ValueError(f"curve_type {curve_type} non reconnu")
        
        return YieldCurve(maturities, rates)


class SwapSpreadAnalyzer:
    """
    Analyse du Swap Spread : différence entre taux swap et taux treasury.
    
    Swap Spread = Taux Swap - Taux Treasury
    
    Indicateur de:
    - Risque de crédit bancaire
    - Liquidité du marché
    - Primes de risque
    """
    
    @staticmethod
    def calculate_spread(swap_curve, treasury_curve, maturity):
        """
        Calcule le swap spread pour une maturité donnée.
        
        Args:
            swap_curve (YieldCurve): Courbe de swaps
            treasury_curve (YieldCurve): Courbe de treasuries
            maturity (float): Maturité en années
        
        Returns:
            dict: Spread et composantes
        """
        swap_rate = swap_curve.get_zero_rate(maturity)
        treasury_rate = treasury_curve.get_zero_rate(maturity)
        
        spread = swap_rate - treasury_rate
        spread_bp = spread * 10000  # En basis points
        
        return {
            'maturity': maturity,
            'swap_rate': swap_rate,
            'treasury_rate': treasury_rate,
            'spread': spread,
            'spread_bp': spread_bp
        }
    
    @staticmethod
    def historical_interpretation(spread_bp):
        """
        Interprète le niveau du swap spread.
        
        Args:
            spread_bp (float): Spread en basis points
        
        Returns:
            str: Interprétation
        """
        if spread_bp < 10:
            return "Très serré - Marché très confiant, faible risque de crédit"
        elif spread_bp < 30:
            return "Normal - Conditions de marché saines"
        elif spread_bp < 60:
            return "Élargi - Inquiétudes sur le risque de crédit ou liquidité"
        else:
            return "Très large - Stress majeur sur le marché (crise potentielle)"


class BasisSwap:
    """
    Basis Swap : Échange de deux flux flottants sur des indices différents.
    
    Exemple: LIBOR 3M vs LIBOR 6M
    Ou: €STR vs EURIBOR
    
    Utilisé pour gérer le risque de base entre différents indices de taux.
    """
    
    def __init__(self, notional, spread, maturity, curve1, curve2, payment_freq=4):
        """
        Args:
            notional (float): Notionnel
            spread (float): Spread additionnel sur une jambe
            maturity (float): Maturité
            curve1 (YieldCurve): Courbe pour jambe 1
            curve2 (YieldCurve): Courbe pour jambe 2
            payment_freq (float): Fréquence de paiement
        """
        self.notional = notional
        self.spread = spread
        self.maturity = maturity
        self.curve1 = curve1
        self.curve2 = curve2
        self.payment_freq = payment_freq
        
        self.payment_dates = np.arange(1/payment_freq, maturity + 1/payment_freq, 1/payment_freq)
    
    def npv(self):
        """
        NPV du basis swap.
        
        Returns:
            float: NPV
        """
        pv_leg1 = 0.0
        pv_leg2 = 0.0
        
        for i, t in enumerate(self.payment_dates):
            # Leg 1
            if i == 0:
                t_prev = 0
            else:
                t_prev = self.payment_dates[i-1]
            
            forward_rate1 = self.curve1.get_forward_rate(t_prev, t)
            payment1 = self.notional * forward_rate1 * (t - t_prev)
            df1 = self.curve1.get_discount_factor(t)
            pv_leg1 += payment1 * df1
            
            # Leg 2 (avec spread)
            forward_rate2 = self.curve2.get_forward_rate(t_prev, t) + self.spread
            payment2 = self.notional * forward_rate2 * (t - t_prev)
            df2 = self.curve2.get_discount_factor(t)
            pv_leg2 += payment2 * df2
        
        return pv_leg2 - pv_leg1


class InflationSwap:
    """
    Inflation Swap : Échange de flux fixe contre inflation.
    
    Utilisé pour se couvrir contre le risque d'inflation.
    """
    
    def __init__(self, notional, fixed_rate, maturity, expected_inflation_curve):
        """
        Args:
            notional (float): Notionnel
            fixed_rate (float): Taux fixe (breakeven inflation)
            maturity (float): Maturité
            expected_inflation_curve (YieldCurve): Courbe d'inflation attendue
        """
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.inflation_curve = expected_inflation_curve
    
    def npv(self, discount_curve):
        """
        NPV du swap d'inflation.
        
        Args:
            discount_curve (YieldCurve): Courbe pour actualisation
        
        Returns:
            float: NPV
        """
        # Jambe fixe
        pv_fixed = self.notional * self.fixed_rate * self.maturity * discount_curve.get_discount_factor(self.maturity)
        
        # Jambe inflation (simplifié)
        expected_inflation = self.inflation_curve.get_zero_rate(self.maturity)
        pv_inflation = self.notional * expected_inflation * self.maturity * discount_curve.get_discount_factor(self.maturity)
        
        return pv_inflation - pv_fixed
