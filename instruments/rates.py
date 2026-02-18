import numpy as np
from core.curves import (
    YieldCurve,
    InterestRateSwap,
    DAY_COUNT_30_360,
    DAY_COUNT_ACT_365,
    DAY_COUNT_ACT_360,
)

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
                 curve, position="payer", day_count: str = DAY_COUNT_30_360):
        """
        Args:
            notional (float): Montant notionnel (en millions)
            fixed_rate (float): Taux fixe annualisé (ex: 0.03 pour 3%)
            payment_freq (float): Fréquence de paiement (1=annuel, 2=semestriel, 4=trimestriel)
            maturity (float): Maturité en années
            curve (YieldCurve): Courbe de taux pour l'actualisation
            position (str): "payer" (paye fixe, reçoit flottant) ou "receiver" (inverse)
            day_count (str): Convention day count ("30/360", "ACT/365", "ACT/360")
        """
        self.swap = InterestRateSwap(
            notional, fixed_rate, payment_freq, maturity, curve, day_count=day_count
        )
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

    Pour une courbe Treasury uniquement, passer deposits={}, futures={}
    et swaps={maturity_years: par_yield} (ex: {0.25: 0.04, 5.0: 0.05, 10.0: 0.05}).
    """

    @staticmethod
    def build_from_market_data(deposits, futures, swaps):
        """
        Construit la courbe de taux complète par bootstrap des par yields.

        Les trois dicts sont fusionnés : {maturity_years: par_yield}.
        Passer {} pour deposits/futures si on n'a que des swaps/Treasury.

        Args:
            deposits (dict): {maturity_years: rate} - peut être {}
            futures (dict): {maturity_years: rate} - peut être {}
            swaps (dict): {maturity_years: rate} - par yields (Treasury, swap rates)

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
