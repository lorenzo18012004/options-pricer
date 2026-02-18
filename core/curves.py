import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline

class YieldCurve:
    """
    Courbe de Taux Zero-Coupon construite par Bootstrapping.
    
    Utilisée pour pricer les produits de taux (IRS, Bonds, Caps/Floors).
    La méthode de bootstrapping permet de construire la courbe des taux
    à partir des instruments de marché (dépôts, futures, swaps).
    """
    
    def __init__(self, maturities, rates):
        """
        Args:
            maturities (list): Liste des maturités en années [0.25, 0.5, 1, 2, 5, 10, 30]
            rates (list): Taux Zero-Coupon correspondants (annualisés)
        """
        self.maturities = np.array(maturities, dtype=float)
        self.rates = np.array(rates, dtype=float)
        if self.maturities.size == 0 or self.rates.size == 0:
            raise ValueError("maturities and rates must be non-empty")
        if self.maturities.size != self.rates.size:
            raise ValueError("maturities and rates must have same length")
        if not np.all(np.isfinite(self.maturities)) or not np.all(np.isfinite(self.rates)):
            raise ValueError("maturities and rates must be finite")
        if np.any(self.maturities <= 0):
            raise ValueError("all maturities must be > 0")
        if np.any(np.diff(self.maturities) <= 0):
            raise ValueError("maturities must be strictly increasing")
        
        # Interpolation Cubic Spline pour les maturités intermédiaires
        self.curve = CubicSpline(self.maturities, self.rates)
    
    def get_zero_rate(self, maturity):
        """
        Retourne le taux zero-coupon pour une maturité donnée.
        
        Args:
            maturity (float): Maturité en années
        
        Returns:
            float: Taux zero-coupon interpolé
        """
        if maturity <= 0:
            return 0.0
        
        # Si hors de la courbe, on utilise une extrapolation flat
        if maturity < self.maturities[0]:
            return self.rates[0]
        elif maturity > self.maturities[-1]:
            return self.rates[-1]
        
        return float(self.curve(maturity))
    
    def get_discount_factor(self, maturity):
        """
        Facteur d'actualisation pour une maturité donnée.
        DF(t) = exp(-r(t) * t)
        
        Args:
            maturity (float): Maturité en années
        
        Returns:
            float: Facteur d'actualisation
        """
        rate = self.get_zero_rate(maturity)
        return np.exp(-rate * maturity)
    
    def get_forward_rate(self, t1, t2):
        """
        Taux forward entre deux dates.
        
        Le taux forward f(t1, t2) est le taux à terme pour emprunter
        entre t1 et t2, implicite dans la courbe actuelle.
        
        Args:
            t1 (float): Date de début en années
            t2 (float): Date de fin en années
        
        Returns:
            float: Taux forward annualisé
        """
        if t1 >= t2:
            raise ValueError("t1 must be strictly less than t2")
        
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        
        forward_rate = -np.log(df2 / df1) / (t2 - t1)
        return forward_rate
    
    @staticmethod
    def bootstrap_from_par_yields(par_yields, maturities, payment_freq=2):
        """
        Bootstrap des zero rates a partir de par yields (obligations/swap rates).

        Hypotheses:
        - Coupons fixes de frequence payment_freq
        - Prix pair (=1) pour chaque maturite
        - Interpolation log-lineaire des DF pour les dates de coupon intermediaires

        Args:
            par_yields (list): Taux par annualises (decimaux)
            maturities (list): Maturites correspondantes en annees
            payment_freq (int): Nb paiements par an (2 = semi-annuel)

        Returns:
            YieldCurve: Courbe zero-coupon bootstrappee
        """
        mats = np.asarray(maturities, dtype=float)
        yields = np.asarray(par_yields, dtype=float)
        if len(mats) != len(yields):
            raise ValueError("maturities and par_yields must have the same length")
        if payment_freq <= 0:
            raise ValueError("payment_freq must be > 0")

        # Trier par maturite
        order = np.argsort(mats)
        mats = mats[order]
        yields = yields[order]

        known_t = []
        known_df = []

        def interp_df(t):
            """Interpolation log-lineaire des discount factors."""
            if len(known_t) == 0:
                return None
            if t <= known_t[0]:
                return float(known_df[0] ** (t / known_t[0])) if known_t[0] > 0 else 1.0
            if t >= known_t[-1]:
                z_last = -np.log(max(known_df[-1], 1e-12)) / known_t[-1]
                return float(np.exp(-z_last * t))

            # Encadrement
            for i in range(len(known_t) - 1):
                t0, t1 = known_t[i], known_t[i + 1]
                if t0 <= t <= t1:
                    z0 = -np.log(max(known_df[i], 1e-12)) / t0
                    z1 = -np.log(max(known_df[i + 1], 1e-12)) / t1
                    w = (t - t0) / max(t1 - t0, 1e-12)
                    z = (1.0 - w) * z0 + w * z1
                    return float(np.exp(-z * t))
            return float(np.exp(-yields[-1] * t))

        for T, c in zip(mats, yields):
            if T <= 0:
                continue

            dt = 1.0 / float(payment_freq)
            # Coupon schedule up to T with potentially stub last period.
            coupon_dates = list(np.arange(dt, T, dt))
            if len(coupon_dates) == 0 or abs(coupon_dates[-1] - T) > 1e-12:
                coupon_dates.append(float(T))
            else:
                coupon_dates[-1] = float(T)

            # Accrual fractions alpha_i
            prev_t = 0.0
            accruals = []
            for t_i in coupon_dates:
                alpha = max(float(t_i - prev_t), 1e-12)
                accruals.append(alpha)
                prev_t = float(t_i)

            # Price at par:
            # 1 = c * sum(alpha_i * DF(t_i)) + DF(T)
            # Unknown appears in the final DF(T), including final coupon alpha_n.
            sum_prev = 0.0
            for t_i, a_i in zip(coupon_dates[:-1], accruals[:-1]):
                df_i = interp_df(float(t_i))
                if df_i is None:
                    # First node fallback: continuous-rate proxy for missing prior data.
                    df_i = float(np.exp(-c * t_i))
                sum_prev += a_i * df_i

            alpha_last = accruals[-1]
            numerator = 1.0 - c * sum_prev
            denominator = 1.0 + c * alpha_last
            df_T = numerator / max(denominator, 1e-12)
            if (not np.isfinite(df_T)) or (df_T <= 0):
                df_T = float(np.exp(-c * T))

            df_T = float(np.clip(df_T, 1e-10, 1.0))
            known_t.append(float(T))
            known_df.append(df_T)

        zero_rates = np.array([
            (-np.log(df) / t) if t > 0 else 0.0
            for t, df in zip(known_t, known_df)
        ], dtype=float)

        return YieldCurve(np.array(known_t, dtype=float), zero_rates)

    @staticmethod
    def bootstrap_from_swaps(swap_rates, maturities):
        """
        Construit la courbe de taux par bootstrapping à partir de swap rates.
        
        Méthode classique utilisée par les desks de taux.
        
        Args:
            swap_rates (list): Taux des swaps de marché (annualisés)
            maturities (list): Maturités correspondantes en années
        
        Returns:
            YieldCurve: Courbe de taux bootstrappée
        """
        # Compatibilite legacy: route vers le bootstrap par par-yields
        return YieldCurve.bootstrap_from_par_yields(
            swap_rates, maturities, payment_freq=1
        )


# Conventions de day count pour les swaps
DAY_COUNT_30_360 = "30/360"
DAY_COUNT_ACT_365 = "ACT/365"
DAY_COUNT_ACT_360 = "ACT/360"


def _accrual_fraction(t_prev: float, t_i: float, day_count: str,
                      ref_date: datetime = None) -> float:
    """
    Fraction d'accrual pour une période [t_prev, t_i] en années.
    t_prev, t_i : fractions d'année (ex: 0.5, 1.0 pour semi-annuel)
    ref_date : date de référence pour ACT/365 et ACT/360 (vrais jours calendaires)
    """
    if day_count == DAY_COUNT_30_360:
        months = (t_i - t_prev) * 12.0
        return months * 30.0 / 360.0  # 30/360 US
    if day_count in (DAY_COUNT_ACT_365, DAY_COUNT_ACT_360) and ref_date is not None:
        date_prev = ref_date + timedelta(days=round(t_prev * 365.25))
        date_i = ref_date + timedelta(days=round(t_i * 365.25))
        actual_days = (date_i - date_prev).days
        if day_count == DAY_COUNT_ACT_365:
            return actual_days / 365.0
        return actual_days / 360.0
    days_approx = (t_i - t_prev) * 365.0
    if day_count == DAY_COUNT_ACT_365:
        return days_approx / 365.0
    if day_count == DAY_COUNT_ACT_360:
        return days_approx / 360.0
    return t_i - t_prev  # fallback: year fraction


class InterestRateSwap:
    """
    Interest Rate Swap (IRS) : Échange de flux fixe contre flux flottant.
    
    Produit vanille du marché des taux. Utilisé pour gérer le risque de taux.
    Supporte les conventions de day count : 30/360, ACT/365, ACT/360.
    """
    
    def __init__(self, notional, fixed_rate, payment_freq, maturity, curve,
                 day_count: str = DAY_COUNT_30_360):
        """
        Args:
            notional (float): Montant notionnel du swap
            fixed_rate (float): Taux fixe annualisé
            payment_freq (float): Fréquence de paiement (1 = annuel, 2 = semestriel, 4 = trimestriel)
            maturity (float): Maturité en années
            curve (YieldCurve): Courbe de taux pour l'actualisation
            day_count (str): Convention de day count ("30/360", "ACT/365", "ACT/360")
        """
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.payment_freq = payment_freq
        self.maturity = maturity
        self.curve = curve
        self.day_count = day_count or DAY_COUNT_30_360
        self.ref_date = datetime.now().date()
        
        # Génération de la schedule de paiements
        self.payment_dates = np.arange(1/payment_freq, maturity + 1/payment_freq, 1/payment_freq)
    
    def get_fixed_leg_pv(self):
        """
        Valeur Présente de la jambe fixe.
        Utilise la convention day_count pour les fractions d'accrual.
        
        Returns:
            float: PV des flux fixes
        """
        pv = 0.0
        t_prev = 0.0
        ref_dt = datetime.combine(self.ref_date, datetime.min.time())
        for t in self.payment_dates:
            alpha = _accrual_fraction(t_prev, float(t), self.day_count, ref_dt)
            payment = self.notional * self.fixed_rate * alpha
            df = self.curve.get_discount_factor(t)
            pv += payment * df
            t_prev = float(t)
        return pv
    
    def get_floating_leg_pv(self):
        """
        Valeur Présente de la jambe flottante.
        
        Pour un swap vanille, la PV de la jambe flottante à l'initiation
        est égale au notionnel * (1 - DF(T))
        
        Returns:
            float: PV des flux flottants
        """
        df_final = self.curve.get_discount_factor(self.maturity)
        return self.notional * (1 - df_final)
    
    def get_npv(self):
        """
        Net Present Value du swap (point de vue payeur du fixe).
        
        NPV = PV(Floating Leg) - PV(Fixed Leg)
        
        Returns:
            float: NPV du swap
        """
        return self.get_floating_leg_pv() - self.get_fixed_leg_pv()
    
    def get_dv01(self):
        """
        DV01 : Dollar Value of 1 basis point.
        
        Sensibilité de la valeur du swap à un mouvement de 1bp (0.01%) de la courbe.
        Utilisé par les traders pour géger le risque de taux.
        
        Returns:
            float: DV01 (en dollars)
        """
        # On shift la courbe de +1bp et on recalcule le NPV
        original_rates = self.curve.rates.copy()
        
        # Shift de +1bp
        shifted_rates = original_rates + 0.0001
        shifted_curve = YieldCurve(self.curve.maturities, shifted_rates)
        
        # Swap avec courbe shiftée
        shifted_swap = InterestRateSwap(
            self.notional, self.fixed_rate, self.payment_freq, 
            self.maturity, shifted_curve, day_count=self.day_count
        )
        
        # DV01 = (NPV_shifted - NPV_original)
        dv01 = shifted_swap.get_npv() - self.get_npv()
        
        return abs(dv01)
    
    def get_par_rate(self):
        """
        Taux swap par (le taux fixe qui rend le NPV = 0 à l'initiation).
        
        C'est le taux swap coté sur le marché.
        Utilise day_count pour les accruals.
        
        Returns:
            float: Taux swap par
        """
        df_final = self.curve.get_discount_factor(self.maturity)
        sum_weighted = 0.0
        t_prev = 0.0
        ref_dt = datetime.combine(self.ref_date, datetime.min.time())
        for t in self.payment_dates:
            alpha = _accrual_fraction(t_prev, float(t), self.day_count, ref_dt)
            sum_weighted += alpha * self.curve.get_discount_factor(t)
            t_prev = float(t)
        if sum_weighted <= 0:
            return 0.0
        par_rate = (1 - df_final) / sum_weighted
        return par_rate
