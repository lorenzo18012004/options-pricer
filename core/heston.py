"""
Modele de Heston (1993) - Volatilite Stochastique

Le modele de Heston suppose que la volatilite n'est PAS constante
mais suit son propre processus stochastique :

    dS = (r - q) S dt + sqrt(v) S dW_1
    dv = kappa (theta - v) dt + xi sqrt(v) dW_2
    Corr(dW_1, dW_2) = rho

Parametres :
    v0    : variance instantanee initiale (sigma^2 actuelle)
    kappa : vitesse de retour a la moyenne de la variance
    theta : variance de long terme (niveau d'equilibre)
    xi    : volatilite de la volatilite (vol-of-vol)
    rho   : correlation spot-vol (typiquement negative pour equities)

Avantages vs BSM :
    - Genere naturellement le smile/skew de volatilite
    - Capture la correlation negative spot-vol (leverage effect)
    - Capture le vol clustering (mean-reversion)
    - Solution semi-analytique (rapide)

Pricing via la methode de la fonction caracteristique (Fourier inversion).
Reference : Heston (1993), Gatheral (2006), Albrecher et al.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class HestonModel:
    """
    Modele de Heston pour le pricing d'options europeennes
    avec volatilite stochastique.
    """

    def __init__(self, v0, kappa, theta, xi, rho):
        """
        Args:
            v0:    Variance instantanee initiale (ex: 0.04 = vol 20%)
            kappa: Vitesse de mean-reversion (ex: 2.0)
            theta: Variance de long terme (ex: 0.04 = vol 20%)
            xi:    Vol-of-vol (ex: 0.3)
            rho:   Correlation spot-vol (ex: -0.7)
        """
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def _characteristic_function(self, phi, S, K, T, r, q):
        """
        Fonction caracteristique du log-prix sous Heston.
        Utilise la formulation de Gatheral pour la stabilite numerique.

        Args:
            phi: Variable de Fourier
            S: Spot
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield

        Returns:
            complex: Valeur de la fonction caracteristique
        """
        v0, kappa, theta, xi, rho = self.v0, self.kappa, self.theta, self.xi, self.rho

        # Gatheral formulation (plus stable)
        alpha = -0.5 * phi * (phi + 1j)
        beta = kappa - rho * xi * 1j * phi
        gamma = 0.5 * xi ** 2

        d = np.sqrt(beta ** 2 - 4 * alpha * gamma)

        # Choisir le signe pour la stabilite (Albrecher & al.)
        r_minus = (beta - d) / (2 * gamma)
        r_plus = (beta + d) / (2 * gamma)

        g = r_minus / r_plus

        # Eviter les instabilites numeriques
        exp_dT = np.exp(-d * T)

        C = kappa * (r_minus * T - (2 / xi ** 2) * np.log((1 - g * exp_dT) / (1 - g)))
        D = r_minus * (1 - exp_dT) / (1 - g * exp_dT)

        # Log spot forward
        x = np.log(S / K) + (r - q) * T

        return np.exp(C * theta + D * v0 + 1j * phi * x)

    def get_price(self, S, K, T, r, option_type="call", q=0.0):
        """
        Prix d'une option europeenne sous le modele de Heston.
        Methode : Inversion de Fourier (Carr-Madan / Lewis).

        Args:
            S: Spot price
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            option_type: "call" ou "put"
            q: Dividend yield

        Returns:
            float: Prix de l'option
        """
        if T <= 0:
            if option_type.lower() == "call":
                return max(S - K, 0)
            return max(K - S, 0)

        def integrand(phi):
            """Integrande pour la formule de Lewis."""
            cf = self._characteristic_function(phi - 0.5j, S, K, T, r, q)
            return np.real(np.exp(-1j * phi * np.log(K / S)) * cf / (phi ** 2 + 0.25))

        # Integration numerique
        try:
            integral, _ = quad(integrand, 0, 100, limit=200)
        except (ValueError, RuntimeError):
            integral, _ = quad(integrand, 0, 50, limit=100)

        # Prix du call via Lewis (2000)
        call_price = S * np.exp(-q * T) - (
            np.sqrt(S * K) * np.exp(-(r + q) * T / 2) / np.pi * integral
        )
        if (not np.isfinite(call_price)) or np.iscomplexobj(call_price):
            raise ValueError("Heston numerical integration produced invalid call price.")

        # Enforce no-arbitrage bounds instead of blind clipping to zero.
        call_lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        call_upper = S * np.exp(-q * T)
        call_price = float(np.clip(np.real(call_price), call_lower, call_upper))

        if option_type.lower() == "call":
            return call_price
        else:
            # Put-Call Parity
            return call_price - S * np.exp(-q * T) + K * np.exp(-r * T)

    def get_implied_vol(self, S, K, T, r, option_type="call", q=0.0):
        """
        Calcule l'IV BSM equivalente du prix Heston.
        Utile pour comparer avec le marche.
        """
        from core.solvers import IVsolver

        price = self.get_price(S, K, T, r, option_type, q)
        if price <= 0:
            return None

        return IVsolver.find_implied_vol(price, S, K, T, r, option_type, q)

    def get_greeks(self, S, K, T, r, option_type="call", q=0.0, bump=0.01):
        """
        Greeks par différences finies (bump & reprice).
        """
        price = self.get_price(S, K, T, r, option_type, q)

        # Delta
        price_up = self.get_price(S * (1 + bump), K, T, r, option_type, q)
        price_down = self.get_price(S * (1 - bump), K, T, r, option_type, q)
        delta = (price_up - price_down) / (2 * S * bump)

        # Gamma
        gamma = (price_up - 2 * price + price_down) / (S * bump) ** 2

        # Vega (bump v0)
        model_up = HestonModel(self.v0 * 1.01, self.kappa, self.theta, self.xi, self.rho)
        model_down = HestonModel(self.v0 * 0.99, self.kappa, self.theta, self.xi, self.rho)
        vega = (model_up.get_price(S, K, T, r, option_type, q) -
                model_down.get_price(S, K, T, r, option_type, q)) / (0.02 * np.sqrt(self.v0)) / 100

        # Theta
        if T > 1 / 365:
            price_tm1 = self.get_price(S, K, T - 1 / 365, r, option_type, q)
            theta = (price_tm1 - price)
        else:
            theta = 0

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
        }

    @staticmethod
    def calibrate(
        market_strikes,
        market_ivs,
        S,
        T,
        r,
        q=0.0,
        option_type="call",
        initial_guess=None,
        spreads=None,
        open_interests=None,
        moneyness_range=(0.85, 1.15),
        max_points=25,
    ):
        """
        Calibration des 5 paramètres Heston.

        Pipeline:
        - Filtrage des points (validite IV + moneyness + max_points)
        - Pondération par vega et liquidite (spread/OI si fournis)
        - Optimisation globale puis locale
        - Diagnostics de qualite (RMSE, bords, Feller, points utilises)

        Args:
            market_strikes: Array de strikes
            market_ivs: Array d'IV du marche (decimal)
            S: Spot
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            option_type: "call" ou "put"
            initial_guess: dict optionnel {v0, kappa, theta, xi, rho}
            spreads: array optionnel de spread bid-ask relatifs (decimal)
            open_interests: array optionnel d'open interest
            moneyness_range: tuple (min, max) filtre de moneyness
            max_points: max points gardes (proche ATM en priorite)

        Returns:
            HestonModel: Modele calibre
            dict: Details de la calibration
        """
        from core.solvers import IVsolver
        from core.black_scholes import BlackScholes

        market_strikes = np.asarray(market_strikes, dtype=float)
        market_ivs = np.asarray(market_ivs, dtype=float)
        spreads = np.asarray(spreads, dtype=float) if spreads is not None else None
        open_interests = np.asarray(open_interests, dtype=float) if open_interests is not None else None

        if len(market_strikes) != len(market_ivs):
            raise ValueError("market_strikes et market_ivs doivent avoir la meme longueur")

        # 1) Filtrage des points
        valid_iv = np.isfinite(market_ivs) & (market_ivs > 0.01) & (market_ivs < 3.0)
        mny = market_strikes / S
        valid_mny = (mny >= moneyness_range[0]) & (mny <= moneyness_range[1])
        mask = valid_iv & valid_mny

        strikes_f = market_strikes[mask]
        ivs_f = market_ivs[mask]

        spreads_f = spreads[mask] if spreads is not None and len(spreads) == len(mask) else None
        oi_f = open_interests[mask] if open_interests is not None and len(open_interests) == len(mask) else None

        # Conserver en priorite les strikes proches ATM
        if len(strikes_f) > max_points:
            order = np.argsort(np.abs(np.log(strikes_f / S)))
            keep = order[:max_points]
            strikes_f = strikes_f[keep]
            ivs_f = ivs_f[keep]
            if spreads_f is not None:
                spreads_f = spreads_f[keep]
            if oi_f is not None:
                oi_f = oi_f[keep]

        if len(strikes_f) < 5:
            raise ValueError("Pas assez de points liquides pour calibrer Heston")

        # Prix de marche derives des IV (pour stabilite numerique de l'objectif)
        market_prices_f = np.array([
            BlackScholes.get_price(S, K, T, r, iv, option_type, q)
            for K, iv in zip(strikes_f, ivs_f)
        ], dtype=float)

        # 2) Poids desk-style = Vega weight * liquidity weight
        vegas_raw = np.array([
            max(1e-8, BlackScholes.get_vega(S, K, T, r, iv, q=q) * 100.0)
            for K, iv in zip(strikes_f, ivs_f)
        ], dtype=float)
        vega_w = vegas_raw / np.mean(vegas_raw)

        liq_w = np.ones_like(vega_w)
        if spreads_f is not None:
            spreads_clip = np.clip(spreads_f, 0.001, 2.0)
            liq_w *= 1.0 / (1.0 + 10.0 * spreads_clip)
        if oi_f is not None:
            oi_clip = np.clip(oi_f, 0.0, None)
            oi_norm = np.log1p(oi_clip)
            if np.max(oi_norm) > 0:
                liq_w *= 0.5 + 0.5 * (oi_norm / np.max(oi_norm))

        weights = vega_w * liq_w
        weights = np.clip(weights, 1e-4, None)
        weights /= np.mean(weights)

        # Guess initial intelligent base sur les donnees
        if initial_guess is None:
            atm_iv = float(np.interp(S, strikes_f, ivs_f))
            # Estimer rho depuis le skew (pente du smile)
            if len(strikes_f) > 2:
                skew = (ivs_f[-1] - ivs_f[0]) / (strikes_f[-1] - strikes_f[0])
                rho_guess = max(-0.95, min(-0.1, skew * S * 10))
            else:
                rho_guess = -0.7
            # Estimer xi depuis la courbure du smile
            xi_guess = max(0.1, min(1.2, np.std(ivs_f) * 4))

            initial_guess = {
                'v0': atm_iv ** 2,
                'kappa': 1.5,
                'theta': atm_iv ** 2,
                'xi': xi_guess,
                'rho': rho_guess
            }

        x0 = [initial_guess['v0'], initial_guess['kappa'],
              initial_guess['theta'], initial_guess['xi'],
              initial_guess['rho']]

        # Bornes (contraintes physiques)
        bounds = [
            (0.0005, 1.5),   # v0
            (0.05, 8.0),     # kappa
            (0.0005, 1.5),   # theta
            (0.05, 1.5),     # xi
            (-0.999, 0.0),   # rho (negatif pour equities)
        ]

        def objective(params):
            """Erreur quadratique pondérée + pénalités Feller."""
            v0, kappa, theta, xi, rho = params

            model = HestonModel(v0, kappa, theta, xi, rho)

            total_error = 0.0
            for i, (K, mkt_price, w) in enumerate(zip(strikes_f, market_prices_f, weights)):
                try:
                    model_price = model.get_price(S, K, T, r, option_type, q)
                    rel_err = (model_price - mkt_price) / max(1e-6, mkt_price)
                    total_error += w * (rel_err ** 2)
                except (ValueError, RuntimeError):
                    total_error += 50.0

            # Penalite si Feller violee
            feller_margin = 2 * kappa * theta - xi ** 2
            if feller_margin < 0:
                total_error += 400.0 * (feller_margin ** 2)

            # Penalite douce pour eviter parameters sur bornes
            for (lo, hi), p in zip(bounds, params):
                dist = min(abs(p - lo), abs(hi - p))
                if dist < 0.02 * (hi - lo):
                    total_error += 0.05

            return total_error

        # Optimisation : differential evolution (globale) puis L-BFGS-B (locale)
        from scipy.optimize import differential_evolution

        try:
            result_global = differential_evolution(
                objective, bounds, maxiter=120, tol=1e-6, popsize=12,
                seed=42, polish=False
            )
            x0 = result_global.x
        except (ValueError, RuntimeError):
            pass  # Garder le x0 initial si DE echoue

        result = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-10}
        )

        v0_cal, kappa_cal, theta_cal, xi_cal, rho_cal = result.x
        calibrated_model = HestonModel(v0_cal, kappa_cal, theta_cal, xi_cal, rho_cal)

        # Calculer les IV du modele calibre pour comparaison
        model_ivs = []
        for K in market_strikes:
            iv = calibrated_model.get_implied_vol(S, K, T, r, option_type, q)
            model_ivs.append(iv if iv else 0)

        rmse = np.sqrt(np.mean((np.array(model_ivs) - market_ivs) ** 2)) * 100
        rmse_filtered = np.sqrt(np.mean((np.array([
            calibrated_model.get_implied_vol(S, K, T, r, option_type, q) or 0.0
            for K in strikes_f
        ]) - ivs_f) ** 2)) * 100

        # Feller condition
        feller = 2 * kappa_cal * theta_cal > xi_cal ** 2

        boundary_hits = int(
            (abs(v0_cal - bounds[0][0]) < 1e-6) or (abs(v0_cal - bounds[0][1]) < 1e-6)
        ) + int(
            (abs(kappa_cal - bounds[1][0]) < 1e-6) or (abs(kappa_cal - bounds[1][1]) < 1e-6)
        ) + int(
            (abs(theta_cal - bounds[2][0]) < 1e-6) or (abs(theta_cal - bounds[2][1]) < 1e-6)
        ) + int(
            (abs(xi_cal - bounds[3][0]) < 1e-6) or (abs(xi_cal - bounds[3][1]) < 1e-6)
        ) + int(
            (abs(rho_cal - bounds[4][0]) < 1e-6) or (abs(rho_cal - bounds[4][1]) < 1e-6)
        )

        calibration_info = {
            'v0': v0_cal,
            'kappa': kappa_cal,
            'theta': theta_cal,
            'xi': xi_cal,
            'rho': rho_cal,
            'long_term_vol': np.sqrt(theta_cal) * 100,
            'current_vol': np.sqrt(v0_cal) * 100,
            'feller_satisfied': feller,
            'rmse_iv_pct': rmse,
            'rmse_iv_filtered_pct': rmse_filtered,
            'model_ivs': model_ivs,
            'success': result.success,
            'message': result.message,
            'n_points_total': int(len(market_strikes)),
            'n_points_used': int(len(strikes_f)),
            'moneyness_range_used': moneyness_range,
            'weights_mean': float(np.mean(weights)),
            'weights_std': float(np.std(weights)),
            'boundary_hits': boundary_hits,
        }

        logger.info(
            f"Heston calibre: v0={v0_cal:.4f}, kappa={kappa_cal:.2f}, "
            f"theta={theta_cal:.4f}, xi={xi_cal:.3f}, rho={rho_cal:.3f}, "
            f"RMSE={rmse:.3f}% (filtered={rmse_filtered:.3f}%), "
            f"points={len(strikes_f)}/{len(market_strikes)}"
        )

        return calibrated_model, calibration_info
