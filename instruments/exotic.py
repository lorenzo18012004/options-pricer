import numpy as np
from scipy.stats import norm
from models.monte_carlo import MonteCarloPricer
from core.black_scholes import BlackScholes


def _barrier_hit_probability(S: float, H: float, T: float, r: float, sigma: float, q: float, is_up: bool) -> float:
    """P(barrier touchée) : formule first-passage. is_up=True si H > S."""
    if T <= 0 or sigma <= 0 or S <= 0 or H <= 0:
        return 0.0
    mu = r - q - 0.5 * sigma ** 2
    sig_sqrt_t = sigma * np.sqrt(T)
    if is_up:  # H > S
        eta = (np.log(H / S) - mu * T) / sig_sqrt_t
        zeta = (np.log(H / S) + mu * T) / sig_sqrt_t
        lam = 2 * mu / (sigma ** 2)
        prob = norm.cdf(eta) + (H / S) ** lam * norm.cdf(zeta)
    else:  # down: H < S
        eta = (np.log(H / S) - mu * T) / sig_sqrt_t
        zeta = (np.log(H / S) + mu * T) / sig_sqrt_t
        lam = 2 * mu / (sigma ** 2)
        prob = norm.cdf(-eta) + (H / S) ** lam * norm.cdf(-zeta)
    return float(np.clip(prob, 0.0, 1.0))


class BarrierOption:
    """
    Option Barrière : Option qui s'active ou se désactive quand le prix
    atteint un certain niveau.
    
    Types:
    - Knock-Out: Option disparaît si barrière touchée
    - Knock-In: Option s'active si barrière touchée
    - Down/Up: Direction de la barrière
    
    Utilisées pour réduire le coût d'une option vanille classique.
    """
    
    def __init__(self, S, K, T, r, sigma, barrier_level, barrier_type="down-and-out",
                 option_type="call", rebate=0, q=0.0):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité
            barrier_level (float): Niveau de la barrière
            barrier_type (str): "down-and-out", "down-and-in", "up-and-out", "up-and-in"
            option_type (str): "call" ou "put"
            rebate (float): Montant payé si la barrière est touchée (pour knock-out)
            q (float): Dividend yield continu
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type.lower()
        self.option_type = option_type.lower()
        self.rebate = rebate
        self.q = q
        
        # Validation
        self._validate_barrier()
    
    def _validate_barrier(self):
        """Valide que la barrière est cohérente avec le type d'option."""
        if "down" in self.barrier_type and self.barrier_level >= self.S:
            raise ValueError("Down barrier must be below spot")
        
        if "up" in self.barrier_type and self.barrier_level <= self.S:
            raise ValueError("Up barrier must be above spot")
    
    def price(self, method="monte_carlo", n_simulations=50000, n_steps=252,
              use_antithetic=True, seed=42, brownian_bridge=True):
        """
        Prix de l'option barrière.
        
        Args:
            method (str): "monte_carlo" ou "analytical" (pour certains cas)
            n_simulations (int): Nombre de simulations MC
            n_steps (int): Pas de temps pour MC
            use_antithetic (bool): Réduction de variance
            seed (int): Seed de reproductibilité
            brownian_bridge (bool): Ajustement de franchissement intra-step
        
        Returns:
            dict: {'price': prix, 'std_error': erreur, 'activation_rate': taux}
        """
        if method == "analytical":
            return self._analytical_price()
        elif method == "monte_carlo":
            # Utiliser analytique si disponible (down/up-and-out/in)
            try:
                return self._analytical_price()
            except NotImplementedError:
                pass
            mc = MonteCarloPricer(
                self.S, self.K, self.T, self.r, self.sigma,
                n_simulations=n_simulations, n_steps=n_steps, seed=seed, q=self.q
            )
            
            result = mc.price_barrier(
                self.barrier_level,
                self.barrier_type,
                self.option_type,
                use_antithetic=use_antithetic,
                brownian_bridge=brownian_bridge,
            )
            
            # Ajouter le rebate si applicable
            if "out" in self.barrier_type and self.rebate > 0:
                rebate_pv = self.rebate * np.exp(-self.r * self.T) * result['activation_rate']
                result['price'] += rebate_pv
                result['rebate_value'] = rebate_pv
            
            return result
        
        else:
            raise ValueError(f"Method {method} not supported")
    
    def _analytical_price(self):
        """
        Prix analytique pour barrières (formules Rubinstein-Reiner / Haug).

        Formule down-and-out call (reflection principle):
            C_do = C(S,K) - (H/S)^λ * C(H²/S, K)
        avec λ = 2(r-q)/σ² - 1

        Down-and-in: C_di = (H/S)^λ * C(H²/S, K)  (parité C_do + C_di = C_vanille)
        """
        H = self.barrier_level
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q

        def _bs_call(s, k):
            return BlackScholes.get_price(s, k, T, r, sigma, "call", q)

        def _make_result(p, m='analytical_rubinstein_reiner'):
            hit_prob = _barrier_hit_probability(S, H, T, r, sigma, q, "up" in self.barrier_type)
            res = {'price': p, 'method': m, 'std_error': 0.0, 'activation_rate': hit_prob}
            if "out" in self.barrier_type and self.rebate > 0:
                res['price'] += self.rebate * np.exp(-r * T) * hit_prob
                res['rebate_value'] = self.rebate * np.exp(-r * T) * hit_prob
            return res

        # Down-and-out call : C_do = C(S,K) - (H/S)^λ * C(H²/S, K)
        if self.barrier_type == "down-and-out" and self.option_type == "call":
            # λ = 2(r-q)/σ² - 1 (convention Haug / reflection principle)
            lambda_val = 2 * (r - q) / (sigma ** 2) - 1
            vanilla = _bs_call(S, K)
            reflected_spot = (H ** 2) / S
            reflected_call = _bs_call(reflected_spot, K)
            price = max(0.0, vanilla - (H / S) ** lambda_val * reflected_call)
            return _make_result(price)

        # Down-and-in call : C_di = (H/S)^λ * C(H²/S, K)
        if self.barrier_type == "down-and-in" and self.option_type == "call":
            lambda_val = 2 * (r - q) / (sigma ** 2) - 1
            reflected_spot = (H ** 2) / S
            reflected_call = _bs_call(reflected_spot, K)
            price = (H / S) ** lambda_val * reflected_call
            return _make_result(price)

        # Down-and-out put : formule symétrique
        if self.barrier_type == "down-and-out" and self.option_type == "put":
            lambda_val = 2 * (r - q) / (sigma ** 2) - 1
            vanilla_put = BlackScholes.get_price(S, K, T, r, sigma, "put", q)
            reflected_spot = (H ** 2) / S
            reflected_put = BlackScholes.get_price(reflected_spot, K, T, r, sigma, "put", q)
            price = max(0.0, vanilla_put - (H / S) ** lambda_val * reflected_put)
            return _make_result(price)

        # Down-and-in put
        if self.barrier_type == "down-and-in" and self.option_type == "put":
            lambda_val = 2 * (r - q) / (sigma ** 2) - 1
            reflected_spot = (H ** 2) / S
            reflected_put = BlackScholes.get_price(reflected_spot, K, T, r, sigma, "put", q)
            price = (H / S) ** lambda_val * reflected_put
            return _make_result(price)

        # Up-and-out call (H > S, H > K) : Howison/Rubinstein-Reiner
        if self.barrier_type == "up-and-out" and self.option_type == "call":
            two_alpha = 1.0 - 2 * (r - q) / (sigma ** 2)
            refl = (H ** 2) / S
            term1 = _bs_call(S, K) - _bs_call(S, H) - (H - K) * BlackScholes.digital_call(S, H, T, r, sigma, q)
            term2 = (S / H) ** two_alpha * (
                _bs_call(refl, K) - _bs_call(refl, H) + (H - K) * BlackScholes.digital_call(refl, H, T, r, sigma, q)
            )
            price = max(0.0, term1 - term2)
            return _make_result(price)

        # Up-and-in call : parité Cu/o + Cu/i = C_vanille
        if self.barrier_type == "up-and-in" and self.option_type == "call":
            vanilla = _bs_call(S, K)
            uao = BarrierOption(S, K, T, r, sigma, H, "up-and-out", "call", 0, q)._analytical_price()
            price = max(0.0, vanilla - uao['price'])
            return _make_result(price)

        # Up-and-out put (H > K) : Pu/o = Pv(S,K) - (S/H)^2α * Pv(H²/S,K)
        if self.barrier_type == "up-and-out" and self.option_type == "put":
            two_alpha = 1.0 - 2 * (r - q) / (sigma ** 2)
            refl = (H ** 2) / S
            vanilla_put = BlackScholes.get_price(S, K, T, r, sigma, "put", q)
            reflected_put = BlackScholes.get_price(refl, K, T, r, sigma, "put", q)
            price = max(0.0, vanilla_put - (S / H) ** two_alpha * reflected_put)
            return _make_result(price)

        # Up-and-in put : parité Pu/o + Pu/i = P_vanille
        if self.barrier_type == "up-and-in" and self.option_type == "put":
            vanilla = BlackScholes.get_price(S, K, T, r, sigma, "put", q)
            uao = BarrierOption(S, K, T, r, sigma, H, "up-and-out", "put", 0, q)._analytical_price()
            price = max(0.0, vanilla - uao['price'])
            return _make_result(price)

        raise NotImplementedError(
            "Analytical price not implemented for this barrier type. "
            "Use method='monte_carlo'."
        )
    
    def compare_with_vanilla(self, n_simulations=50000, n_steps=252,
                             use_antithetic=True, seed=42, brownian_bridge=True):
        """
        Compare le prix de la barrière avec une option vanille équivalente.
        
        Returns:
            dict: Comparaison des prix et discount
        """
        # Prix de l'option barrière
        barrier_result = self.price(
            n_simulations=n_simulations,
            n_steps=n_steps,
            use_antithetic=use_antithetic,
            seed=seed,
            brownian_bridge=brownian_bridge,
        )
        barrier_price = barrier_result['price']
        
        # Prix vanille équivalent
        vanilla_bs = BlackScholes.get_price(
            self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q
        )
        
        # Discount (réduction de prix grâce à la barrière)
        discount = vanilla_bs - barrier_price
        discount_pct = (discount / vanilla_bs) * 100 if vanilla_bs > 0 else 0
        
        return {
            'barrier_price': barrier_price,
            'vanilla_price': vanilla_bs,
            'discount': discount,
            'discount_pct': discount_pct,
            'activation_rate': barrier_result.get('activation_rate', None)
        }
    
    def sensitivity_to_barrier(self, barrier_range, n_simulations=10000, n_steps=126):
        """
        Analyse de sensibilité : comment le prix varie avec le niveau de barrière.
        
        Args:
            barrier_range (array): Range de niveaux de barrière à tester
        
        Returns:
            dict: {'barriers': array, 'prices': array}
        """
        prices = []
        
        for barrier in barrier_range:
            temp_option = BarrierOption(
                self.S, self.K, self.T, self.r, self.sigma,
                barrier, self.barrier_type, self.option_type, self.rebate, self.q
            )
            
            try:
                result = temp_option.price(
                    n_simulations=n_simulations, n_steps=n_steps
                )  # Moins de sims pour rapidité
                prices.append(result['price'])
            except (ValueError, ZeroDivisionError, RuntimeError):
                prices.append(np.nan)
        
        return {
            'barriers': barrier_range,
            'prices': np.array(prices)
        }
