import numpy as np
from models.monte_carlo import MonteCarloPricer
from core.black_scholes import BlackScholes

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
        if method == "monte_carlo":
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
        
        elif method == "analytical":
            # Formules analytiques disponibles pour certains cas
            return self._analytical_price()
        
        else:
            raise ValueError(f"Method {method} not supported")
    
    def _analytical_price(self):
        """
        Prix analytique pour barrière simple (formules de Merton).
        
        Disponible seulement pour certaines configurations.
        """
        # Implémentation simplifiée pour down-and-out call
        if self.barrier_type == "down-and-out" and self.option_type == "call":
            H = self.barrier_level
            
            # Paramètres
            lambda_val = (self.r + 0.5 * self.sigma ** 2) / (self.sigma ** 2)
            y = np.log(H ** 2 / (self.S * self.K)) / (self.sigma * np.sqrt(self.T))
            x1 = np.log(self.S / H) / (self.sigma * np.sqrt(self.T))
            y1 = np.log(H / self.S) / (self.sigma * np.sqrt(self.T))
            
            # Prix vanille
            vanilla_price = BlackScholes.get_price(
                self.S, self.K, self.T, self.r, self.sigma, self.option_type
            )
            
            # Approximation simplifiée
            if H >= self.K:
                # Barrière au-dessus du strike
                price = vanilla_price * (self.S / H) ** (2 * lambda_val)
            else:
                price = vanilla_price
            
            return {'price': price, 'method': 'analytical_approximation'}
        
        else:
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
            except:
                prices.append(np.nan)
        
        return {
            'barriers': barrier_range,
            'prices': np.array(prices)
        }


class AsianOption:
    """
    Option Asiatique : Payoff dépend de la moyenne du prix du sous-jacent.
    
    Payoff Call: max(Avg(S) - K, 0)
    Payoff Put: max(K - Avg(S), 0)
    
    Moins volatile qu'une option vanille car dépend de la moyenne et non du prix final.
    Utilisée pour les commodités et le forex.
    """
    
    def __init__(self, S, K, T, r, sigma, option_type="call", averaging_type="arithmetic"):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité
            option_type (str): "call" ou "put"
            averaging_type (str): "arithmetic" ou "geometric"
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.averaging_type = averaging_type
    
    def price(self, n_simulations=50000):
        """
        Prix de l'option asiatique par Monte Carlo.
        
        Returns:
            dict: {'price': prix, 'std_error': erreur}
        """
        mc = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_simulations=n_simulations, n_steps=252
        )
        
        return mc.price_asian(
            self.option_type,
            self.averaging_type,
            use_antithetic=True
        )
    
    def compare_with_vanilla(self):
        """Compare avec une option vanille équivalente."""
        # Prix asiatique
        asian_result = self.price()
        asian_price = asian_result['price']
        
        # Prix vanille
        vanilla_price = BlackScholes.get_price(
            self.S, self.K, self.T, self.r, self.sigma, self.option_type
        )
        
        # Discount
        discount = vanilla_price - asian_price
        discount_pct = (discount / vanilla_price) * 100 if vanilla_price > 0 else 0
        
        return {
            'asian_price': asian_price,
            'vanilla_price': vanilla_price,
            'discount': discount,
            'discount_pct': discount_pct,
            'reasoning': "Asiatique moins chère car moyenne réduit la volatilité du payoff"
        }


class LookbackOption:
    """
    Option Lookback : Payoff dépend du min ou max pendant la vie de l'option.
    
    Lookback Call: S(T) - min(S)
    Lookback Put: max(S) - S(T)
    
    "L'option du rêveur" : vous achetez au plus bas ou vendez au plus haut.
    Très chère car pas de strike fixe.
    """
    
    def __init__(self, S, T, r, sigma, option_type="call"):
        """
        Args:
            S (float): Prix spot
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité
            option_type (str): "call" ou "put"
        """
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
    
    def price(self, n_simulations=50000):
        """Prix par Monte Carlo."""
        mc = MonteCarloPricer(
            self.S, self.S, self.T, self.r, self.sigma,  # K = S (pas utilisé)
            n_simulations=n_simulations, n_steps=252
        )
        
        return mc.price_lookback(
            self.option_type,
            use_antithetic=True
        )
    
    def compare_with_vanilla_atm(self):
        """Compare avec un call/put ATM."""
        lookback_result = self.price()
        lookback_price = lookback_result['price']
        
        # Vanille ATM (K = S)
        vanilla_price = BlackScholes.get_price(
            self.S, self.S, self.T, self.r, self.sigma, self.option_type
        )
        
        premium = lookback_price - vanilla_price
        premium_pct = (premium / vanilla_price) * 100 if vanilla_price > 0 else 0
        
        return {
            'lookback_price': lookback_price,
            'vanilla_atm_price': vanilla_price,
            'premium': premium,
            'premium_pct': premium_pct,
            'reasoning': "Lookback plus chère car garantit le meilleur prix possible"
        }


class DigitalOption:
    """
    Option Digitale (Binary/Cash-or-Nothing).
    
    Payoff:
    - Call: Paye 1$ si S(T) > K, sinon 0
    - Put: Paye 1$ si S(T) < K, sinon 0
    
    Utilisée pour parier sur la direction (pas l'amplitude) du mouvement.
    """
    
    def __init__(self, S, K, T, r, sigma, option_type="call", payout=1.0):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité
            option_type (str): "call" ou "put"
            payout (float): Montant du payout digital
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.payout = payout
    
    def price(self):
        """
        Prix analytique de l'option digitale.
        
        Digital Call = exp(-rT) * N(d2)
        Digital Put = exp(-rT) * N(-d2)
        """
        from scipy.stats import norm
        
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type.lower() == "call":
            price = np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return price * self.payout
    
    def probability_of_payout(self):
        """
        Probabilité risque-neutre de recevoir le payout.
        """
        from scipy.stats import norm
        
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type.lower() == "call":
            prob = norm.cdf(d2)
        else:
            prob = norm.cdf(-d2)
        
        return prob
