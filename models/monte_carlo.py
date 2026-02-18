import numpy as np

class MonteCarloPricer:
    """
    Pricing d'options par simulations de Monte Carlo.
    
    Utilisé pour les produits path-dependent (barrières).
    Implémente des techniques de réduction de variance (Antithetic Variates).
    """
    
    def __init__(self, S, K, T, r, sigma, n_simulations=10000, n_steps=252, seed=42, q=0.0):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité en années
            r (float): Taux sans risque
            sigma (float): Volatilité
            n_simulations (int): Nombre de simulations (paths)
            n_steps (int): Nombre de pas de temps (252 = daily pour 1 an)
            seed (int): Seed pour reproductibilité
            q (float): Dividend yield continu (decimal)
        """
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be > 0")
        if T <= 0:
            raise ValueError("T must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if int(n_simulations) <= 0 or int(n_steps) <= 0:
            raise ValueError("n_simulations and n_steps must be > 0")

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed
        
        self.dt = T / n_steps
        self.rng = np.random.default_rng(seed)
    
    def _simulate_paths(self, use_antithetic=True):
        """
        Simule des trajectoires du prix du sous-jacent avec un schéma d'Euler.
        
        dS = r * S * dt + sigma * S * dW
        où dW ~ N(0, sqrt(dt))
        
        Args:
            use_antithetic (bool): Utiliser Antithetic Variates pour réduire la variance
        
        Returns:
            np.array: Matrice (n_simulations, n_steps+1) des prix
        """
        if use_antithetic:
            # On génère seulement n_simulations/2 paths
            # et on crée les antithetic paths (avec -Z au lieu de Z)
            half_sims = (self.n_simulations + 1) // 2
            
            # Générer les chocs aléatoires
            Z = self.rng.standard_normal((half_sims, self.n_steps))
            Z_anti = -Z  # Antithetic variates
            
            # Combiner
            Z_combined = np.vstack([Z, Z_anti])[:self.n_simulations]
        else:
            Z_combined = self.rng.standard_normal((self.n_simulations, self.n_steps))
        
        # Initialisation des paths
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.S
        
        # Simulation avec schéma d'Euler
        for t in range(1, self.n_steps + 1):
            drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * Z_combined[:, t - 1]
            
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)
        
        return paths
    
    def price_european(self, option_type="call", use_antithetic=True):
        """
        Prix d'une option européenne vanille par Monte Carlo.
        
        (Pour benchmark avec Black-Scholes)
        
        Args:
            option_type (str): "call" ou "put"
            use_antithetic (bool): Réduction de variance
        
        Returns:
            dict: {'price': prix, 'std_error': erreur standard}
        """
        paths = self._simulate_paths(use_antithetic)
        
        # Prix final de chaque path
        final_prices = paths[:, -1]
        
        # Payoff
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - final_prices, 0)
        
        # Actualisation
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Prix = moyenne des payoffs actualisés
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        return {'price': price, 'std_error': std_error}
    
    def price_barrier(self, barrier_level, barrier_type="down-and-out",
                      option_type="call", use_antithetic=True,
                      brownian_bridge=True):
        """
        Prix d'une option barrière par Monte Carlo.
        
        Types de barrières:
        - "down-and-out": Option disparaît si S touche la barrière en dessous
        - "down-and-in": Option s'active si S touche la barrière en dessous
        - "up-and-out": Option disparaît si S touche la barrière au dessus
        - "up-and-in": Option s'active si S touche la barrière au dessus
        
        Args:
            barrier_level (float): Niveau de la barrière
            barrier_type (str): Type de barrière
            option_type (str): "call" ou "put"
            use_antithetic (bool): Réduction de variance
            brownian_bridge (bool): Ajustement de franchissement intra-step
        
        Returns:
            dict: {'price': prix, 'std_error': erreur, 'activation_rate': % de paths activés}
        """
        paths = self._simulate_paths(use_antithetic)
        
        # Détection du franchissement de barrière pour chaque path
        barrier_hit = self._check_barrier_hit(
            paths, barrier_level, barrier_type, brownian_bridge=brownian_bridge
        )
        
        # Prix final
        final_prices = paths[:, -1]
        
        # Payoff standard
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - final_prices, 0)
        
        # Application de la condition de barrière
        if "out" in barrier_type.lower():
            # Knock-out : payoff = 0 si barrière touchée
            payoffs = np.where(barrier_hit, 0, payoffs)
        else:
            # Knock-in : payoff = 0 si barrière PAS touchée
            payoffs = np.where(barrier_hit, payoffs, 0)
        
        # Actualisation
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        activation_rate = np.mean(barrier_hit)
        
        return {
            'price': price, 
            'std_error': std_error,
            'activation_rate': activation_rate
        }
    
    def _check_barrier_hit(self, paths, barrier_level, barrier_type, brownian_bridge=True):
        """
        Vérifie si chaque path a touché la barrière.
        
        Args:
            paths (np.array): Matrice des prix simulés
            barrier_level (float): Niveau de barrière
            barrier_type (str): Type de barrière
        
        Returns:
            np.array (bool): True si barrière touchée
        """
        barrier_type = barrier_type.lower()
        if not brownian_bridge:
            if "down" in barrier_type:
                return np.min(paths, axis=1) <= barrier_level
            return np.max(paths, axis=1) >= barrier_level

        # Brownian-bridge hit detection in log-space for GBM between dates.
        # This reduces the discrete monitoring bias when barriers are close.
        if barrier_level <= 0:
            return np.zeros(paths.shape[0], dtype=bool)

        eps = 1e-12
        sigma2_dt = max(self.sigma ** 2 * self.dt, eps)
        log_h = np.log(max(barrier_level, eps))
        log_paths = np.log(np.maximum(paths, eps))

        n_paths = paths.shape[0]
        hit = np.zeros(n_paths, dtype=bool)

        for t in range(1, paths.shape[1]):
            if np.all(hit):
                break
            x0 = log_paths[:, t - 1]
            x1 = log_paths[:, t]
            alive = ~hit

            if "down" in barrier_type:
                direct = (x0 <= log_h) | (x1 <= log_h)
                hit |= direct
                idx = alive & (~direct)
                if np.any(idx):
                    a = x0[idx] - log_h
                    b = x1[idx] - log_h
                    p_cross = np.exp(np.clip(-2.0 * a * b / sigma2_dt, -700.0, 0.0))
                    u = self.rng.random(np.sum(idx))
                    tmp = np.zeros(n_paths, dtype=bool)
                    tmp[idx] = u < p_cross
                    hit |= tmp
            else:
                direct = (x0 >= log_h) | (x1 >= log_h)
                hit |= direct
                idx = alive & (~direct)
                if np.any(idx):
                    a = log_h - x0[idx]
                    b = log_h - x1[idx]
                    p_cross = np.exp(np.clip(-2.0 * a * b / sigma2_dt, -700.0, 0.0))
                    u = self.rng.random(np.sum(idx))
                    tmp = np.zeros(n_paths, dtype=bool)
                    tmp[idx] = u < p_cross
                    hit |= tmp

        return hit
    
    def calculate_greeks_mc(self, option_type="call", bump_size=0.01):
        """
        Calcul des Grecs par différences finies (bumping).

        Moins précis que les formules analytiques, mais fonctionne
        pour tous les produits exotiques.

        Args:
            option_type (str): "call" ou "put"
            bump_size (float): Taille du bump pour les différences finies

        Returns:
            dict: {'delta': delta, 'gamma': gamma, 'vega': vega}
        """
        base_S, base_sigma = self.S, self.sigma
        try:
            # Prix de base
            base_price = self.price_european(option_type)['price']

            # Delta : bump du spot
            self.S = base_S + bump_size
            price_up = self.price_european(option_type)['price']
            self.S = base_S - bump_size
            price_down = self.price_european(option_type)['price']

            delta = (price_up - price_down) / (2 * bump_size)
            gamma = (price_up - 2 * base_price + price_down) / (bump_size ** 2)

            # Vega : bump de la vol
            self.S = base_S  # Restaurer S avant vega
            self.sigma = base_sigma + bump_size
            price_vol_up = self.price_european(option_type)['price']
            vega = (price_vol_up - base_price) / bump_size

            return {'delta': delta, 'gamma': gamma, 'vega': vega}
        finally:
            self.S, self.sigma = base_S, base_sigma
