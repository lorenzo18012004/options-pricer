import numpy as np

class BinomialTree:
    """
    Modèle d'arbre binomial pour le pricing d'options américaines.
    
    Contrairement au modèle Black-Scholes, l'arbre binomial permet de gérer
    l'exercice anticipé des options américaines en comparant à chaque nœud
    la valeur de continuation vs la valeur d'exercice immédiat.
    
    Modèle de Cox-Ross-Rubinstein (CRR).
    """
    
    def __init__(self, S, K, T, r, sigma, option_type="call", n_steps=100, dividends=None, q=0.0):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité en années
            r (float): Taux sans risque
            sigma (float): Volatilité
            option_type (str): "call" ou "put"
            n_steps (int): Nombre de steps dans l'arbre (plus = précis mais lent)
            dividends (list): Liste de tuples (t, div_amount) pour dividendes discrets
            q (float): Dividend yield continu (decimal)
        """
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be > 0")
        if T <= 0:
            raise ValueError("T must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if int(n_steps) <= 0:
            raise ValueError("n_steps must be > 0")

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_steps = n_steps
        self.dividends = dividends if dividends else []
        
        # Paramètres de l'arbre CRR
        self.dt = T / n_steps
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        # Probabilite risque-neutre avec cost-of-carry (r - q)
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(
                f"Risk-neutral probability out of bounds: p={self.p:.6f}. "
                "Check inputs (dt, r, q, sigma)."
            )
        
        # Facteur d'actualisation au taux SANS risque (pas r-q)
        self.discount = np.exp(-r * self.dt)
    
    def _build_stock_tree(self):
        """
        Construit l'arbre des prix du sous-jacent.
        
        Returns:
            np.array: Matrice (n_steps+1, n_steps+1) des prix
        """
        tree = np.zeros((self.n_steps + 1, self.n_steps + 1))
        
        for i in range(self.n_steps + 1):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)
        
        # Ajuster pour les dividendes discrets
        if self.dividends:
            for div_time, div_amount in self.dividends:
                div_step = int(div_time / self.dt)
                if div_step <= self.n_steps:
                    # Réduire les prix après le paiement du dividende
                    tree[:, div_step:] -= div_amount
        
        return tree
    
    def _get_payoff(self, stock_price):
        """
        Calcul du payoff à l'exercice.
        
        Args:
            stock_price (float): Prix du sous-jacent
        
        Returns:
            float: Payoff de l'option
        """
        if self.option_type == "call":
            return max(stock_price - self.K, 0)
        else:
            return max(self.K - stock_price, 0)
    
    def price(self):
        """
        Calcule le prix de l'option américaine.
        
        Utilise la backward induction : on part de la maturité et on remonte
        en comparant à chaque nœud la valeur de continuation vs exercice.
        
        Returns:
            float: Prix de l'option américaine
        """
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        
        # Initialisation à maturité (T)
        for j in range(self.n_steps + 1):
            option_tree[j, self.n_steps] = self._get_payoff(stock_tree[j, self.n_steps])
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Valeur de continuation (espérance actualisée)
                continuation_value = self.discount * (
                    self.p * option_tree[j, i + 1] +
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                
                # Valeur d'exercice immédiat
                exercise_value = self._get_payoff(stock_tree[j, i])
                
                # Option américaine : max(exercice, continuation)
                option_tree[j, i] = max(exercise_value, continuation_value)
        
        return option_tree[0, 0]
    
    def get_early_exercise_boundary(self):
        """
        Retourne la frontière d'exercice anticipé optimal.
        
        C'est la courbe qui sépare les régions "Exercer maintenant" vs "Continuer".
        Utile pour visualiser quand exercer une option américaine.
        
        Returns:
            list: Liste de tuples (temps, prix_critique)
        """
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        
        # Initialisation à maturité
        for j in range(self.n_steps + 1):
            option_tree[j, self.n_steps] = self._get_payoff(stock_tree[j, self.n_steps])
        
        boundary = []
        
        # Backward induction avec détection de la frontière
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = self.discount * (
                    self.p * option_tree[j, i + 1] +
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )
                exercise_value = self._get_payoff(stock_tree[j, i])
                
                option_tree[j, i] = max(exercise_value, continuation_value)
                
                # Si on exerce, on enregistre le prix critique
                if exercise_value > continuation_value and exercise_value > 0:
                    boundary.append((i * self.dt, stock_tree[j, i]))
        
        return boundary


class TrinomialTree:
    """
    Modèle d'arbre trinomial pour le pricing d'options.
    
    Plus flexible que le binomial (3 branches au lieu de 2).
    Permet une meilleure convergence pour certains produits exotiques.
    """
    
    def __init__(self, S, K, T, r, sigma, option_type="call", n_steps=100):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité
            option_type (str): "call" ou "put"
            n_steps (int): Nombre de steps
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.n_steps = n_steps
        
        # Paramètres de l'arbre trinomial
        self.dt = T / n_steps
        self.u = np.exp(sigma * np.sqrt(2 * self.dt))
        self.d = 1 / self.u
        self.m = 1  # Middle (pas de mouvement)
        
        # Probabilités risque-neutre
        dx = sigma * np.sqrt(self.dt)
        nu = r - 0.5 * sigma ** 2
        
        self.pu = 0.5 * ((sigma ** 2 * self.dt + nu ** 2 * self.dt ** 2) / dx ** 2 + nu * self.dt / dx)
        self.pd = 0.5 * ((sigma ** 2 * self.dt + nu ** 2 * self.dt ** 2) / dx ** 2 - nu * self.dt / dx)
        self.pm = 1 - self.pu - self.pd
        
        self.discount = np.exp(-r * self.dt)
    
    def _get_payoff(self, stock_price):
        """Calcul du payoff."""
        if self.option_type == "call":
            return max(stock_price - self.K, 0)
        else:
            return max(self.K - stock_price, 0)
    
    def price(self):
        """
        Prix de l'option avec arbre trinomial.
        
        Returns:
            float: Prix de l'option
        """
        # Construction de l'arbre des prix
        n = self.n_steps
        stock_tree = {}
        
        # Initialisation
        for i in range(-n, n + 1):
            stock_tree[(n, i)] = self.S * (self.u ** i)
        
        # Valeurs finales
        option_tree = {}
        for i in range(-n, n + 1):
            option_tree[(n, i)] = self._get_payoff(stock_tree[(n, i)])
        
        # Backward induction
        for step in range(n - 1, -1, -1):
            for i in range(-step, step + 1):
                stock_price = self.S * (self.u ** i)
                stock_tree[(step, i)] = stock_price
                
                # Valeur de continuation
                continuation = self.discount * (
                    self.pu * option_tree.get((step + 1, i + 1), 0) +
                    self.pm * option_tree.get((step + 1, i), 0) +
                    self.pd * option_tree.get((step + 1, i - 1), 0)
                )
                
                # Pour américaine : max(exercice, continuation)
                exercise = self._get_payoff(stock_price)
                option_tree[(step, i)] = max(exercise, continuation)
        
        return option_tree[(0, 0)]
