import numpy as np
from core.black_scholes import BlackScholes
from models.trees import BinomialTree
from models.monte_carlo import MonteCarloPricer

class VanillaOption:
    """
    Option Vanille Européenne ou Américaine.
    
    Classe de base pour les options simples.
    """
    
    def __init__(self, S, K, T, r, sigma, option_type="call", style="european", q=0.0):
        """
        Args:
            S (float): Prix spot
            K (float): Strike
            T (float): Maturité (années)
            r (float): Taux sans risque
            sigma (float): Volatilité implicite
            option_type (str): "call" ou "put"
            style (str): "european" ou "american"
            q (float): Dividend yield continu (decimal)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type
        self.style = style
    
    def price(self, method="analytical"):
        """
        Prix de l'option.
        
        Args:
            method (str): "analytical" (Black-Scholes) ou "tree" (Binomial)
        
        Returns:
            float: Prix de l'option
        """
        if self.style == "european" and method == "analytical":
            return BlackScholes.get_price(
                self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q
            )
        elif method == "tree":
            tree = BinomialTree(
                self.S, self.K, self.T, self.r, self.sigma, 
                self.option_type, n_steps=1000, q=self.q
            )
            bs_european = BlackScholes.get_price(
                self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q
            )
            return tree.price_american_cv(bs_european)
        else:
            raise ValueError(f"Method {method} not supported for {self.style}")
    
    def greeks(self):
        """Calcule tous les Grecs."""
        return BlackScholes.get_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q
        )
    
    def intrinsic_value(self):
        """Valeur intrinsèque de l'option."""
        if self.option_type.lower() == "call":
            return max(self.S - self.K, 0)
        else:
            return max(self.K - self.S, 0)
    
    def time_value(self):
        """Valeur temps de l'option."""
        return self.price() - self.intrinsic_value()
    
    def moneyness(self):
        """
        Retourne le moneyness de l'option.
        
        Returns:
            str: "ITM", "ATM", ou "OTM"
        """
        ratio = self.S / self.K
        
        if abs(ratio - 1.0) < 0.02:  # ±2%
            return "ATM"
        elif self.option_type.lower() == "call":
            return "ITM" if ratio > 1.0 else "OTM"
        else:
            return "ITM" if ratio < 1.0 else "OTM"
    
    def summary(self):
        """Résumé complet de l'option."""
        greeks = self.greeks()
        
        return {
            'type': f"{self.style.capitalize()} {self.option_type.capitalize()}",
            'spot': self.S,
            'strike': self.K,
            'maturity': f"{self.T:.2f} years",
            'volatility': f"{self.sigma*100:.1f}%",
            'moneyness': self.moneyness(),
            'price': greeks['price'],
            'intrinsic_value': self.intrinsic_value(),
            'time_value': self.time_value(),
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'],
            'theta': greeks['theta'],
            'rho': greeks['rho']
        }


class Straddle:
    """
    Straddle : Achat simultané d'un Call et Put ATM.
    
    Stratégie de volatilité pure (delta-neutre).
    Profite des mouvements importants du sous-jacent dans les deux directions.
    """
    
    def __init__(self, S, K, T, r, sigma):
        """
        Args:
            S (float): Prix spot
            K (float): Strike (typiquement ATM)
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité implicite
        """
        self.call = VanillaOption(S, K, T, r, sigma, "call")
        self.put = VanillaOption(S, K, T, r, sigma, "put")
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def price(self):
        """Coût total du straddle."""
        return self.call.price() + self.put.price()
    
    def greeks(self):
        """
        Grecs du straddle (somme des Grecs du call et du put).
        
        Delta devrait être ~0 pour un straddle ATM.
        Vega et Gamma positifs (exposition à la volatilité).
        """
        call_greeks = self.call.greeks()
        put_greeks = self.put.greeks()
        
        return {
            'price': call_greeks['price'] + put_greeks['price'],
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }
    
    def breakeven_points(self):
        """
        Points morts (breakeven) du straddle.
        
        Le straddle est profitable si le prix final est en dehors de ces bornes.
        """
        cost = self.price()
        
        return {
            'lower_breakeven': self.K - cost,
            'upper_breakeven': self.K + cost
        }
    
    def pnl_at_expiry(self, final_spot):
        """
        P&L du straddle à l'expiration pour un prix final donné.
        
        Args:
            final_spot (float): Prix du sous-jacent à l'expiration
        
        Returns:
            float: P&L
        """
        call_payoff = max(final_spot - self.K, 0)
        put_payoff = max(self.K - final_spot, 0)
        
        total_payoff = call_payoff + put_payoff
        cost = self.price()
        
        return total_payoff - cost
    
    def pnl_explain(self, new_S, new_sigma, time_passed_days=0):
        """
        P&L Explain : Décomposition du P&L en contributions des facteurs de marché.
        
        Args:
            new_S (float): Nouveau prix spot
            new_sigma (float): Nouvelle volatilité
            time_passed_days (int): Jours écoulés
        
        Returns:
            dict: Décomposition du P&L
        """
        # Valeur initiale
        initial_value = self.price()
        initial_greeks = self.greeks()
        
        # Nouvelle maturité
        new_T = self.T - (time_passed_days / 365)
        if new_T < 0:
            new_T = 0
        
        # Créer un nouveau straddle avec les nouveaux paramètres
        new_straddle = Straddle(new_S, self.K, new_T, self.r, new_sigma)
        new_value = new_straddle.price()
        
        # P&L total
        total_pnl = new_value - initial_value
        
        # Contributions (approximations de premier ordre)
        delta_pnl = initial_greeks['delta'] * (new_S - self.S)
        gamma_pnl = 0.5 * initial_greeks['gamma'] * (new_S - self.S) ** 2
        vega_pnl = initial_greeks['vega'] * (new_sigma - self.sigma) * 100
        theta_pnl = initial_greeks['theta'] * time_passed_days
        
        # Résiduel (termes d'ordre supérieur + interactions)
        explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        residual = total_pnl - explained_pnl
        
        return {
            'total_pnl': total_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'residual': residual,
            'initial_value': initial_value,
            'new_value': new_value
        }


class Strangle:
    """
    Strangle : Achat d'un Call OTM et Put OTM.
    
    Moins cher qu'un straddle mais nécessite des mouvements plus importants.
    """
    
    def __init__(self, S, K_put, K_call, T, r, sigma):
        """
        Args:
            S (float): Prix spot
            K_put (float): Strike du put (< S)
            K_call (float): Strike du call (> S)
            T (float): Maturité
            r (float): Taux sans risque
            sigma (float): Volatilité implicite
        """
        if K_put >= S or K_call <= S:
            raise ValueError("Pour un strangle: K_put < S < K_call")
        
        self.call = VanillaOption(S, K_call, T, r, sigma, "call")
        self.put = VanillaOption(S, K_put, T, r, sigma, "put")
        
        self.S = S
        self.K_put = K_put
        self.K_call = K_call
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def price(self):
        """Coût total du strangle."""
        return self.call.price() + self.put.price()
    
    def greeks(self):
        """Grecs du strangle."""
        call_greeks = self.call.greeks()
        put_greeks = self.put.greeks()
        
        return {
            'price': call_greeks['price'] + put_greeks['price'],
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }
    
    def breakeven_points(self):
        """Points morts du strangle."""
        cost = self.price()
        
        return {
            'lower_breakeven': self.K_put - cost,
            'upper_breakeven': self.K_call + cost
        }
    
    def pnl_at_expiry(self, final_spot):
        """P&L à l'expiration."""
        call_payoff = max(final_spot - self.K_call, 0)
        put_payoff = max(self.K_put - final_spot, 0)
        
        total_payoff = call_payoff + put_payoff
        cost = self.price()
        
        return total_payoff - cost
    
    def compare_with_straddle(self):
        """
        Compare le strangle avec un straddle ATM équivalent.
        
        Returns:
            dict: Comparaison des deux stratégies
        """
        # Straddle ATM
        K_atm = self.S
        straddle = Straddle(self.S, K_atm, self.T, self.r, self.sigma)
        
        return {
            'strangle_cost': self.price(),
            'straddle_cost': straddle.price(),
            'cost_savings': straddle.price() - self.price(),
            'cost_savings_pct': ((straddle.price() - self.price()) / straddle.price()) * 100,
            'strangle_breakevens': self.breakeven_points(),
            'straddle_breakevens': straddle.breakeven_points()
        }


class BidAskSpread:
    """
    Modélisation du coût réel avec Bid-Ask Spread.
    
    Montre que beaucoup de stratégies théoriquement gagnantes
    sont perdantes en réalité à cause des coûts de transaction.
    """
    
    @staticmethod
    def apply_spread(theoretical_price, spread_pct=0.02):
        """
        Applique un spread bid-ask à un prix théorique.
        
        Args:
            theoretical_price (float): Prix théorique (mid)
            spread_pct (float): Spread en % du mid
        
        Returns:
            dict: {'bid': bid, 'ask': ask, 'mid': mid}
        """
        half_spread = theoretical_price * spread_pct / 2
        
        return {
            'bid': theoretical_price - half_spread,
            'ask': theoretical_price + half_spread,
            'mid': theoretical_price,
            'spread_pct': spread_pct * 100
        }
    
    @staticmethod
    def calculate_roundtrip_cost(strategy_legs, spread_pct=0.02):
        """
        Calcule le coût d'un aller-retour (buy + sell) pour une stratégie.
        
        Args:
            strategy_legs (list): Liste de dicts avec 'price' et 'quantity'
            spread_pct (float): Spread moyen
        
        Returns:
            dict: Coûts de transaction
        """
        total_cost = 0
        
        for leg in strategy_legs:
            price = leg['price']
            quantity = abs(leg['quantity'])
            
            # Coût d'achat (on paye l'ask) + vente (on reçoit le bid)
            spread = BidAskSpread.apply_spread(price, spread_pct)
            roundtrip_cost = (spread['ask'] - spread['bid']) * quantity
            
            total_cost += roundtrip_cost
        
        return {
            'total_roundtrip_cost': total_cost,
            'cost_per_leg': total_cost / len(strategy_legs) if strategy_legs else 0
        }
