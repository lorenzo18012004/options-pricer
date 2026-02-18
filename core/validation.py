"""
Validation stricte des inputs utilisateur.
Lève ValidationError en cas d'inputs invalides.
"""

from typing import Optional

from data.exceptions import ValidationError


def validate_ticker(ticker: str) -> str:
    """
    Valide le symbole ticker.
    
    Args:
        ticker: Symbole à valider
        
    Returns:
        Ticker normalisé (uppercase, stripped)
        
    Raises:
        ValidationError: Si ticker vide ou invalide
    """
    if not ticker or not isinstance(ticker, str):
        raise ValidationError("Le ticker ne peut pas être vide.")
    t = ticker.strip().upper()
    if not t:
        raise ValidationError("Le ticker ne peut pas être vide.")
    if len(t) > 20:
        raise ValidationError("Ticker trop long.")
    return t


def validate_option_params(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> None:
    """
    Valide les paramètres d'une option vanille.
    
    Raises:
        ValidationError: Si un paramètre est hors bornes
    """
    if S <= 0:
        raise ValidationError("Le spot S doit être strictement positif.")
    if K <= 0:
        raise ValidationError("Le strike K doit être strictement positif.")
    if T < 0:
        raise ValidationError("La maturité T doit être >= 0.")
    if sigma < 0:
        raise ValidationError("La volatilité sigma doit être >= 0.")
    if r < -0.5 or r > 1.0:
        raise ValidationError("Le taux r doit être dans [-50%, 100%].")
    if q < 0 or q > 1.0:
        raise ValidationError("Le dividend yield q doit être dans [0, 100%].")


def validate_barrier_params(
    S: float,
    barrier: float,
    option_type: str,
    barrier_type: str,
) -> None:
    """
    Valide la cohérence des paramètres d'une option barrière.
    
    Raises:
        ValidationError: Si barrière incohérente (ex: up-and-out avec barrier < S)
    """
    if S <= 0 or barrier <= 0:
        raise ValidationError("Spot et barrière doivent être strictement positifs.")
    opt = option_type.lower()
    btype = barrier_type.lower()
    if "up" in btype and barrier <= S:
        raise ValidationError(
            "Pour une barrière up, le niveau doit être strictement au-dessus du spot."
        )
    if "down" in btype and barrier >= S:
        raise ValidationError(
            "Pour une barrière down, le niveau doit être strictement en-dessous du spot."
        )


def validate_swap_params(
    notional: float,
    fixed_rate: float,
    maturity: float,
    payment_freq: float,
) -> None:
    """
    Valide les paramètres d'un swap.
    
    Raises:
        ValidationError: Si un paramètre est invalide
    """
    if notional <= 0:
        raise ValidationError("Le notionnel doit être strictement positif.")
    if maturity <= 0:
        raise ValidationError("La maturité doit être strictement positive.")
    if payment_freq <= 0:
        raise ValidationError("La fréquence de paiement doit être > 0.")
    if fixed_rate < -0.5 or fixed_rate > 1.0:
        raise ValidationError("Le taux fixe doit être dans [-50%, 100%].")
