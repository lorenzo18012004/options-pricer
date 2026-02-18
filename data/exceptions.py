"""
Exceptions spécifiques pour le module data.

Évite de masquer les erreurs avec except Exception.
"""


class DataError(Exception):
    """Erreur liée aux données (format, valeur manquante, incohérence)."""
    pass


class NetworkError(DataError):
    """Erreur réseau (connexion, timeout, rate limit)."""
    pass


class ConfigurationError(DataError):
    """Erreur de configuration (ticker inconnu, paramètre invalide)."""
    pass


class ValidationError(ValueError, DataError):
    """Erreur de validation des inputs utilisateur (strike négatif, ticker vide, etc.)."""
    pass
