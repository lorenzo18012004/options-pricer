"""
Constantes nommées pour le Pricer.
Évite les magic numbers dispersés dans le code.
"""

# Tolérances numériques
IV_EPS = 1e-10
SIGMA_MIN = 1e-10
PARITY_TOLERANCE = 1e-6

# Bornes IV
IV_MIN = 0.001
IV_MAX = 5.0

# Cache
CACHE_TTL_SECONDS = 30

# Retry
REQUEST_TIMEOUT = 15
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_BACKOFF = 2.0

# Spinner timeout (secondes) - message affiché si chargement long
SPINNER_TIMEOUT_SECONDS = 60
SPINNER_TIMEOUT_MESSAGE = "Si le chargement dépasse 60 s, vérifiez votre connexion."
