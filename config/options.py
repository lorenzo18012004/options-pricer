"""
Conventions centralisées pour le Pricer Options.
Toutes les marges d'erreur, seuils et tolérances sont définis ici pour garantir la cohérence.
"""

# --- Spread bid-ask (en fraction, ex: 0.15 = 15%) ---
MAX_SPREAD_PCT = 0.15  # Spread max acceptable pour options liquides (parité, surface, calibration)
MAX_SPREAD_PCT_RELAXED = 0.50  # Spread max pour clean_option_chain initial (pre-filtrage)

# --- IV (Implied Volatility) ---
IV_MIN_PCT = 0.05   # 5% - exclure IV trop basses (outliers)
IV_MAX_PCT = 0.80   # 80% - exclure IV trop hautes (outliers)
IV_CAP_DISPLAY_PCT = 100.0  # Plafond affichage surface 3D (%) — pas de masquage, afficher les vrais problèmes

# --- Liquidité ---
MIN_VOLUME = 5      # Volume minimum pour surface 3D
MIN_OPEN_INTEREST = 50  # OI minimum si volume = 0
MIN_MID_PRICE = 0.05   # Prix mid minimum ($)

# --- Tolérance breach Put-Call Parity ---
BREACH_TOLERANCE_FLOOR = 0.02   # Plancher $0.02 (bruit numérique)
BREACH_TOLERANCE_CAP = 0.10     # Plafond $0.10 (éviter masquer anomalies)

# --- Breach rate seuils (Put-Call Parity desk check) ---
BREACH_RATE_GOOD = 5.0    # < 5% = OK
BREACH_RATE_WARNING = 20.0  # < 20% = warning, >= 20% = error

# --- Moneyness (K/S) ---
MONEYNESS_MIN = 0.75  # Filtre chain principale (élargi pour ≥80% rétention)
MONEYNESS_MAX = 1.25
MONEYNESS_SURFACE_MIN = 0.70  # Filtre surface 3D (élargi pour ailes)
MONEYNESS_SURFACE_MAX = 1.30

# --- Surface 3D ---
SURFACE_SMOOTH_RBF = 0.001  # Très peu de lissage RBF pour préserver smile/term structure
SURFACE_GAUSSIAN_SIGMA = 0.2  # Léger smoothing gaussien (0 = désactivé)

OPTIONS_CONFIG = {
    "max_spread_pct": MAX_SPREAD_PCT,
    "max_spread_pct_relaxed": MAX_SPREAD_PCT_RELAXED,
    "iv_min_pct": IV_MIN_PCT,
    "iv_max_pct": IV_MAX_PCT,
    "iv_cap_display_pct": IV_CAP_DISPLAY_PCT,
    "min_volume": MIN_VOLUME,
    "min_open_interest": MIN_OPEN_INTEREST,
    "min_mid_price": MIN_MID_PRICE,
    "breach_tolerance_floor": BREACH_TOLERANCE_FLOOR,
    "breach_tolerance_cap": BREACH_TOLERANCE_CAP,
    "breach_rate_good": BREACH_RATE_GOOD,
    "breach_rate_warning": BREACH_RATE_WARNING,
    "moneyness_min": MONEYNESS_MIN,
    "moneyness_max": MONEYNESS_MAX,
    "moneyness_surface_min": MONEYNESS_SURFACE_MIN,
    "moneyness_surface_max": MONEYNESS_SURFACE_MAX,
    "surface_smooth_rbf": SURFACE_SMOOTH_RBF,
    "surface_gaussian_sigma": SURFACE_GAUSSIAN_SIGMA,
}
