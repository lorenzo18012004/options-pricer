# Feuille de route pour atteindre 18/20

## Vue d'ensemble

| Priorité | Tâche | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | Remplacer `except Exception` par des exceptions ciblées | Robustesse + Architecture | Élevé |
| **P0** | Validation stricte des inputs utilisateur | Robustesse + UX | Moyen |
| **P1** | Découper `vanilla_helpers.py` en sous-modules | Architecture + Maintenabilité | Moyen |
| **P1** | Utiliser le Protocol pour le typage des connecteurs | Architecture | Faible |
| **P1** | ACT/365 avec vraies dates calendaires | Modèles (Swap) | Moyen |
| **P1** | Garde sigma quasi-nul + IV retourne raison explicite | Modèles | Faible |
| **P2** | Tests FallbackDataConnector, Heston, MC barrières | Tests | Moyen |
| **P2** | Constantes nommées (magic numbers) | Qualité code | Faible |
| **P2** | Timeout sur les spinners | UX | Faible |
| **P2** | Docstrings complètes (API) | Documentation | Moyen |
| **P3** | Cache TTL configurable, Fallback sans session_state | Données | Faible |
| **P3** | Correlation ID pour le logging | Robustesse | Faible |

---

## P0 — CRITIQUE (obligatoire pour 18)

### 1. Remplacer `except Exception` par des exceptions ciblées

**Fichiers à modifier :**

| Fichier | Lignes | Action |
|---------|--------|--------|
| `data/connector.py` | 34, 39, 63, 144, 158, 204, 244, 280, 319, 365, 439, 479, 570, 637, 662 | Capturer `NetworkError`, `DataError`, `KeyError`, `ValueError` séparément. Propager ou logger selon le cas. |
| `data/synthetic.py` | 36, 40, 43, 327 | Capturer `FileNotFoundError`, `KeyError`, `ValueError`. |
| `services/market_service.py` | 154 | Dans `build_expiration_options` : capturer `DataError` ou `KeyError`, logger et continuer. Ne pas avaler tout. |
| `pages/vanilla_helpers.py` | 1062 | Import scipy : capturer `ImportError` uniquement. |
| `services/vanilla_service.py` | 103 | Capturer les exceptions spécifiques de calibration (ValueError, etc.). Retourner fallback uniquement si pertinent. |
| `pages/vanilla_page.py` | 304 | Capturer `ValueError`, `DataError`, `NetworkError`. Afficher message utilisateur adapté. |
| `pages/volatility_page.py` | 590 | Idem. |
| `pages/barrier_page.py` | 278, 319, 392 | Capturer `ValueError` pour les calculs. `st.error` avec message clair. |
| `pages/swap_page.py` | 59 | Déjà partiel : afficher `e_curve`. S'assurer que c'est une exception métier. |
| `instruments/exotic.py` | 280 | `sensitivity_to_barrier` : capturer `ValueError`, `ZeroDivisionError`. Propager le reste. |
| `core/heston.py` | 128, 349, 374 | Intégration quad : capturer `IntegrationWarning` ou erreur de convergence. DE : capturer `RuntimeError`. |
| `models/surfaces.py` | 106 | Capturer `ValueError` ou `KeyError` pour le point de surface. |

**Exceptions à utiliser (déjà dans `data/exceptions.py`) :**
- `NetworkError` : timeout, connexion refusée
- `DataError` : données manquantes, format invalide, ticker inconnu

**À ajouter si besoin :**
- `ValidationError` : inputs invalides (strike négatif, etc.)

---

### 2. Validation stricte des inputs utilisateur

**Où valider :**
- **Entrée des pages** : ticker non vide, strike > 0, T > 0, sigma > 0, barrier cohérent (up > S pour up-and-out, etc.)
- **Avant appel aux services** : dans `vanilla_service`, `barrier_page`, `swap_page`, etc.

**Actions :**
1. Créer un module `core/validation.py` avec des fonctions :
   - `validate_ticker(ticker: str) -> str` : lève `ValidationError` si vide ou invalide
   - `validate_option_params(S, K, T, r, sigma, ...)` : lève `ValidationError` si hors bornes
   - `validate_barrier_params(S, barrier, option_type, barrier_type)` : cohérence up/down, in/out
2. Appeler ces validateurs au début de chaque handler de page et de chaque fonction de service exposée.
3. Afficher un message clair à l'utilisateur en cas d'erreur (ex. "Strike doit être strictement positif").

---

## P1 — IMPORTANT

### 3. Découper `vanilla_helpers.py` (~1155 lignes)

**Structure proposée :**
```
pages/
  vanilla/
    __init__.py
    tabs.py          # render_tab_* (par onglet)
    chains.py        # logique chaînes d'options, expiration
    calibration.py   # put-call parity, calibration UI
    display.py       # affichage tableaux, graphiques
    inputs.py        # widgets communs (ticker, strike, etc.)
```

**Ou plus simple :**
- `vanilla_helpers_tabs.py` : rendu des onglets
- `vanilla_helpers_chains.py` : chaînes, expirations
- `vanilla_helpers_display.py` : tableaux, graphiques
- Garder `vanilla_helpers.py` comme façade qui réexporte

Objectif : aucun fichier > 400 lignes.

---

### 4. Utiliser le Protocol pour le typage des connecteurs

**Actions :**
1. Dans `data/connector.py` : ajouter `class YahooDataConnector(DataConnectorProtocol):` (ou faire hériter/implements).
2. Dans `data/synthetic.py` : `class SyntheticDataConnector(DataConnectorProtocol):`
3. Dans `services/market_service.py` : `FallbackDataConnector` doit aussi respecter le protocol.
4. Typer les paramètres : `def get_data(connector: DataConnectorProtocol, ...)` au lieu de types génériques.

---

### 5. ACT/365 avec vraies dates calendaires

**Fichier :** `core/curves.py` ou module dédié (ex. `core/daycount.py`).

**Action :**
- Utiliser `datetime` ou `date` pour calculer les jours réels entre deux dates.
- Remplacer `days_approx = (t_i - t_prev) * 365` par un calcul basé sur les jours calendaires.
- Adapter `accrual_fraction_act365(t1, t2)` pour utiliser `(date2 - date1).days / 365`.

---

### 6. Garde sigma quasi-nul + IV retourne raison explicite

**Fichier :** `core/black_scholes.py`
- Dans `_d1d2` : ajouter `if sigma < 1e-10: raise ValueError("sigma too small, numerical instability")`.

**Fichier :** `core/solvers.py`
- Au lieu de retourner `None` quand IV échoue, retourner une structure ou lever une exception avec message :
  - `IVOutOfBoundsError` : sigma serait < 0 ou > 5
  - `IVNotConvergedError` : Brent n'a pas convergé en N itérations
- Ou retourner `Optional[Tuple[float, Optional[str]]]` : (iv, reason) avec reason = "out_of_bounds" | "not_converged" | None.

---

## P2 — SOUHAITABLE

### 7. Tests manquants

**À ajouter dans `test_quant_regression.py` :**

```python
# FallbackDataConnector
def test_fallback_connector_on_yahoo_failure():
    """Quand Yahoo échoue, FallbackDataConnector utilise Synthetic."""
    # Mock Yahoo pour lever NetworkError, vérifier que Synthetic est utilisé
    ...

# Heston edge cases
def test_heston_v0_zero():
    """v0=0 doit être géré (ou rejeté proprement)."""
def test_heston_kappa_zero():
    """kappa=0 : variance constante."""
def test_heston_theta_negative():
    """theta négatif rejeté ou géré."""

# Monte Carlo barrières
def test_mc_barrier_converges_to_analytic():
    """MC up-and-out avec 50k paths proche de la formule analytique à 1%."""
```

---

### 8. Constantes nommées

**Créer `core/constants.py` (ou dans chaque module) :**
```python
# Tolérances numériques
IV_EPS = 1e-10
SIGMA_MIN = 1e-10
PARITY_TOLERANCE = 1e-6

# Bornes IV
IV_MIN = 0.001
IV_MAX = 5.0

# Cache
CACHE_TTL_SECONDS = 30
```

Remplacer les magic numbers dans `black_scholes.py`, `solvers.py`, `connector.py`, etc.

---

### 9. Timeout sur les spinners

**Action :**
- Utiliser `st.spinner` avec un wrapper ou un thread qui timeout après 30–60 s.
- Ou : `with st.spinner("Chargement...")` + `threading.Timer` pour afficher un message "Timeout, réessayez" si trop long.
- Alternative simple : afficher un message "Si le chargement dépasse 30 s, vérifiez votre connexion."

---

### 10. Docstrings complètes (API)

**Format attendu (Google ou NumPy style) :**
```python
def get_price(S: float, K: float, T: float, r: float, sigma: float, ...) -> float:
    """Prix d'une option européenne (Black-Scholes-Merton).

    Args:
        S: Prix spot de l'actif sous-jacent.
        K: Strike.
        T: Maturité en années.
        r: Taux sans risque (continu).
        sigma: Volatilité implicite (annuelle).
        option_type: 'call' ou 'put'.
        q: Dividend yield continu.

    Returns:
        Prix de l'option.

    Raises:
        ValueError: Si S, K <= 0 ou sigma <= 0.
    """
```

**Fichiers prioritaires :** `core/black_scholes.py`, `core/solvers.py`, `core/curves.py`, `instruments/options.py`, `instruments/exotic.py`, `services/vanilla_service.py`, `data/connector.py` (méthodes publiques).

---

## P3 — BONUS

### 11. Cache TTL configurable

**Fichier :** `data/connector.py`
- Ajouter un paramètre `cache_ttl_seconds: int = 30` au constructeur ou à une config.
- Utiliser cette valeur au lieu du 30 en dur.

---

### 12. FallbackDataConnector sans st.session_state

**Action :**
- Passer le connecteur synthétique en paramètre au lieu de le lire dans `st.session_state`.
- Ou : injecter une factory `get_synthetic_connector()` pour découpler de Streamlit.

---

### 13. Correlation ID pour le logging

**Action :**
- Générer un `request_id = uuid.uuid4()` au début de chaque requête (ex. chargement ticker).
- Passer ce `request_id` dans les logs : `logger.info("...", extra={"request_id": request_id})`.
- Permet de tracer une requête à travers les couches.

---

## Récapitulatif par ordre d'exécution recommandé

1. **Validation inputs** (rapide, impact immédiat)
2. **Constantes nommées** (rapide)
3. **Garde sigma + IV raison** (rapide)
4. **Protocol connecteurs** (1–2 h)
5. **except Exception → exceptions ciblées** (2–4 h, le plus long)
6. **Découpage vanilla_helpers** (2–3 h)
7. **ACT/365 vraies dates** (1–2 h)
8. **Tests Fallback, Heston, MC** (2–3 h)
9. **Timeout spinners** (30 min)
10. **Docstrings** (1–2 h)
11. **Cache TTL, Fallback sans session, correlation ID** (optionnel)

---

## Estimation temps total

| Priorité | Temps estimé |
|----------|--------------|
| P0 | 4–6 h |
| P1 | 5–8 h |
| P2 | 4–6 h |
| P3 | 1–2 h |
| **Total** | **14–22 h** |

En ciblant P0 + P1 + une partie de P2, tu peux viser 18/20 en ~12–15 h de travail.
