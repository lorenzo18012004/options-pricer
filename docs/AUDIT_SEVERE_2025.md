# Audit sévère et complet du Pricer — Février 2025

## Note globale : **14,5 / 20**

---

## 1. MODÈLES (Black-Scholes, IV, Barrières) — 4/5

### Points positifs
- BSM correct avec dividend yield, Grecs 1er et 2nd ordre
- IV solver : guess adaptatif Brenner-Subrahmanyam, validation `np.isfinite`
- Barrières down et up : formules analytiques Rubinstein-Reiner
- Digital call pour up-and-out
- Probabilité de touch pour rebate

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Mineur** | Delta à expiry S=K | Retourne 0.0. Convention Haug : 0.5 pour ATM à l’échéance. Pas faux, mais discutable. |
| **Mineur** | sigma quasi-nul | Pas de garde `sigma < 1e-6` → risque d’overflow sur d1/d2. |
| **Mineur** | IV : retourne None sans raison | Pas de distinction hors bornes / non-convergence. Debug difficile. |
| **Mineur** | Rebate payé à l’expiration | Convention actuelle : rebate à T. Pas de rebate payé au moment du touch. |

---

## 2. DONNÉES & CONNECTEURS — 3,5/5

### Points positifs
- Feuille dividends chargée et utilisée pour les options américaines
- Fallback Yahoo → Synthétique
- DataCleaner : `fillna(0)` pour volume/openInterest
- Retry avec backoff (3 tentatives, 1s, 2s, 4s)
- Timeout et retries configurés sur yfinance
- Treasury : 2Y, 3Y, 7Y interpolés
- Séparation stricte Excel / Yahoo

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | `except Exception` dans le connector | ~15 occurrences. Erreurs réseau, KeyError, etc. traitées de la même façon. |
| **Moyen** | Cache TTL : 30s fixe | Non configurable. |
| **Mineur** | FallbackDataConnector : `st.session_state` | Couplage fort à Streamlit. |
| **Mineur** | get_expirations : retry 2 fois | `time.sleep(1.5)` bloque le thread. Pas de timeout global. |

---

## 3. SWAP & RATES — 4/5

### Points positifs
- Day count : 30/360, ACT/365, ACT/360
- Accrual fraction correcte pour la jambe fixe
- Par rate avec day count
- Treasury enrichi (2Y, 3Y, 7Y)

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | ACT/365 : approximation | `days_approx = (t_i - t_prev) * 365`. Pas de vraies dates calendaires. |
| **Moyen** | DV01 : shift sur les nodes | Interpolation CubicSpline entre nodes. Shift parallèle approximatif. |
| **Mineur** | Pas de stub period | Schedule `np.arange(1/freq, T+1/freq, 1/freq)`. Pas de gestion des stubs courts. |

---

## 4. PAGES & UI — 3/5

### Points positifs
- Logique métier extraite dans `services/vanilla_service.py`
- Calibration put-call parity déplacée dans le service

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | vanilla_helpers.py : ~1155 lignes | Toujours très long. Logique UI et métier encore mélangées. |
| **Moyen** | `except Exception` dans les pages | vanilla_helpers, vanilla_page, volatility_page, barrier_page, swap_page. Erreurs avalées. |
| **Moyen** | Pas de loading timeout | Spinners sans limite. Utilisateur peut rester bloqué. |
| **Mineur** | Validation des inputs utilisateur | Ticker vide, strike négatif : comportement imprévisible. |
| **Mineur** | build_expiration_options : `except Exception` | Ligne 57 market_service : `continue` silencieux. |

---

## 5. TESTS — 4/5

### Points positifs
- 51 tests passent
- Edge cases IV : NaN, Inf, guess ATM
- Barrières up : parité et bornes
- Intégration : load synthetic → price → roundtrip
- Swap day count testé
- Treasury : test sur 2Y, 3Y, 7Y

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Mineur** | Pas de tests Heston edge cases | v0=0, kappa=0, etc. |
| **Mineur** | Pas de tests Monte Carlo barrières | Convergence MC vs analytique. |
| **Mineur** | Pas de tests FallbackDataConnector | Comportement en cas d’échec Yahoo. |

---

## 6. ARCHITECTURE — 3,5/5

### Points positifs
- `DataConnectorProtocol` défini
- `vanilla_service` : logique métier extraite
- Exceptions : `NetworkError`, `DataError`

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Protocol non utilisé pour le typage | DataConnector et SyntheticDataConnector n’implémentent pas explicitement le protocol. |
| **Moyen** | Duplication div_yield | Connector et Synthetic : logiques différentes pour `get_dividend_yield_forecast`. |
| **Mineur** | Magic numbers | 0.01, 1e-8, 1e-10 dispersés. Pas de constantes nommées. |
| **Mineur** | Type hints partiels | Core et services typés. Pages et instruments partiellement. |

---

## 7. ROBUSTESSE — 3,5/5

### Points positifs
- Validation `np.isfinite` dans BlackScholes et IVsolver
- Retry + timeout sur Yahoo
- Fallback automatique

### Problèmes restants

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | ~33 `except Exception` dans le projet | Connector (15), synthetic (4), heston (3), barrier_page (3), pages (4), exotic (1), surfaces (1), etc. |
| **Mineur** | Pas de correlation ID | Logging sans trace de requête. |
| **Mineur** | sensitivity_to_barrier : `except Exception` | exotic.py ligne 280 : toute exception → np.nan. |

---

## GRILLE DE NOTATION SÉVÈRE (post-améliorations)

| Critère | Note /5 | Commentaire |
|---------|---------|-------------|
| **Modèles** | 4.0 | BSM solide, IV adaptatif, barrières up/down complètes |
| **Données** | 3.5 | Dividends OK, fallback, retry, NaN gérés. `except Exception` reste. |
| **Rates/Swap** | 4.0 | Day count, Treasury enrichi. ACT/365 approximatif. |
| **Architecture** | 3.5 | Protocol défini, service extrait. Protocol peu utilisé. |
| **Robustesse** | 3.5 | Validation inputs, retry, fallback. Beaucoup d’`except Exception`. |
| **Tests** | 4.0 | 51 tests, edge cases, intégration. Manque Heston, MC barrières. |
| **Produits** | 4.0 | Vanilles, vol, barrières, swap, dividendes discrets |
| **UX** | 3.0 | Pages longues, pas de timeout loading |
| **Total** | **14,5 / 20** | |

---

## VERDICT

Pricer **solide pour un stage**, avec une base quantitative correcte et des améliorations nettes par rapport à l’audit initial (11/20).

### Ce qui a bien été fait
- Dividends discrets pour les américaines
- Fallback Yahoo → Synthétique
- IV solver adaptatif
- Barrières up analytiques
- Day count swap
- Treasury enrichi
- Tests edge cases et intégration
- Extraction de la logique métier

### Ce qui reste à améliorer pour viser 16/20
1. Remplacer les `except Exception` restants par des exceptions ciblées
2. Réduire la taille de `vanilla_helpers` (découper en sous-modules)
3. Utiliser le Protocol pour le typage des connecteurs
4. Ajouter un timeout sur les spinners
5. ACT/365 avec vraies dates pour le swap

### Pour viser 18/20
- Tests FallbackDataConnector
- Tests Heston edge cases
- Validation stricte des inputs utilisateur
- Constantes nommées pour les magic numbers
- Documentation API (docstrings complètes)
