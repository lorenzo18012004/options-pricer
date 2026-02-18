# Audit sévère et complet du Pricer (post-nettoyage)

## Note globale : **11 / 20**

---

## 1. BLACK-SCHOLES & IV SOLVER (3/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Mineur** | Delta à expiry S=K | `get_delta(S=K, T=0, "call")` retourne 0.0. Convention discutable : certains (Haug) utilisent 0.5 pour l'ATM à l'échéance. Pas faux, mais incohérent avec la continuité. |
| **Moyen** | IV solver : pas de validation des inputs | `find_implied_vol(np.nan, ...)` ou `find_implied_vol(price, np.inf, ...)` peut planter ou retourner des résultats absurdes. Aucun `np.isfinite()` sur les entrées. |
| **Moyen** | IV solver : sigma_init=0.3 fixe | Pour une option deep ITM avec vol implicite 8%, Newton-Raphson part de 30% → convergence lente ou échec. Pas de guess adaptatif (Brenner-Subrahmanyam, ou vol ATM du smile). |
| **Mineur** | IV solver : retourne None silencieusement | L'appelant doit gérer. Pas de raison d'échec (hors bornes ? non-convergence ?). Debugging difficile. |
| **Mineur** | get_price avec sigma=0 et T>0 | Lève ValueError. Correct. Mais si sigma=1e-10 (quasi-nul), d1/d2 explosent → overflow possible. Pas de garde `sigma < 1e-6`. |

---

## 2. DONNÉES & CONNECTEURS (2/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Élevé** | Feuille dividends jamais chargée | Tu as ajouté une feuille "dividends" dans l'Excel. `synthetic.py` ne la charge pas. Code mort / données inutilisées. |
| **Élevé** | Pas de fallback Yahoo → Synthétique | Si Yahoo bloque (cloud, rate limit), l'app plante. Aucun fallback automatique vers SyntheticDataConnector. |
| **Moyen** | DataCleaner : NaN dans volume/OI | `(df['volume'] > 0) | (df['openInterest'] > 0)` : si volume=NaN, `NaN > 0` = False. Ligne supprimée même si OI=1000. Utiliser `df['volume'].fillna(0)` et `df['openInterest'].fillna(0)`. |
| **Moyen** | DataCleaner : spread_pct avec mid=0 | Après `df['mid'] = (bid+ask)/2`, si bid=ask=0.01, mid=0.01. Mais si une ligne a bid=0, ask=0 (avant filtrage), division par zéro. Le filtre `bid > min_bid` est avant, donc OK. Mais `spread_pct` peut être inf si bid/ask incohérents. |
| **Moyen** | Treasury : 4 points seulement | 0.25, 5, 10, 30Y. Pas de 2Y, 3Y, 7Y. Un swap 3Y est interpolé entre 0.25 et 5 → imprécision. |
| **Moyen** | Cache TTL : 30s fixe | Pas configurable. Pour un démo, OK. Pour un usage sérieux, bloquant. |
| **Mineur** | get_expirations : retry 2 fois | Bien. Mais `time.sleep(1.5)` bloque le thread. Pas de timeout global sur la requête. |
| **Mineur** | Connector : `except Exception` partout | 15+ occurrences. Une erreur réseau, une KeyError, une division par zéro → même traitement. Impossible de distinguer. |

---

## 3. BARRIÈRES (3/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Up-and-out/in : non implémentés analytiquement | Seul down-and-out et down-and-in ont Rubinstein-Reiner. Up-and-out/in = Monte Carlo uniquement. Pas de formule de type Haug pour up-barriers. |
| **Moyen** | sensitivity_to_barrier : `except Exception` | Ligne 217-218 : toute exception → np.nan. Une ValueError (barrier invalide) ou une KeyError (bug) sont traités pareil. |
| **Mineur** | Rebate : formule simplifiée | `rebate_pv = rebate * exp(-rT) * activation_rate`. Correct pour un rebate payé à l'expiration si touché. Mais si le rebate est payé au moment du touch (convention courante), il faudrait actualiser au temps de hit. Approximation. |
| **Mineur** | Pas de validation K vs H | Pour down-and-out call, si K < H (strike sous la barrière), la formule Rubinstein-Reiner peut donner des résultats bizarres. Pas de check. |

---

## 4. SWAP & RATES (2.5/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Pas de day count convention | 30/360, ACT/360, ACT/365 non gérés. Les `payment_dates` sont `np.arange(1/freq, T+1/freq, 1/freq)` – approximation. En production, les conventions sont cruciales. |
| **Moyen** | DV01 : shift sur les nodes uniquement | `shifted_rates = original_rates + 0.0001`. Les rates sont aux maturités de la courbe. L'interpolation CubicSpline entre les nodes ne reflète pas un vrai shift parallèle. Pour un swap 3Y avec nodes 0.25, 5, 10, 30, l'impact dépend de l'interpolation. |
| **Moyen** | pnl_scenario : mutation de curve.rates | `shifted_rates = self.curve.rates + shift`. Si `self.curve.rates` est une vue (référence), `+ shift` crée un nouveau array. OK. Mais si quelqu'un fait `curve.rates += shift` par erreur ailleurs, ça muterait la courbe. Pas de `copy()` explicite. |
| **Mineur** | Par rate : pas de check sum_df > 0 | `par_rate = (1 - df_final) / sum_df * payment_freq`. Si sum_df est 0 (courbe dégénérée), division par zéro. |
| **Mineur** | Bootstrap : pas de check df_T > 0 | `df_T = numerator / denominator`. Si denominator <= 0, df_T peut être négatif ou inf. Le `np.clip(df_T, 1e-10, 1.0)` est là, mais pas de log si ça arrive. |

---

## 5. PAGES & UI (2.5/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Pages trop longues | vanilla_helpers.py ~1150 lignes, vanilla_page.py ~300, volatility_page.py ~590. Logique métier mélangée à l'UI. Difficile à maintenir. |
| **Moyen** | `except Exception` dans les pages | vanilla_helpers.py ligne 834 : `except Exception: return spot_market, rate_init, q_init` – avale toute erreur et retourne des valeurs par défaut. Une bug silencieuse. |
| **Moyen** | Pas de loading state cohérent | Certains spinners sont courts, d'autres longs. Pas de timeout. L'utilisateur peut rester bloqué sans feedback. |
| **Mineur** | Pas de validation des inputs utilisateur | Si l'utilisateur entre un ticker vide, un strike négatif, etc., le comportement peut être imprévisible. |
| **Mineur** | Welcome modal : st.dialog avec @st.dialog | Dépend de la version Streamlit. Peut ne pas exister dans les anciennes versions. |

---

## 6. TESTS & COUVERTURE (3/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Pas de tests pour edge cases IV | IV avec price=0, sigma=0, T=0, S=K, etc. |
| **Moyen** | Pas de tests pour barrières up | Seul down-and-out/in est testé. |
| **Moyen** | Pas de tests d'intégration | Aucun test qui simule : load data → price → vérifier. |
| **Mineur** | Pas de tests pour SyntheticDataConnector | Si l'Excel change de format, rien ne casse. |
| **Mineur** | Pas de tests pour DataCleaner avec NaN | `filter_surface_quality` avec volume=NaN. |

---

## 7. ARCHITECTURE & CODE (2.5/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Moyen** | Pas d'interface commune pour les connecteurs | `DataConnector` et `SyntheticDataConnector` n'implémentent pas de protocol/ABC. `get_data_connector()` retourne une classe. Pas de typage. |
| **Moyen** | Duplication : div_yield logic | Connector et Synthetic ont chacun leur `get_dividend_yield_forecast`. Logique différente. Pas de source unique. |
| **Mineur** | Imports circulaires potentiels | core → black_scholes, curves. instruments → core, models. Si on ajoute des dépendances, risque de cycle. |
| **Mineur** | Pas de type hints partout | Certains fichiers en ont, d'autres non. Incohérent. |
| **Mineur** | Magic numbers | 0.01, 0.01, 1e-8, 1e-10 dispersés. Pas de constantes nommées. |

---

## 8. SÉCURITÉ & ROBUSTESSE (2/5)

### Problèmes identifiés

| Sévérité | Problème | Détail |
|----------|----------|--------|
| **Élevé** | Aucune validation des inputs | `BlackScholes.get_price(np.nan, ...)` → crash. `np.log(S/K)` avec S/K négatif → NaN. Pas de garde `np.isfinite(S)`. |
| **Moyen** | Pas de retry avec backoff | Yahoo Finance échoue ? Une seule tentative. Pas de retry exponentiel. |
| **Moyen** | Pas de timeout sur les requêtes | yfinance peut bloquer indéfiniment. Aucun timeout. |
| **Mineur** | Logging : pas de correlation ID | En cas d'erreur, difficile de tracer une requête précise. |

---

## SYNTHÈSE DES CHANGEMENTS À FAIRE

### Priorité 0 (Critique)
1. **Charger la feuille dividends** dans synthetic.py ou la supprimer de l'Excel (pas de données orphelines).
2. **DataCleaner** : gérer les NaN dans volume/openInterest avec `fillna(0)`.
3. **Validation des inputs** : `np.isfinite(S)`, `np.isfinite(K)` dans BlackScholes et IVsolver.

### Priorité 1 (Important)
4. **Fallback Yahoo → Synthétique** : si DataConnector échoue, basculer automatiquement.
5. **IV solver** : guess initial adaptatif (Brenner-Subrahmanyam ou vol ATM).
6. **Barrières up-and-out/in** : implémenter analytiquement (Rubinstein-Reiner / Haug).
7. **Remplacer `except Exception`** par des exceptions spécifiques (NetworkError, DataError, etc.).

### Priorité 2 (Souhaitable)
8. **Treasury** : ajouter 2Y, 3Y, 7Y ou interpoler.
9. **Swap** : day count convention.
10. **Tests** : edge cases IV, barrières up, intégration.
11. **Interface** : protocol pour DataConnector et SyntheticDataConnector.

### Priorité 3 (Nice to have)
12. **Retry** avec backoff sur Yahoo.
13. **Timeout** sur les requêtes.
14. **Refactor** des pages : extraire la logique métier.
15. **Type hints** partout.

---

## GRILLE DE NOTATION SÉVÈRE

| Critère | Note /5 | Commentaire |
|---------|---------|-------------|
| **Modèles** | 3.0 | BSM correct, mais IV solver fragile, barrières up manquantes |
| **Données** | 2.0 | Feuille dividends inutilisée, pas de fallback, NaN mal gérés |
| **Rates/Swap** | 2.5 | Pas de day count, DV01 approximatif |
| **Architecture** | 2.5 | Pas d'interface, pages trop longues |
| **Robustesse** | 2.0 | except Exception, pas de validation inputs |
| **Tests** | 3.0 | 44 tests OK, mais pas d'edge cases |
| **Produits** | 3.0 | Vanilles, vol, barrières, swap. Pas de dividendes discrets |
| **UX** | 2.5 | Streamlit, pas de timeout, pas de feedback cohérent |
| **Total** | **11 / 20** | |

---

## VERDICT

Pricer **correct pour un stage**, avec une base mathématique solide. Mais dès qu'on cherche la petite bête :

- **Données** : feuille dividends inutilisée, pas de fallback, NaN mal gérés.
- **Robustesse** : `except Exception` partout, pas de validation des inputs.
- **Complétude** : barrières up manquantes, pas de day count pour le swap.
- **Architecture** : pas d'interface, pages trop longues.

Pour viser **14/20** : traiter P0 et P1. Pour **16/20** : ajouter P2 + refactor des pages.
