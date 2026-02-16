# Analyse : Tolérance Put-Call Parity & Audit des Inputs

## 1. Logique de la Tolérance

### Formule actuelle (vanilla_helpers.py, lignes 787-790)

```python
call_spread = row["call_ask"] - row["call_bid"]   # Spread $ du call
put_spread = row["put_ask"] - row["put_bid"]      # Spread $ du put
tolerance = max(0.5 * (call_spread + put_spread), 0.02)
```

**Condition de breach** : `breach_count` compte un strike seulement si `|Breach $| > tolerance`.

### Interprétation

- **Demi-spread (C-P)** : L'incertitude sur `C - P` (mid) est approximativement la moitié du spread combiné. Si le call a un spread de $0.50 et le put de $0.50, l'écart type sur `C-P` est ~$0.50. On tolère donc les écarts inférieurs à cette marge.
- **Plancher $0.02** : Évite de compter comme breach des écarts < 2 cts (bruit numérique).

### Risque : spread large

**Oui, la tolérance peut masquer des anomalies si le spread est large.**

Exemple : Call spread 15%, Put spread 15%, option à $20 → call_spread ≈ $3, put_spread ≈ $3 → `tolerance = max(3, 0.02) = $3`. Une vraie anomalie de $1.50 serait ignorée.

**Recommandation** : Plafonner la tolérance, par ex. `tolerance = min(demi_spread, 0.10)` pour ne jamais tolérer plus de $0.10 de breach pour des raisons de spread seul.

**Implémenté** : `tolerance = min(max(0.5 * (call_spread + put_spread), 0.02), 0.10)`

---

## 2. Audit des Inputs

### Risk-Free Rate (r)

| Aspect | Détail |
|--------|--------|
| **Source** | `DataConnector.get_risk_free_rate(T)` |
| **Fichier** | `data/connector.py` lignes 386-417 |
| **Méthode** | Courbe Treasury US live (^IRX, ^FVX, ^TNX, ^TYX) → bootstrap zero-curve → interpolation pour T |
| **Statique ?** | Non. Mis à jour dynamiquement via Yahoo Finance. Cache 30 s. |
| **Par actif ?** | Non. Même courbe pour tous les actifs (Treasury US). |

### Dividendes (q)

| Aspect | Détail |
|--------|--------|
| **Source** | `load_market_snapshot(ticker)` → `market_data["dividend_yield"]` |
| **Fichier** | `data/connector.py` lignes 168-183, `services/market_service.py` ligne 21 |
| **Méthode** | `info.get('dividendYield')` de yfinance, avec fallback `dividendRate/spot` si yield > 15% |
| **Statique ?** | Non. Récupéré par ticker. Cache 30 s. |
| **Par actif ?** | Oui. Chaque ticker a son propre dividend yield. |

**Conclusion** : r et q sont dynamiques et par actif. Aucun fix nécessaire pour les rendre dynamiques.

---

## 3. Vérification de la Parité EU Theory

### Formule utilisée (ligne 779)

```python
cp_theory_eu = spot_eff * np.exp(-div_yield * T) - K_pc * np.exp(-rate * T)
```

Équivalent à : **C - P = S·e^(-qT) - K·e^(-rT)**

### Valeur actuelle du strike : K·e^(-rT)

- **Formule** : `K_pc * np.exp(-rate * T)`
- **Convention** : Taux continu (r en décimal, T en années).
- **Correct** : Oui. Pas de simplification incorrecte.

### Bornes américaines (lignes 781-783)

- **LB** : `S·e^(-qT) - K`
- **UB** : `S - K·e^(-rT)`

Conformes à la littérature (Hull, etc.).

---

## 4. Stress-test

Une modification temporaire est proposée dans le code : une checkbox "Stress-test: taux +5%" dans l'onglet Put-Call Parity. Quand activée, elle ajoute 5 points de pourcentage au taux pour forcer des breaches et vérifier que la détection fonctionne.
