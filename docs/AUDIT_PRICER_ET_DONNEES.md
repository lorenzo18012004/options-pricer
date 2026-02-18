# Audit du Pricer et des Données Synthétiques

## 1. AUDIT GLOBAL DU PRICER

### 1.1 Black-Scholes (`core/black_scholes.py`) ✅
- **d1, d2** : Formules correctes avec dividend yield q
- **Call/Put** : Formules BSM standard
- **Delta, Gamma, Vega, Theta, Rho** : Corrects
- **Vanna, Volga, Charm, Speed, Zomma, Color** : Formules vérifiées
- **P&L Attribution** : Décomposition Delta/Gamma/Vega/Theta/Vanna cohérente

### 1.2 IV Solver (`core/solvers.py`) ✅
- Newton-Raphson avec vega comme dérivée : correct
- Fallback Brent : robuste
- Bornes intrinsèques vérifiées

### 1.3 Options Barrières (`instruments/exotic.py`) ✅
- **Rubinstein-Reiner** : λ = 2(r-q)/σ² - 1 correct
- Down-and-out/in call et put : formules analytiques OK
- Monte Carlo avec Brownian Bridge : standard

### 1.4 Monte Carlo (`models/monte_carlo.py`) ✅
- Schéma Euler : drift (r - q - 0.5σ²)dt correct
- Antithetic variates : implémenté
- Barrier hit detection : Brownian bridge correct

### 1.5 Arbre Binomial (`models/trees.py`) ✅
- CRR : u, d, p corrects avec (r - q)
- Backward induction : standard

### 1.6 Surface SVI (`models/surfaces.py`) ✅
- Paramétrisation raw Gatheral : w(k) = a + b[ρ(k-m) + √((k-m)²+σ²)]
- Calibration multi-start : robuste

### 1.7 Digital Option (`instruments/exotic.py`) ✅ (corrigé)
- **Correction** : Ajout du paramètre q (dividend yield) dans d1/d2
- **Formule** : d1 = [ln(S/K) + (r - q + 0.5σ²)T] / (σ√T), d2 = d1 - σ√T

---

## 2. ANALYSE DES DONNÉES SYNTHÉTIQUES

### 2.1 Sheet "spot"
| Colonne | Valeur type | Interprétation | Vérification |
|---------|-------------|----------------|--------------|
| ticker | AAPL, MSFT, TSLA, SPY | Symboles | OK |
| spot | 175, 420, 250, 580 | Prix réalistes (fév 2026) | OK |
| base_vol | 0.22, 0.20, 0.45, 0.15 | Vol de base (22%, 20%, 45%, 15%) | TSLA 45% élevé mais cohérent (volatil) |
| div_yield | 0.005, 0.007, 0, 0.012 | Dividend yield (0.5%, 0.7%, 0%, 1.2%) | SPY 1.2% réaliste, TSLA 0% OK |

### 2.2 Sheet "options"
| Colonne | Valeur type | Interprétation | Vérification |
|---------|-------------|----------------|--------------|
| bid, ask | Prix BSM ± 1.25% | Spread 2.5% | Réaliste pour options liquides |
| iv | SVI(k) 10-60% | Smile Gatheral | OK |
| volume | 200-600 (ATM max) | Plus liquide à l'ATM | OK |
| openInterest | volume × 2.5 | OI > volume typique | OK |
| lastPrice | (bid+ask)/2 | Mid | OK |

**Cohérence Put-Call** : Prix générés par BSM avec même IV → parité respectée par construction.

### 2.3 Sheet "treasury"
| Maturité | Rate | Interprétation |
|----------|------|----------------|
| 0.25-30Y | 3.5%-4.6% | Courbe normale (pente positive) |

### 2.4 Sheet "expirations"
- 18 maturités, vendredis
- Jours calendaires cohérents

### 2.5 Sheet "hist_vol"
| window_days | Formule | Interprétation |
|-------------|---------|----------------|
| 10-252 | base_vol × (0.92 + 0.16×exp(-w/60)) | Court terme légèrement plus volatile | OK |

**Exemple AAPL** : 20d ≈ 23%, 252d ≈ 21% → term structure HV raisonnable.

### 2.6 Points d'attention
1. **Volume/OI identiques call et put** : En pratique souvent différents (puts plus tradés en crise). Acceptable pour synthétique.
2. **Spread 2.5% constant** : En pratique spread varie (OTM plus large). Mineur.
3. **Strikes 0.75S - 1.25S** : Couverture standard.

---

## 3. RÉSUMÉ

| Composant | Statut | Action |
|-----------|--------|--------|
| Black-Scholes | ✅ | - |
| IV Solver | ✅ | - |
| Barrières | ✅ | - |
| Monte Carlo | ✅ | - |
| Arbre | ✅ | - |
| SVI | ✅ | - |
| Digital Option | ✅ | Corrigé (q ajouté) |
| Données synthétiques | ✅ | Cohérentes |

**44 tests passent** → Pricer globalement fiable.
