# Vérification de cohérence : Vol Strategies, Barriers, Swap

## Résumé

Script de vérification : `python scripts/verify_vol_strategies_barriers_swap.py`

---

## 1. Volatility Strategies (Straddle)

### Contrôles effectués
- **Premium BSM** : C + P cohérent avec Black-Scholes
- **Breakevens** : K - premium (lower), K + premium (upper) → payoff = premium
- **Position Delta** : Straddle ATM ~ 0 (delta call + delta put)
- **Monte Carlo vs BSM** : Prix MC dans l’intervalle de confiance BSM

### Exemple AAPL
- Premium BSM ≈ $7.35
- Breakevens : [167.65, 182.35]
- Position delta ≈ 0.03
- MC ≈ BSM (SE ≈ 0.04)

---

## 2. Barrier Options

### Contrôles effectués
- **Parité In/Out** : C_down-and-out + C_down-and-in = C_vanilla
- **Barrier ≤ Vanilla** : Prix barrière ≤ prix vanille
- **Analytique vs MC** : Rubinstein-Reiner vs Monte Carlo cohérents

### Exemple S=100, K=100, H=90, T=0.25
- DO + DI ≈ Vanilla (écart < $0.10)
- DO ≤ Vanilla
- Analytique ≈ MC

---

## 3. Interest Rate Swap

### Contrôles effectués
- **Swap au par** : NPV = 0 au taux par
- **Payer + Receiver** : NPV_payer + NPV_receiver = 0
- **PV(Floating)** : N × (1 - DF(T)) à l’initiation
- **DV01** : Cohérence avec shift +1 bp
- **Signe Payer** : fixed < par → NPV > 0 ; fixed > par → NPV < 0

### Exemple N=$1M, T=5Y, freq=2
- Par rate ≈ 4.41 %
- NPV par ≈ 0
- DV01 ≈ $449
- Payer (fixed 4 % < par 4.41 %) → NPV > 0

---

## Conclusion

Tous les contrôles passent pour les trois modules.
