# Vérification du Pricer - Call AAPL ~8 jours

## Contexte

Analyse mathématique et financière du pricer sur un exemple concret : **Call AAPL ATM** avec expiration la plus proche de 8 jours (9 jours dans les données synthétiques).

---

## Données de l'exemple

| Paramètre | Valeur |
|-----------|--------|
| Ticker | AAPL |
| Spot | $175.00 |
| Expiration | 2026-02-27 (9 jours) |
| Strike | $175.00 (ATM) |
| T | 0.024658 ans |
| r | 3.48% |
| q | 0.50% |
| Bid | $3.70 |
| Ask | $3.80 |
| Mid | $3.75 |
| IV marché | 33.53% |

---

## Contrôles effectués

### 1. Round-trip IV ✓
BS(IV(Mid)) = $3.7500 ≈ Mid $3.7500

L'IV solver retrouve correctement la volatilité implicite depuis le prix mid, et le prix Black-Scholes avec cette IV redonne bien le mid.

### 2. Put-Call Parity ✓
- C - P (marché) = $0.1550
- C - P (théorie) = S·e^(-qT) - K·e^(-rT) = $0.1287
- Écart : $0.0263

L'écart est dû au spread bid-ask sur les données synthétiques (call et put générés indépendamment avec spread ±1.25%). En pratique, la parité est respectée par construction dans le modèle BSM ; l'écart vient des données bid/ask.

### 3. Bornes du prix call ✓
- Borne inf : max(S·e^(-qT) - K·e^(-rT), 0) = $0.1287
- Mid : $3.75
- Borne sup : S·e^(-qT) = $174.98

### 4. Delta call ✓
Delta = 0.5160 ∈ [0, e^(-qT)] = [0, 0.9999]

### 5. Gamma ✓
Gamma = 0.043257 > 0 (convexité positive)

### 6. Vega ✓
Vega = 0.109526 > 0 (sensibilité positive à la volatilité)

### 7. Theta ✓
Theta = -0.211050 < 0 (décroissance du temps pour un long call)

### 8. Delta C - Delta P = e^(-qT) ✓
Delta_call - Delta_put = 0.9999 ≈ e^(-qT) = 0.9999

### 9. Gamma call = Gamma put ✓
Identité vérifiée : 0.043257

### 10. Intrinsic ≤ BS_Price ≤ Spot ✓
$0 ≤ $3.7383 ≤ $175.00

---

## Conclusion

**Tous les contrôles mathématiques et financiers passent.**

Le pricer est cohérent :
- Formules Black-Scholes correctes
- IV solver fiable
- Grecs cohérents (signes, bornes, relations put-call)
- Bornes de prix respectées

### Script de vérification

Exécuter : `python scripts/verify_pricer_aapl.py`

---

## Référence

- Vérification effectuée le 16 février 2026
- Données : synthetic_data.xlsx (SVI Gatheral)
- Rapport d'audit complet : `docs/AUDIT_PRICER_ET_DONNEES.md`
