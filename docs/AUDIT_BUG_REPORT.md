# Audit Bug Report – Options Pricer (AAPL)

## Synthèse : Code vs Données Yahoo

| Point | Problème | Cause | Correction |
|-------|----------|-------|-------------|
| **1. Avg Spread 8.2%** | Code | Moyenne sur tous les strikes (80–120% moneyness) incluant les OTM illiquides | Afficher l’**ATM Spread** (95–105% moneyness) |
| **2. Volume vs OI (85%)** | Données Yahoo | Volume = intraday, OI = clôture veille | Tooltips explicatifs ajoutés |
| **3. SVI aberrant** | Code | Pas de seuil RMSE → SVI utilisé même si fit médiocre | Fallback sur IV marché si RMSE > 5% |
| **4. Taux 3.58%** | Données Yahoo | Taux dynamiques via Treasury (^IRX, ^FVX, ^TNX, ^TYX) | Tooltip ajouté ; niveau cohérent avec marché |

---

## 1. Avg Spread (8.2%)

**Problème :** Un spread de 8,2 % sur les options ATM d’Apple paraît trop élevé (typiquement < 1–2 %).

**Cause :** Le calcul utilisait `df['Spread_%'].mean()` sur **tous** les strikes entre 80 % et 120 % de moneyness. Les strikes très OTM (80 % ou 120 %) sont peu liquides et ont des spreads larges, ce qui augmente la moyenne.

**Correction :** Affichage de l’**ATM Spread** (moyenne sur les strikes 95–105 % moneyness), plus représentatif des options liquides. L’ancienne moyenne reste visible dans Data Quality avec un tooltip explicatif.

---

## 2. Volume vs OI (17 190 vs 20 121)

**Problème :** Volume ≈ 85 % de l’OI, ce qui semble atypique pour une méga-cap.

**Cause :** Comportement normal des données Yahoo Finance :
- **Volume** : volume du jour (intraday)
- **Open Interest** : clôture de la veille (mise à jour quotidienne)

Un ratio Volume/OI élevé indique une journée de trading active, pas une erreur.

**Correction :** Tooltips ajoutés sur les métriques Volume et OI pour préciser la source et la fréquence de mise à jour.

---

## 3. SVI Smoothed IV

**Problème :** Un mauvais fit SVI peut produire des IV aberrantes et fausser les Grecs.

**Cause :** Le code utilisait SVI dès que la calibration réussissait, sans vérifier la qualité du fit (RMSE).

**Correction :** Si le RMSE du fit SVI > 5 %, on revient automatiquement à l’IV marché. Tooltip mis à jour sur la checkbox et sur la métrique SVI fit RMSE.

---

## 4. Taux (3.58%)

**Problème :** 3,58 % paraît bas par rapport aux taux sans risque actuels.

**Cause :** Le taux n’est **pas** hardcodé. Il est calculé dynamiquement via :
- Courbe Treasury US (Yahoo : ^IRX 3M, ^FVX 5Y, ^TNX 10Y, ^TYX 30Y)
- Bootstrap des zero rates puis interpolation pour la maturité de l’option

Pour une option à 24 jours, le taux est interpolé/extrapolé à partir du T-Bill 3 mois. Les taux Treasury 3 mois en début 2026 sont autour de 3,6–3,7 %, donc 3,58 % est cohérent.

**Correction :** Tooltip ajouté sur la métrique Rate pour indiquer la source des données.
