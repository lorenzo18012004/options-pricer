# Options Pricer

Pricer d'options développé from scratch — projet pédagogique et portfolio pour un stage en finance quantitative.

**Lorenzo Philippe** · M1 Portfolio Management, IAE Paris Est

[![GitHub](https://img.shields.io/badge/GitHub-options--pricer-blue?logo=github)](https://github.com/lorenzo18012004/options-pricer)

---

## Fonctionnalités

| Produit | Modèles / Méthodes |
|---------|--------------------|
| **Options vanilles** | Black-Scholes-Merton, Heston, Monte Carlo, Binomial (CRR), SVI |
| **Stratégies de volatilité** | Straddle, Strangle |
| **Options barrières** | Monte Carlo + Brownian bridge (knock-in/out) |
| **Swaps de taux** | Bootstrap courbe Treasury, NPV, DV01, hedge sizing |

- **Données live** : Yahoo Finance (spot, options, vol, Treasury)
- **Conventions** : ACT/365, jours calendaires
- **Greeks** : Delta, Gamma, Vega, Theta, Rho + ordre 2 (Vanna, Volga, Charm, etc.)
- **P&L attribution** : décomposition Delta, Gamma, Vega, Theta, Vanna

---

## Installation

```bash
git clone https://github.com/lorenzo18012004/options-pricer.git
cd options-pricer
pip install -r requirements.txt
```

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur (par défaut `http://localhost:8501`).

---

## Tests

```bash
pytest test_quant_regression.py -v
```

Ou via le script de release :

```bash
python run_release_checks.py
```

**42 tests** couvrent : Black-Scholes, solveur IV, binomial, Monte Carlo, Heston, barrières, courbe de taux, swap, DataCleaner, SVI.

---

## Structure du projet

```
├── app.py                 # Point d'entrée Streamlit
├── core/                  # Cœur quantitatif
│   ├── black_scholes.py   # BSM, Greeks
│   ├── heston.py          # Volatilité stochastique
│   ├── solvers.py         # Solveur IV (Newton-Raphson, Brent)
│   └── curves.py         # Courbe de taux (bootstrap)
├── models/
│   ├── trees.py          # Arbre binomial CRR
│   ├── monte_carlo.py    # Monte Carlo
│   └── surfaces.py      # SVI
├── instruments/
│   ├── options.py        # Options vanilles
│   ├── exotic.py        # Options barrières
│   └── rates.py         # Swaps
├── data/
│   ├── connector.py     # Yahoo Finance
│   └── cleaner.py       # Nettoyage chaînes
├── pages/                # Interface Streamlit
├── services/             # Services marché
├── docs/                 # Documentation (Word, PowerPoint)
└── static/               # CV (optionnel, placer cv.pdf)
```

---

## Documentation

- **Word** : `docs/Pricer_Documentation_Complete.docx` — description détaillée du projet
- **PowerPoint** : `docs/Pricer_Resume.pptx` — synthèse exécutive

Génération :

```bash
python generate_documentation.py
```

---

## Stack technique

- Python, NumPy, SciPy, Pandas
- yfinance (données)
- Streamlit (UI), Plotly (graphiques)
- pytest (tests)

---

## Essayer l'app

Clonez le repo et lancez en local (Yahoo Finance bloque les requêtes depuis les serveurs cloud) :

```bash
git clone https://github.com/lorenzo18012004/options-pricer.git
cd options-pricer
pip install -r requirements.txt
streamlit run app.py
```

---

## Contact

[LinkedIn](https://www.linkedin.com/in/lorenzo-philippe-9584a82b1)
