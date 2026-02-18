"""
Génère le fichier Excel des données synthétiques.

Lancement : python scripts/generate_synthetic_excel.py

Le fichier data/synthetic_data.xlsx est créé avec :
- Sheet "spot" : ticker, spot, base_vol, div_yield
- Sheet "options" : ticker, expiration, strike, type, bid, ask, iv, volume, oi
- Sheet "treasury" : maturity, rate
- Sheet "expirations" : ticker, expiration, days_to_exp
- Sheet "hist_vol" : ticker, window_days, hist_vol
- Sheet "dividends" : ticker, ex_date, amount (dates ex-dividende réalistes)

Smile : paramétrisation SVI de Gatheral (standard industrie)
  w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]  avec k = log(K/S)
  IV = √(w(k)/T)
Term structure : court terme +35% vs long terme
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.black_scholes import BlackScholes
from datetime import datetime, timedelta


def _svi_total_variance(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """
    SVI raw (Gatheral 2004) : w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
    Retourne la variance totale (pas la vol).
    """
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


def _smile_iv(strike: float, spot: float, atm_vol: float, T: float) -> float:
    """
    SVI de Gatheral - formule standard utilisée en pratique.
    Paramètres typiques equity : ρ négatif (skew), forme smile en U.
    """
    k = np.log(strike / spot)
    w_atm = (atm_vol ** 2) * T
    # Paramètres SVI calibrés pour reproduire un smile equity réaliste
    # a,b,ρ,m,σ : a+bσ√(1-ρ²) >= 0 pour w >= 0
    a = 0.5 * w_atm
    b = 0.15 * np.sqrt(w_atm)  # slope des ailes
    rho = -0.35  # skew equity (aile put > aile call)
    m = 0.02  # léger décalage
    sigma = 0.25  # courbure ATM
    w = _svi_total_variance(k, a, b, rho, m, sigma)
    iv = np.sqrt(max(w / T, 1e-8))
    return float(np.clip(iv, 0.10, 0.60))


def _generate_expirations(n: int = 20) -> list:
    today = datetime.now().date()
    expirations = []
    seen = set()
    for days in [7, 14, 21, 28, 35, 42, 49, 63, 77, 91, 105, 119, 140, 161, 182, 210, 238, 273, 365]:
        d = today + timedelta(days=days)
        days_until_friday = (4 - d.weekday()) % 7
        d = d + timedelta(days=days_until_friday)
        key = d.strftime("%Y-%m-%d")
        if d > today and key not in seen:
            expirations.append((key, (d - today).days))
            seen.add(key)
    expirations.sort(key=lambda x: x[1])
    return expirations[:n]


def _term_structure(T: float, base_vol: float) -> float:
    """Court terme +35% vs long terme."""
    return base_vol * (1.0 + 0.35 * np.exp(-5.0 * T))


def main():
    out_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_data.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = {
        "AAPL": (175.0, 0.22, 0.005),
        "MSFT": (420.0, 0.20, 0.007),
        "TSLA": (250.0, 0.45, 0.0),
        "SPY": (580.0, 0.15, 0.012),
        "DEFAULT": (100.0, 0.25, 0.02),
    }

    expirations = _generate_expirations(18)
    r = 0.04

    # Sheet spot
    spot_rows = []
    for ticker, (spot, base_vol, div_yield) in profiles.items():
        if ticker == "DEFAULT":
            continue
        spot_rows.append({"ticker": ticker, "spot": spot, "base_vol": base_vol, "div_yield": div_yield})
    df_spot = pd.DataFrame(spot_rows)

    # Sheet options
    option_rows = []
    for ticker, (spot, base_vol, div_yield) in profiles.items():
        if ticker == "DEFAULT":
            continue
        for exp_date, days in expirations:
            T = max(days / 365.0, 1e-4)
            atm_vol = _term_structure(T, base_vol)
            strikes = np.round(np.linspace(spot * 0.75, spot * 1.25, 45), 2)
            for K in strikes:
                iv = _smile_iv(K, spot, atm_vol, T)
                for opt_type in ["call", "put"]:
                    price = BlackScholes.get_price(spot, K, T, r, iv, opt_type, div_yield)
                    spread = price * 0.025
                    bid = max(0.01, price - spread / 2)
                    ask = price + spread / 2
                    vol = int(200 + 400 * np.exp(-0.5 * ((K / spot - 1) ** 2) * 50))
                    oi = int(vol * 2.5)
                    mid = (bid + ask) / 2
                    option_rows.append({
                        "ticker": ticker, "expiration": exp_date, "strike": K, "type": opt_type,
                        "bid": round(bid, 2), "ask": round(ask, 2), "lastPrice": round(mid, 2),
                        "iv": round(iv, 4), "volume": vol, "openInterest": oi,
                    })
    df_options = pd.DataFrame(option_rows)

    # Sheet expirations
    exp_rows = []
    for ticker in [t for t in profiles if t != "DEFAULT"]:
        for exp_date, days in expirations:
            exp_rows.append({"ticker": ticker, "expiration": exp_date, "days": days})
    df_exp = pd.DataFrame(exp_rows)

    # Sheet treasury (0.25, 0.5, 1, 2, 3, 5, 7, 10, 30 pour swaps précis)
    df_treasury = pd.DataFrame({
        "maturity": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0],
        "rate": [0.035, 0.038, 0.040, 0.042, 0.043, 0.044, 0.0445, 0.045, 0.046],
    })

    # Sheet hist_vol (comme Yahoo Finance - volatilité historique par fenêtre)
    hist_vol_rows = []
    for ticker, (spot, base_vol, div_yield) in profiles.items():
        if ticker == "DEFAULT":
            continue
        for window in [10, 20, 30, 60, 126, 252]:
            # Légère variation autour de base_vol pour réalisme
            hv = base_vol * (0.92 + 0.16 * np.exp(-window / 60))
            hist_vol_rows.append({"ticker": ticker, "window_days": window, "hist_vol": round(hv, 4)})
    df_hist_vol = pd.DataFrame(hist_vol_rows)

    # Sheet dividends : dates ex-dividende réalistes (calendrier typique US)
    # AAPL: ~fév, mai, août, nov | MSFT: idem | SPY: mensuel | TSLA: 0
    today = datetime.now().date()
    div_rows = []

    def _next_ex_date(base, month_offset):
        y, m = base.year, base.month
        m += month_offset
        while m > 12:
            m -= 12
            y += 1
        while m < 1:
            m += 12
            y -= 1
        try:
            return datetime(y, m, 15).date()
        except ValueError:
            return datetime(y, m, 28).date()

    # AAPL: ~0.24$ trimestriel (fév, mai, août, nov)
    for i in range(5):
        d = _next_ex_date(today, i * 3)
        if d >= today:
            div_rows.append({"ticker": "AAPL", "ex_date": d.strftime("%Y-%m-%d"), "amount": 0.24})
    # MSFT: ~0.75$ trimestriel
    for i in range(5):
        d = _next_ex_date(today, i * 3)
        if d >= today:
            div_rows.append({"ticker": "MSFT", "ex_date": d.strftime("%Y-%m-%d"), "amount": 0.75})
    # SPY: ~1.50$ trimestriel
    for i in range(5):
        d = _next_ex_date(today, i * 3)
        if d >= today:
            div_rows.append({"ticker": "SPY", "ex_date": d.strftime("%Y-%m-%d"), "amount": 1.50})
    df_dividends = pd.DataFrame(div_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_spot.to_excel(writer, sheet_name="spot", index=False)
        df_options.to_excel(writer, sheet_name="options", index=False)
        df_exp.to_excel(writer, sheet_name="expirations", index=False)
        df_treasury.to_excel(writer, sheet_name="treasury", index=False)
        df_hist_vol.to_excel(writer, sheet_name="hist_vol", index=False)
        df_dividends.to_excel(writer, sheet_name="dividends", index=False)

    print(f"Fichier créé : {out_path}")
    print(f"  - {len(df_spot)} tickers")
    print(f"  - {len(df_options)} lignes options")
    print(f"  - {len(expirations)} maturités")
    print(f"  - {len(df_dividends)} dividendes (ex_date, amount)")


if __name__ == "__main__":
    main()
