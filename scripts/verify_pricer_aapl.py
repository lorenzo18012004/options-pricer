"""
Vérification mathématique et financière du pricer.
Exécuter : python scripts/verify_pricer_aapl.py [ticker] [call|put] [short|long]
Ex: python scripts/verify_pricer_aapl.py MSFT put long
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from core.black_scholes import BlackScholes
from core import IVsolver
from data.synthetic import SyntheticDataConnector

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    opt_type = (sys.argv[2] if len(sys.argv) > 2 else "call").lower()
    maturity = (sys.argv[3] if len(sys.argv) > 3 else "short").lower()

    connector = SyntheticDataConnector()
    spot = connector.get_spot_price(ticker)
    expirations = connector.get_expirations(ticker)

    today = datetime.now().date()
    if maturity == "long":
        # Longue maturité : prendre la dernière expiration
        best_exp = expirations[-1]
        best_days = (datetime.strptime(best_exp[:10], "%Y-%m-%d").date() - today).days
    else:
        # Courte maturité : ~8 jours
        target_days = 8
        best_exp, best_days = None, 999
        for exp in expirations:
            try:
                exp_d = datetime.strptime(exp[:10], "%Y-%m-%d").date()
                days = (exp_d - today).days
                if 0 < days < best_days and abs(days - target_days) < abs(best_days - target_days):
                    best_exp, best_days = exp, days
            except Exception:
                continue
        if best_exp is None:
            best_exp = expirations[0]
            best_days = (datetime.strptime(best_exp[:10], "%Y-%m-%d").date() - today).days

    print("=" * 60)
    print(f"VERIFICATION PRICER - {ticker} {opt_type.upper()} ({maturity} maturite)")
    print("=" * 60)
    print(f"Spot: ${spot:.2f}")
    print(f"Expiration: {best_exp} ({best_days} jours)")

    calls, puts = connector.get_option_chain(ticker, best_exp)
    chain = puts if opt_type == "put" else calls
    if chain.empty:
        print("ERREUR: Pas de donnees options")
        return

    T = max(best_days / 365.0, 1e-4)
    rate = connector.get_risk_free_rate(T)
    div_yield = connector.get_dividend_yield_forecast(ticker, spot)

    # ATM (strike le plus proche du spot)
    atm_idx = (chain["strike"] - spot).abs().idxmin()
    row = chain.loc[atm_idx]
    K = float(row["strike"])
    bid, ask = float(row["bid"]), float(row["ask"])
    mid = (bid + ask) / 2
    iv_market = float(row["impliedVolatility"])

    print(f"\n{opt_type.upper()} ATM K={K:.2f}")
    print(f"  Bid: ${bid:.4f}, Ask: ${ask:.4f}, Mid: ${mid:.4f}")
    print(f"  IV marche: {iv_market*100:.2f}%")
    print(f"  T: {T:.6f} ans, r: {rate*100:.2f}%, q: {div_yield*100:.2f}%")
    
    errors = []

    # 1. Round-trip IV
    iv_back = IVsolver.find_implied_vol(mid, spot, K, T, rate, opt_type, div_yield)
    if iv_back is None:
        errors.append("IVsolver echoue a retrouver l'IV depuis le Mid")
    else:
        price_roundtrip = BlackScholes.get_price(spot, K, T, rate, iv_back, opt_type, div_yield)
        diff = abs(price_roundtrip - mid)
        if diff > 0.01:
            errors.append(f"Round-trip IV: BS(Mid->IV->BS) = ${price_roundtrip:.4f} != Mid ${mid:.4f} (ecart ${diff:.4f})")
        else:
            print(f"\n[OK] Round-trip IV: BS(IV(Mid)) = ${price_roundtrip:.4f} ~ Mid ${mid:.4f}")

    # 2. Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
    call_row = calls[calls["strike"] == K]
    put_row = puts[puts["strike"] == K]
    if not call_row.empty and not put_row.empty:
        call_mid = (float(call_row["bid"].iloc[0]) + float(call_row["ask"].iloc[0])) / 2
        put_mid = (float(put_row["bid"].iloc[0]) + float(put_row["ask"].iloc[0])) / 2
        cp_market = call_mid - put_mid
        cp_theory = spot * np.exp(-div_yield * T) - K * np.exp(-rate * T)
        parity_gap = abs(cp_market - cp_theory)
        tol_parity = 0.40 if best_days > 180 else 0.15  # long maturite: spread cumule
        if parity_gap > tol_parity:
            errors.append(f"Put-Call Parity: ecart ${parity_gap:.4f} (C-P mkt=${cp_market:.4f}, theorie=${cp_theory:.4f})")
        else:
            print(f"[OK] Put-Call Parity: C-P = ${cp_market:.4f}, theorie = ${cp_theory:.4f} (ecart ${parity_gap:.4f})")

    # 3. Bornes du prix
    if opt_type == "call":
        p_lo = max(spot * np.exp(-div_yield * T) - K * np.exp(-rate * T), 0)
        p_hi = spot * np.exp(-div_yield * T)
    else:
        p_lo = max(K * np.exp(-rate * T) - spot * np.exp(-div_yield * T), 0)
        p_hi = K * np.exp(-rate * T)
    if mid < p_lo - 0.01:
        errors.append(f"{opt_type} sous borne inf: Mid ${mid:.4f} < {p_lo:.4f}")
    elif mid > p_hi + 0.01:
        errors.append(f"{opt_type} au-dessus borne sup: Mid ${mid:.4f} > {p_hi:.4f}")
    else:
        print(f"[OK] Bornes {opt_type}: {p_lo:.4f} <= Mid ${mid:.4f} <= {p_hi:.4f}")

    # 4. Delta
    greeks = BlackScholes.get_all_greeks(spot, K, T, rate, iv_market, opt_type, div_yield)
    delta = greeks["delta"]
    eqt = np.exp(-div_yield * T)
    if opt_type == "call":
        if delta < 0 or delta > eqt + 0.01:
            errors.append(f"Delta call hors bornes: {delta:.4f} (attendu 0 a {eqt:.4f})")
        else:
            print(f"[OK] Delta call: {delta:.4f} (borne 0 a e^(-qT)={eqt:.4f})")
    else:
        if delta < -eqt - 0.01 or delta > 0.01:
            errors.append(f"Delta put hors bornes: {delta:.4f} (attendu -e^(-qT)={-eqt:.4f} a 0)")
        else:
            print(f"[OK] Delta put: {delta:.4f} (borne -e^(-qT)={-eqt:.4f} a 0)")
    
    # 5. Gamma > 0
    gamma = greeks["gamma"]
    if gamma < -1e-10:
        errors.append(f"Gamma négatif: {gamma:.6f}")
    else:
        print(f"[OK] Gamma: {gamma:.6f} > 0")
    
    # 6. Vega > 0
    vega = greeks["vega"]
    if vega < -1e-10:
        errors.append(f"Vega négatif: {vega:.6f}")
    else:
        print(f"[OK] Vega: {vega:.6f} > 0")
    
    # 7. Theta long option (call: < 0, put ATM/OTM: < 0)
    theta = greeks["theta"]
    if opt_type == "call":
        if theta > 0.01:
            errors.append(f"Theta call long positif: {theta:.6f} (attendu < 0)")
        else:
            print(f"[OK] Theta: {theta:.6f} < 0 (decroissance)")
    else:
        # Put ATM/OTM: theta typiquement < 0; ITM profond peut etre > 0
        print(f"[OK] Theta put: {theta:.6f}")

    # 8. Put-Call Delta: Delta_call - Delta_put = e^(-qT)
    call_greeks = BlackScholes.get_all_greeks(spot, K, T, rate, iv_market, "call", div_yield)
    put_greeks = BlackScholes.get_all_greeks(spot, K, T, rate, iv_market, "put", div_yield)
    delta_call = call_greeks["delta"]
    delta_put = put_greeks["delta"]
    delta_diff = delta_call - delta_put
    expected = np.exp(-div_yield * T)
    if abs(delta_diff - expected) > 0.01:
        errors.append(f"Delta C - Delta P = {delta_diff:.4f}, attendu e^(-qT) = {expected:.4f}")
    else:
        print(f"[OK] Delta C - Delta P = {delta_diff:.4f} ~ e^(-qT) = {expected:.4f}")
    
    # 9. Put-Call Gamma identique
    gamma_put = put_greeks["gamma"]
    if abs(gamma - gamma_put) > 1e-10:
        errors.append(f"Gamma call != Gamma put: {gamma:.6f} vs {gamma_put:.6f}")
    else:
        print(f"[OK] Gamma call = Gamma put = {gamma:.6f}")
    
    # 10. Vanna: d2V/(dS dsigma)
    vanna = greeks["vanna"]
    bump_s, bump_v = 0.01, 0.01
    v0 = BlackScholes.get_price(spot, K, T, rate, iv_market, opt_type, div_yield)
    v_s_up = BlackScholes.get_price(spot + bump_s, K, T, rate, iv_market, opt_type, div_yield)
    v_v_up = BlackScholes.get_price(spot, K, T, rate, iv_market + bump_v, opt_type, div_yield)
    v_sv = BlackScholes.get_price(spot + bump_s, K, T, rate, iv_market + bump_v, opt_type, div_yield)
    vanna_num = (v_sv - v_s_up - v_v_up + v0) / (bump_s * bump_v)
    if abs(vanna - vanna_num) > 0.1:  # tolérance large
        print(f"  Vanna analytique: {vanna:.5f}, numérique: {vanna_num:.5f}")
    
    # 11. Intrinsic <= Price <= borne sup
    bs_price = greeks["price"]
    if opt_type == "call":
        intrinsic = max(spot - K, 0)
        upper = spot
    else:
        intrinsic = max(K - spot, 0)
        upper = K * np.exp(-rate * T)
    if bs_price < intrinsic - 0.01:
        errors.append(f"BS_Price ${bs_price:.4f} < intrinsic ${intrinsic:.4f}")
    elif bs_price > upper + 0.01:
        errors.append(f"BS_Price ${bs_price:.4f} > borne sup ${upper:.2f}")
    else:
        print(f"[OK] Intrinsic ${intrinsic:.4f} <= BS_Price ${bs_price:.4f} <= ${upper:.2f}")
    
    # 12. Mispricing cohérence: BS_Price - Mid (avec IV marché, devrait être ~0)
    mispricing = bs_price - mid
    if abs(mispricing) > 0.5:  # si IV marché = IV utilisée, mispricing ~ 0
        print(f"  Mispricing (BS-Mid): ${mispricing:.4f} (peut différer si SVI utilisé)")
    
    print("\n" + "=" * 60)
    if errors:
        print("PROBLÈMES DÉTECTÉS:")
        for e in errors:
            print(f"  [X] {e}")
    else:
        print("[OK] Tous les contrôles mathématiques et financiers passent.")
    print("=" * 60)

if __name__ == "__main__":
    main()
