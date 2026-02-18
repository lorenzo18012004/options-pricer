"""
Verification de coherence pour:
- Volatility Strategies (Straddle, Strangle)
- Barrier Options
- Interest Rate Swap

Executer: python scripts/verify_vol_strategies_barriers_swap.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from core import BlackScholes
from data.synthetic import SyntheticDataConnector
from instruments import BarrierOption, VanillaSwap, SwapCurveBuilder
from models import BinomialTree, MonteCarloPricer


def verify_volatility_strategies():
    """Verification Straddle / Strangle."""
    print("\n" + "=" * 60)
    print("1. VOLATILITY STRATEGIES (Straddle)")
    print("=" * 60)

    connector = SyntheticDataConnector()
    spot = connector.get_spot_price("AAPL")
    exps = connector.get_expirations("AAPL")
    exp = exps[0]
    calls, puts = connector.get_option_chain("AAPL", exp)
    div_yield = connector.get_dividend_yield_forecast("AAPL", spot)

    from datetime import datetime
    exp_d = datetime.strptime(exp[:10], "%Y-%m-%d").date()
    today = datetime.now().date()
    days = (exp_d - today).days
    T = max(days / 365.0, 1e-4)
    rate = connector.get_risk_free_rate(T)

    atm_idx = (calls["strike"] - spot).abs().idxmin()
    K = float(calls.loc[atm_idx, "strike"])
    call_mid = (float(calls.loc[atm_idx, "bid"]) + float(calls.loc[atm_idx, "ask"])) / 2
    put_row = puts[puts["strike"] == K]
    put_mid = (float(put_row["bid"].iloc[0]) + float(put_row["ask"].iloc[0])) / 2
    iv = float(calls.loc[atm_idx, "impliedVolatility"])

    premium_market = call_mid + put_mid
    call_bs = BlackScholes.get_price(spot, K, T, rate, iv, "call", div_yield)
    put_bs = BlackScholes.get_price(spot, K, T, rate, iv, "put", div_yield)
    premium_bsm = call_bs + put_bs

    errors = []

    # Straddle: C + P at same strike
    if abs(premium_bsm - (call_bs + put_bs)) > 0.01:
        errors.append(f"BSM premium incoherent: {premium_bsm} vs {call_bs + put_bs}")

    # Breakevens: K - premium (lower), K + premium (upper)
    lower_be = K - premium_bsm
    upper_be = K + premium_bsm
    payoff_at_lower = max(lower_be - K, 0) + max(K - lower_be, 0)
    payoff_at_upper = max(upper_be - K, 0) + max(K - upper_be, 0)
    if abs(payoff_at_lower - premium_bsm) > 0.01:
        errors.append(f"Lower breakeven incoherent: payoff {payoff_at_lower} vs premium {premium_bsm}")
    if abs(payoff_at_upper - premium_bsm) > 0.01:
        errors.append(f"Upper breakeven incoherent: payoff {payoff_at_upper} vs premium {premium_bsm}")

    # Position delta: straddle ATM ~ 0 (call delta + put delta)
    delta_c = BlackScholes.get_delta(spot, K, T, rate, iv, "call", div_yield)
    delta_p = BlackScholes.get_delta(spot, K, T, rate, iv, "put", div_yield)
    pos_delta = delta_c + delta_p
    if abs(pos_delta) > 0.15:
        errors.append(f"Straddle ATM delta should ~0, got {pos_delta:.4f}")

    # P&L at expiry: max(S-K,0)+max(K-S,0) - premium
    for s_test in [spot * 0.9, spot, spot * 1.1]:
        payoff = max(s_test - K, 0) + max(K - s_test, 0)
        pnl = payoff - premium_bsm
        if s_test == spot and abs(pnl + premium_bsm) > 0.01:
            errors.append(f"P&L at spot incoherent")

    # Monte Carlo vs BSM for straddle
    mc = MonteCarloPricer(spot, K, T, rate, iv, n_simulations=20000, n_steps=max(50, int(T * 252)), seed=42, q=div_yield)
    paths = mc._simulate_paths(use_antithetic=True)
    finals = paths[:, -1]
    straddle_payoff = np.maximum(finals - K, 0) + np.maximum(K - finals, 0)
    mc_price = float(np.mean(straddle_payoff * np.exp(-rate * T)))
    mc_se = float(np.std(straddle_payoff * np.exp(-rate * T)) / np.sqrt(len(finals)))
    if abs(mc_price - premium_bsm) > 3 * mc_se:
        errors.append(f"MC vs BSM: MC={mc_price:.4f}, BSM={premium_bsm:.4f}, 3*SE={3*mc_se:.4f}")

    if errors:
        for e in errors:
            print(f"  [X] {e}")
    else:
        print(f"  [OK] Straddle: premium BSM=${premium_bsm:.4f}, breakevens [{lower_be:.2f}, {upper_be:.2f}]")
        print(f"  [OK] Position delta ~0: {pos_delta:.4f}")
        print(f"  [OK] MC price ${mc_price:.4f} ~ BSM ${premium_bsm:.4f} (SE={mc_se:.4f})")
    return len(errors) == 0


def verify_barrier_options():
    """Verification Barrier: In/Out parity, barrier <= vanilla."""
    print("\n" + "=" * 60)
    print("2. BARRIER OPTIONS")
    print("=" * 60)

    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.04, 0.25, 0.0
    H = 90.0  # down barrier
    n_sims, n_steps = 30000, 126

    errors = []

    # Down-and-out + Down-and-in = Vanilla
    vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call", q)
    do_opt = BarrierOption(S, K, T, r, sigma, H, "down-and-out", "call", rebate=0, q=q)
    di_opt = BarrierOption(S, K, T, r, sigma, H, "down-and-in", "call", rebate=0, q=q)
    do_res = do_opt.price(n_simulations=n_sims, n_steps=n_steps, use_antithetic=True, seed=42)
    di_res = di_opt.price(n_simulations=n_sims, n_steps=n_steps, use_antithetic=True, seed=7)
    do_plus_di = do_res["price"] + di_res["price"]
    parity_gap = abs(do_plus_di - vanilla)
    if parity_gap > 0.15:
        errors.append(f"In/Out parity: DO+DI={do_plus_di:.4f} vs Vanilla={vanilla:.4f} (gap={parity_gap:.4f})")
    else:
        print(f"  [OK] In/Out parity: DO+DI=${do_plus_di:.4f} ~ Vanilla=${vanilla:.4f} (gap=${parity_gap:.4f})")

    # Barrier price <= Vanilla
    if do_res["price"] > vanilla + 0.01:
        errors.append(f"Down-and-out price {do_res['price']:.4f} > Vanilla {vanilla:.4f}")
    else:
        print(f"  [OK] Barrier <= Vanilla: DO=${do_res['price']:.4f} <= Vanilla=${vanilla:.4f}")

    # Analytical vs MC for down-and-out call
    try:
        ana = do_opt.price(method="analytical")
        ana_price = ana["price"]
        mc_ana_gap = abs(ana_price - do_res["price"])
        if mc_ana_gap > 0.20:
            errors.append(f"Analytical vs MC: analytical={ana_price:.4f}, MC={do_res['price']:.4f}")
        else:
            print(f"  [OK] Analytical Rubinstein-Reiner: ${ana_price:.4f} ~ MC ${do_res['price']:.4f}")
    except NotImplementedError:
        pass

    return len(errors) == 0


def verify_interest_rate_swap():
    """Verification Interest Rate Swap: par rate, NPV, DV01."""
    print("\n" + "=" * 60)
    print("3. INTEREST RATE SWAP")
    print("=" * 60)

    connector = SyntheticDataConnector()
    mats, rates = connector.get_treasury_par_curve()
    curve = SwapCurveBuilder.build_from_market_data(
        {}, {}, {float(T): float(r) for T, r in zip(mats, rates)}
    )

    notional = 1_000_000
    maturity = 5.0
    payment_freq = 2

    errors = []

    # Par swap: NPV = 0
    par = VanillaSwap(notional, 0.0, payment_freq, maturity, curve, "payer").par_rate()
    swap_at_par = VanillaSwap(notional, par, payment_freq, maturity, curve, "payer")
    npv_par = swap_at_par.npv()
    if abs(npv_par) > 10.0:
        errors.append(f"Par swap NPV should ~0, got {npv_par:.2f}")
    else:
        print(f"  [OK] Par swap NPV ~ 0: ${npv_par:.2f} (par rate={par*100:.3f}%)")

    # Payer vs Receiver: NPV_payer = -NPV_receiver (same fixed rate)
    fixed = 0.04
    payer = VanillaSwap(notional, fixed, payment_freq, maturity, curve, "payer")
    receiver = VanillaSwap(notional, fixed, payment_freq, maturity, curve, "receiver")
    if abs(payer.npv() + receiver.npv()) > 1.0:
        errors.append(f"Payer + Receiver NPV should = 0: {payer.npv()} + {receiver.npv()}")
    else:
        print(f"  [OK] Payer NPV + Receiver NPV = 0: ${payer.npv():,.0f} + ${receiver.npv():,.0f}")

    # PV(Floating) = N * (1 - DF(T))
    df_T = curve.get_discount_factor(maturity)
    pv_float_theory = notional * (1 - df_T)
    irs = payer.swap
    pv_float = irs.get_floating_leg_pv()
    if abs(pv_float - pv_float_theory) > 1.0:
        errors.append(f"PV Float: got {pv_float:.2f}, theory N*(1-DF)={pv_float_theory:.2f}")
    else:
        print(f"  [OK] PV(Floating) = N*(1-DF): ${pv_float:,.0f}")

    # DV01: payer loses when rates rise
    dv01 = payer.dv01()
    from core.curves import YieldCurve
    curve_up = YieldCurve(curve.maturities, curve.rates + 0.0001)
    swap_up = VanillaSwap(notional, fixed, payment_freq, maturity, curve_up, "payer")
    pnl_1bp = swap_up.npv() - payer.npv()
    if abs(abs(pnl_1bp) - dv01) > 5.0:
        errors.append(f"DV01 vs 1bp finite-diff: DV01={dv01:.2f}, 1bp P&L={pnl_1bp:.2f}")
    else:
        print(f"  [OK] DV01 consistency: DV01=${dv01:,.0f}, 1bp P&L=${pnl_1bp:,.0f}")

    # Payer: fixed > par => NPV < 0
    if fixed > par and payer.npv() >= 0:
        errors.append(f"Payer with fixed > par should have NPV < 0")
    elif fixed < par and payer.npv() <= 0:
        errors.append(f"Payer with fixed < par should have NPV > 0")
    else:
        print(f"  [OK] Payer sign: fixed={fixed*100:.2f}%, par={par*100:.2f}%, NPV=${payer.npv():,.0f}")

    return len(errors) == 0


def main():
    print("VERIFICATION COHERENCE: Vol Strategies, Barriers, Swap")
    ok1 = verify_volatility_strategies()
    ok2 = verify_barrier_options()
    ok3 = verify_interest_rate_swap()
    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("[OK] Tous les controles passent.")
    else:
        print("[X] Certains controles ont echoue.")
    print("=" * 60)


if __name__ == "__main__":
    main()
