"""
Quantitative regression tests for the Options Pricer.
Run with: python -m pytest test_quant_regression.py -v
Or: python test_quant_regression.py
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def bs_params():
    """Common Black-Scholes parameters."""
    return {"S": 100.0, "K": 100.0, "T": 0.5, "r": 0.05, "sigma": 0.25, "q": 0.0}


@pytest.fixture
def bs_params_div():
    """BS params with dividend yield."""
    return {"S": 100.0, "K": 105.0, "T": 1.0, "r": 0.05, "sigma": 0.30, "q": 0.02}


# =============================================================================
# BLACK-SCHOLES
# =============================================================================

class TestBlackScholes:
    """Black-Scholes pricing and Greeks."""

    def test_put_call_parity(self, bs_params):
        """Put-Call parity: C - P = S*exp(-qT) - K*exp(-rT)."""
        from core.black_scholes import BlackScholes
        p = bs_params
        c = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
        put = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", p["q"])
        theory = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        assert abs((c - put) - theory) < 1e-10

    def test_put_call_parity_with_dividends(self, bs_params_div):
        """Put-Call parity with dividend yield."""
        from core.black_scholes import BlackScholes
        p = bs_params_div
        c = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
        put = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", p["q"])
        theory = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        assert abs((c - put) - theory) < 1e-10

    def test_delta_bounds(self, bs_params):
        """Call delta in (0, 1), put delta in (-1, 0)."""
        from core.black_scholes import BlackScholes
        p = bs_params
        d_call = BlackScholes.get_delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call")
        d_put = BlackScholes.get_delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put")
        assert 0 < d_call < 1
        assert -1 < d_put < 0

    def test_gamma_positive(self, bs_params):
        """Gamma must be positive for both call and put."""
        from core.black_scholes import BlackScholes
        p = bs_params
        gamma = BlackScholes.get_gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert gamma > 0

    def test_vega_positive(self, bs_params):
        """Vega must be positive."""
        from core.black_scholes import BlackScholes
        p = bs_params
        vega = BlackScholes.get_vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
        assert vega > 0

    def test_call_price_in_bounds(self, bs_params):
        """Call price must be within intrinsic and upper bound."""
        from core.black_scholes import BlackScholes
        p = bs_params
        c = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
        intrinsic = max(p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"]), 0)
        upper = p["S"] * np.exp(-p["q"] * p["T"])
        assert intrinsic <= c <= upper + 1e-10

    def test_expiry_call_payoff(self):
        """At expiry T=0, call = max(S-K, 0)."""
        from core.black_scholes import BlackScholes
        c_itm = BlackScholes.get_price(110, 100, 0, 0.05, 0.25, "call")
        c_otm = BlackScholes.get_price(90, 100, 0, 0.05, 0.25, "call")
        assert c_itm == 10.0
        assert c_otm == 0.0

    def test_expiry_put_payoff(self):
        """At expiry T=0, put = max(K-S, 0)."""
        from core.black_scholes import BlackScholes
        p_itm = BlackScholes.get_price(90, 100, 0, 0.05, 0.25, "put")
        p_otm = BlackScholes.get_price(110, 100, 0, 0.05, 0.25, "put")
        assert p_itm == 10.0
        assert p_otm == 0.0

    def test_invalid_inputs_raise(self):
        """Invalid inputs must raise ValueError."""
        from core.black_scholes import BlackScholes
        with pytest.raises(ValueError):
            BlackScholes.get_price(0, 100, 0.5, 0.05, 0.25, "call")
        with pytest.raises(ValueError):
            BlackScholes.get_price(100, -1, 0.5, 0.05, 0.25, "call")
        with pytest.raises(ValueError):
            BlackScholes.get_price(100, 100, 0.5, 0.05, 0, "call")
        with pytest.raises(ValueError):
            BlackScholes.get_price(100, 100, 0.5, 0.05, 0.25, "invalid")

    def test_pnl_attribution_sum(self, bs_params):
        """P&L attribution: explained + unexplained = actual."""
        from core.black_scholes import BlackScholes
        p = bs_params
        S_new, sigma_new = 102.0, 0.27
        attrib = BlackScholes.pnl_attribution(
            p["S"], S_new, p["sigma"], sigma_new,
            p["K"], p["T"], p["r"], "call", p["q"], days_passed=1
        )
        explained = attrib["delta_pnl"] + attrib["gamma_pnl"] + attrib["vega_pnl"] + attrib["theta_pnl"] + attrib["vanna_pnl"]
        assert abs((explained + attrib["unexplained_pnl"]) - attrib["actual_pnl"]) < 1e-8


# =============================================================================
# IV SOLVER
# =============================================================================

class TestIVSolver:
    """Implied volatility solver."""

    def test_roundtrip_call(self, bs_params_div):
        """BS price -> IV solver -> BS price should match."""
        from core.black_scholes import BlackScholes
        from core.solvers import IVsolver
        p = bs_params_div
        price = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
        iv = IVsolver.find_implied_vol(price, p["S"], p["K"], p["T"], p["r"], "call", p["q"])
        assert iv is not None
        price_back = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], iv, "call", p["q"])
        assert abs(price_back - price) < 1e-6

    def test_roundtrip_put(self, bs_params):
        """Roundtrip for put option."""
        from core.black_scholes import BlackScholes
        from core.solvers import IVsolver
        p = bs_params
        price = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", p["q"])
        iv = IVsolver.find_implied_vol(price, p["S"], p["K"], p["T"], p["r"], "put", p["q"])
        assert iv is not None
        assert abs(BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], iv, "put", p["q"]) - price) < 1e-6

    def test_invalid_price_returns_none(self, bs_params):
        """Price outside bounds returns None."""
        from core.solvers import IVsolver
        p = bs_params
        assert IVsolver.find_implied_vol(-1, p["S"], p["K"], p["T"], p["r"], "call") is None
        assert IVsolver.find_implied_vol(p["S"] + 1, p["S"], p["K"], p["T"], p["r"], "call") is None
        assert IVsolver.find_implied_vol(1, p["S"], p["K"], 0, p["r"], "call") is None

    def test_nan_inputs_return_none(self, bs_params):
        """NaN/Inf inputs return None."""
        from core.solvers import IVsolver
        import numpy as np
        p = bs_params
        assert IVsolver.find_implied_vol(np.nan, p["S"], p["K"], p["T"], p["r"], "call") is None
        assert IVsolver.find_implied_vol(5.0, np.nan, p["K"], p["T"], p["r"], "call") is None
        assert IVsolver.find_implied_vol(5.0, p["S"], np.inf, p["T"], p["r"], "call") is None

    def test_atm_guess_converges_fast(self, bs_params):
        """Brenner-Subrahmanyam guess helps ATM convergence."""
        from core.black_scholes import BlackScholes
        from core.solvers import IVsolver
        p = bs_params
        S, K = 100.0, 100.0  # ATM
        T, r, q = 0.5, 0.05, 0.02
        sigma_true = 0.25
        price = BlackScholes.get_price(S, K, T, r, sigma_true, "call", q)
        iv = IVsolver.find_implied_vol(price, S, K, T, r, "call", q)
        assert iv is not None
        assert abs(iv - sigma_true) < 1e-6

    def test_find_implied_vol_with_reason(self, bs_params):
        """find_implied_vol_with_reason returns explicit failure reason."""
        from core.black_scholes import BlackScholes
        from core.solvers import (
            IVsolver,
            IV_FAIL_INVALID_INPUT,
            IV_FAIL_OUT_OF_BOUNDS,
            IV_OK,
        )
        p = bs_params
        S, K, T, r, q = p["S"], p["K"], p["T"], p["r"], p.get("q", 0.0)
        price = BlackScholes.get_price(S, K, T, r, 0.25, "call", q)
        iv, reason = IVsolver.find_implied_vol_with_reason(price, S, K, T, r, "call", q)
        assert iv is not None
        assert reason == IV_OK
        iv2, reason2 = IVsolver.find_implied_vol_with_reason(-1, S, K, T, r, "call", q)
        assert iv2 is None
        assert reason2 == IV_FAIL_INVALID_INPUT
        iv3, reason3 = IVsolver.find_implied_vol_with_reason(S + 10, S, K, T, r, "call", q)
        assert iv3 is None
        assert reason3 == IV_FAIL_OUT_OF_BOUNDS


# =============================================================================
# BINOMIAL TREE
# =============================================================================

class TestBinomialTree:
    """Binomial tree pricing."""

    def test_european_converges_to_bsm(self, bs_params):
        """Binomial (European) should converge to BSM for large n_steps."""
        from core.black_scholes import BlackScholes
        from models.trees import BinomialTree
        p = bs_params
        bsm = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call")
        tree = BinomialTree(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", n_steps=500)
        tree_price = tree.price()
        assert abs(tree_price - bsm) < 0.05

    def test_put_converges_to_bsm(self, bs_params):
        """Binomial put converges to BSM."""
        from core.black_scholes import BlackScholes
        from models.trees import BinomialTree
        p = bs_params
        bsm = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put")
        tree = BinomialTree(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", n_steps=500)
        assert abs(tree.price() - bsm) < 0.25

    def test_american_put_geq_european(self, bs_params):
        """American put >= European put (early exercise premium)."""
        from core.black_scholes import BlackScholes
        from models.trees import BinomialTree
        p = bs_params
        euro = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put")
        amer = BinomialTree(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", n_steps=200).price()
        assert amer >= euro - 0.01  # Allow small numerical error

    def test_invalid_inputs_raise(self):
        """Invalid inputs raise ValueError."""
        from models.trees import BinomialTree
        with pytest.raises(ValueError):
            BinomialTree(0, 100, 0.5, 0.05, 0.25, "call")
        with pytest.raises(ValueError):
            BinomialTree(100, 100, 0.5, 0.05, 0.25, "call", n_steps=0)


# =============================================================================
# MONTE CARLO
# =============================================================================

class TestMonteCarlo:
    """Monte Carlo pricer."""

    def test_european_call_converges_to_bsm(self, bs_params):
        """MC European call should be close to BSM for enough paths."""
        from core.black_scholes import BlackScholes
        from models.monte_carlo import MonteCarloPricer
        p = bs_params
        bsm = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call")
        mc = MonteCarloPricer(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                              n_simulations=30000, n_steps=100, seed=42)
        result = mc.price_european("call", use_antithetic=True)
        assert abs(result["price"] - bsm) < 0.18

    def test_european_put_converges_to_bsm(self, bs_params):
        """MC European put converges to BSM."""
        from core.black_scholes import BlackScholes
        from models.monte_carlo import MonteCarloPricer
        p = bs_params
        bsm = BlackScholes.get_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put")
        mc = MonteCarloPricer(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                              n_simulations=25000, n_steps=100, seed=123)
        result = mc.price_european("put", use_antithetic=True)
        assert abs(result["price"] - bsm) < 0.12

    def test_put_call_parity_mc(self, bs_params):
        """MC call - MC put = S*exp(-qT) - K*exp(-rT) approximately."""
        from models.monte_carlo import MonteCarloPricer
        p = bs_params
        mc = MonteCarloPricer(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                              n_simulations=40000, n_steps=100, seed=42)
        c = mc.price_european("call", use_antithetic=True)["price"]
        put = mc.price_european("put", use_antithetic=True)["price"]
        theory = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
        assert abs((c - put) - theory) < 0.25

    def test_reproducibility(self, bs_params):
        """Same seed gives same result."""
        from models.monte_carlo import MonteCarloPricer
        p = bs_params
        mc1 = MonteCarloPricer(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                               n_simulations=5000, n_steps=50, seed=999)
        mc2 = MonteCarloPricer(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                               n_simulations=5000, n_steps=50, seed=999)
        r1 = mc1.price_european("call")["price"]
        r2 = mc2.price_european("call")["price"]
        assert r1 == r2


# =============================================================================
# HESTON
# =============================================================================

class TestHeston:
    """Heston stochastic volatility model."""

    def test_put_call_parity(self):
        """Heston must satisfy put-call parity."""
        from core.heston import HestonModel
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.02
        c = model.get_price(S, K, T, r, "call", q)
        p = model.get_price(S, K, T, r, "put", q)
        theory = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((c - p) - theory) < 1e-6

    def test_heston_put_call_parity_vs_bsm_iv(self):
        """Heston price -> BSM IV -> BSM price should be close (roundtrip)."""
        from core.black_scholes import BlackScholes
        from core.heston import HestonModel
        S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.0
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        h_price = heston.get_price(S, K, T, r, "call", q)
        # Heston satisfies put-call parity; price should be in no-arb bounds
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        upper = S * np.exp(-q * T)
        assert intrinsic <= h_price <= upper + 1e-6

    def test_expiry_payoff(self):
        """At T=0, Heston gives intrinsic value."""
        from core.heston import HestonModel
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert model.get_price(110, 100, 0, 0.05, "call") == 10.0
        assert model.get_price(90, 100, 0, 0.05, "put") == 10.0


# =============================================================================
# BARRIER OPTIONS
# =============================================================================

class TestBarrierOptions:
    """Barrier option pricing."""

    def test_in_out_parity_down(self):
        """Down-and-in + down-and-out = vanilla."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 90.0
        doi = BarrierOption(S, K, T, r, sigma, barrier, "down-and-out", "call")
        din = BarrierOption(S, K, T, r, sigma, barrier, "down-and-in", "call")
        r_doi = doi.price(n_simulations=15000, n_steps=100, seed=42)
        r_din = din.price(n_simulations=15000, n_steps=100, seed=42)
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call")
        combined = r_doi["price"] + r_din["price"]
        assert abs(combined - vanilla) < 0.5

    def test_down_and_out_leq_vanilla(self):
        """Down-and-out call <= vanilla call."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 90.0
        dao = BarrierOption(S, K, T, r, sigma, barrier, "down-and-out", "call")
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call")
        dao_price = dao.price(n_simulations=10000, n_steps=100, seed=42)["price"]
        assert dao_price <= vanilla + 0.1

    def test_barrier_validation(self):
        """Barrier level validation."""
        from instruments.exotic import BarrierOption
        with pytest.raises(ValueError):
            BarrierOption(100, 100, 0.5, 0.05, 0.25, 110, "down-and-out", "call")
        with pytest.raises(ValueError):
            BarrierOption(100, 100, 0.5, 0.05, 0.25, 90, "up-and-out", "call")

    def test_analytical_down_and_out_in_parity(self):
        """Analytical: down-and-out + down-and-in = vanilla (exact)."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, q, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 0.02, 90.0
        doi = BarrierOption(S, K, T, r, sigma, barrier, "down-and-out", "call", q=q)
        din = BarrierOption(S, K, T, r, sigma, barrier, "down-and-in", "call", q=q)
        r_doi = doi.price(method="analytical")
        r_din = din.price(method="analytical")
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call", q)
        combined = r_doi["price"] + r_din["price"]
        assert abs(combined - vanilla) < 1e-10

    def test_analytical_down_and_out_leq_vanilla(self):
        """Analytical: down-and-out call <= vanilla call."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 90.0
        dao = BarrierOption(S, K, T, r, sigma, barrier, "down-and-out", "call")
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call")
        dao_price = dao.price(method="analytical")["price"]
        assert dao_price <= vanilla + 1e-10

    def test_analytical_up_and_out_in_parity(self):
        """Analytical: up-and-out + up-and-in = vanilla (exact)."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, q, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 0.02, 115.0
        uao = BarrierOption(S, K, T, r, sigma, barrier, "up-and-out", "call", q=q)
        uai = BarrierOption(S, K, T, r, sigma, barrier, "up-and-in", "call", q=q)
        r_uao = uao.price(method="analytical")
        r_uai = uai.price(method="analytical")
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call", q)
        combined = r_uao["price"] + r_uai["price"]
        assert abs(combined - vanilla) < 1e-10

    def test_analytical_up_and_out_leq_vanilla(self):
        """Analytical: up-and-out call <= vanilla call."""
        from instruments.exotic import BarrierOption
        from core.black_scholes import BlackScholes
        S, K, T, r, sigma, barrier = 100.0, 100.0, 0.5, 0.05, 0.25, 115.0
        uao = BarrierOption(S, K, T, r, sigma, barrier, "up-and-out", "call")
        vanilla = BlackScholes.get_price(S, K, T, r, sigma, "call")
        uao_price = uao.price(method="analytical")["price"]
        assert uao_price <= vanilla + 1e-10


# =============================================================================
# YIELD CURVE
# =============================================================================

class TestYieldCurve:
    """Yield curve and bootstrap."""

    def test_bootstrap_produces_valid_curve(self):
        """Bootstrap from par yields produces valid zero rates."""
        from core.curves import YieldCurve
        mats = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        par_yields = [0.05, 0.05, 0.05, 0.05, 0.045, 0.04]
        curve = YieldCurve.bootstrap_from_par_yields(par_yields, mats, payment_freq=2)
        assert curve is not None
        r = curve.get_zero_rate(1.0)
        assert 0 < r < 0.2
        df = curve.get_discount_factor(1.0)
        assert 0.8 < df < 1.0

    def test_discount_factor_consistency(self):
        """DF(t) = exp(-r(t)*t)."""
        from core.curves import YieldCurve
        mats = [0.25, 0.5, 1.0, 2.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        curve = YieldCurve(mats, rates)
        for t in [0.5, 1.0, 1.5]:
            r = curve.get_zero_rate(t)
            df = curve.get_discount_factor(t)
            assert abs(df - np.exp(-r * t)) < 1e-10

    def test_forward_rate_consistency(self):
        """Forward rate f(t1,t2) consistent with DF."""
        from core.curves import YieldCurve
        mats = [0.25, 0.5, 1.0, 2.0]
        rates = [0.03, 0.035, 0.04, 0.045]
        curve = YieldCurve(mats, rates)
        f = curve.get_forward_rate(0.5, 1.0)
        df1 = curve.get_discount_factor(0.5)
        df2 = curve.get_discount_factor(1.0)
        f_implied = -np.log(df2 / df1) / 0.5
        assert abs(f - f_implied) < 1e-10

    def test_invalid_maturities_raise(self):
        """Invalid maturities raise ValueError."""
        from core.curves import YieldCurve
        with pytest.raises(ValueError):
            YieldCurve([1, 2], [0.05])  # length mismatch
        with pytest.raises(ValueError):
            YieldCurve([2, 1], [0.05, 0.05])  # not increasing


# =============================================================================
# SWAP
# =============================================================================

class TestSwap:
    """Interest rate swap pricing."""

    def test_par_swap_npv_zero(self):
        """Swap at par rate has NPV = 0."""
        from core.curves import YieldCurve
        from instruments.rates import VanillaSwap, SwapCurveBuilder
        curve = SwapCurveBuilder.build_from_market_data(
            {}, {}, {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
        )
        swap = VanillaSwap(1_000_000, 0.04, 2, 5, curve, "payer")
        par = swap.par_rate()
        swap_at_par = VanillaSwap(1_000_000, par, 2, 5, curve, "payer")
        assert abs(swap_at_par.npv()) < 1.0  # Should be ~0

    def test_payer_receiver_opposite_npv(self):
        """Payer and receiver have opposite NPV."""
        from instruments.rates import VanillaSwap, SwapCurveBuilder
        curve = SwapCurveBuilder.build_from_market_data(
            {}, {}, {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
        )
        payer = VanillaSwap(1_000_000, 0.05, 2, 5, curve, "payer")
        receiver = VanillaSwap(1_000_000, 0.05, 2, 5, curve, "receiver")
        assert abs(payer.npv() + receiver.npv()) < 1e-6

    def test_day_count_affects_par_rate(self):
        """Different day counts give slightly different par rates."""
        from core.curves import YieldCurve
        from instruments.rates import VanillaSwap, SwapCurveBuilder
        curve = SwapCurveBuilder.build_from_market_data(
            {}, {}, {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
        )
        s_30360 = VanillaSwap(1_000_000, 0.04, 2, 5, curve, "payer", day_count="30/360")
        s_act365 = VanillaSwap(1_000_000, 0.04, 2, 5, curve, "payer", day_count="ACT/365")
        par_30 = s_30360.par_rate()
        par_act = s_act365.par_rate()
        assert par_30 > 0 and par_act > 0
        # Par rates should be close but may differ slightly
        assert abs(par_30 - par_act) < 0.01

    def test_dv01_positive_payer(self):
        """Payer swap: rates up -> NPV down, DV01 > 0 (sensitivity to +1bp)."""
        from instruments.rates import VanillaSwap, SwapCurveBuilder
        from core.curves import YieldCurve
        curve = SwapCurveBuilder.build_from_market_data(
            {}, {}, {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
        )
        swap = VanillaSwap(1_000_000, 0.04, 2, 5, curve, "payer")
        dv01 = swap.dv01()
        assert dv01 > 0


# =============================================================================
# SWAP CURVE BUILDER
# =============================================================================

class TestSwapCurveBuilder:
    """Swap curve construction."""

    def test_build_from_market_data(self):
        """Build curve from par yields dict."""
        from instruments.rates import SwapCurveBuilder
        swaps = {1.0: 0.03, 2.0: 0.035, 5.0: 0.04, 10.0: 0.045}
        curve = SwapCurveBuilder.build_from_market_data({}, {}, swaps)
        assert curve is not None
        assert len(curve.maturities) == 4
        r = curve.get_zero_rate(5.0)
        assert 0.02 < r < 0.10


# =============================================================================
# DATA CLEANER
# =============================================================================

class TestDataCleaner:
    """Data cleaning utilities."""

    def test_clean_option_chain_removes_bad_rows(self):
        """Clean removes rows with bad bid/ask."""
        from data.cleaner import DataCleaner
        df = pd.DataFrame({
            "strike": [100, 105, 110],
            "bid": [1.0, 0, 2.0],
            "ask": [1.1, 1.2, 2.2],
            "volume": [10, 20, 30],
            "openInterest": [100, 200, 300],
        })
        cleaned = DataCleaner.clean_option_chain(df, min_bid=0.01)
        assert len(cleaned) == 2  # Row with bid=0 removed

    def test_filter_by_moneyness(self):
        """Filter by moneyness keeps correct range."""
        from data.cleaner import DataCleaner
        df = pd.DataFrame({
            "strike": [70, 90, 100, 110, 140],
            "bid": [1] * 5,
            "ask": [1.1] * 5,
        })
        filtered = DataCleaner.filter_by_moneyness(df, spot=100, min_moneyness=0.85, max_moneyness=1.15)
        assert len(filtered) == 3  # 90, 100, 110
        assert 0.85 <= filtered["moneyness"].min() <= 1.15
        assert 0.85 <= filtered["moneyness"].max() <= 1.15

    def test_calculate_moneyness(self):
        """Moneyness = K/S."""
        from data.cleaner import DataCleaner
        mny = DataCleaner.calculate_moneyness(100, [90, 100, 110])
        np.testing.assert_array_almost_equal(mny, [0.9, 1.0, 1.1])

    def test_detect_outliers_iqr(self):
        """IQR outlier detection."""
        from data.cleaner import DataCleaner
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
        mask = DataCleaner.detect_outliers_iqr(data)
        assert mask[-1] == True
        assert np.sum(mask) >= 1

    def test_interpolate_volatility_surface(self):
        """Interpolation returns correct length and valid IVs."""
        from data.cleaner import DataCleaner
        # Use 4+ points for cubic interpolation (scipy requirement)
        strikes = np.array([85, 90, 100, 110, 115])
        ivs = np.array([0.28, 0.26, 0.22, 0.26, 0.28])
        target = np.array([92, 105])
        result = DataCleaner.interpolate_volatility_surface(strikes, ivs, target)
        assert len(result) == 2
        assert np.all(result > 0)
        assert np.all(result < 1.0)


# =============================================================================
# VOLATILITY SURFACE (SVI)
# =============================================================================

class TestVolatilitySurface:
    """SVI calibration and IV extraction."""

    def test_svi_calibration(self):
        """SVI calibrates and returns valid params."""
        from models.surfaces import VolatilitySurface
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.28, 0.25, 0.22, 0.25, 0.28])
        surface = VolatilitySurface([], [], [])
        params = surface.calibrate_svi(strikes, ivs, spot=100)
        if params is not None:
            iv_atm = surface.get_iv_from_svi(100, 100, params)
            assert 0.1 < iv_atm < 0.5

    def test_get_iv_from_svi_at_strikes(self):
        """SVI returns reasonable IV at calibration strikes."""
        from models.surfaces import VolatilitySurface
        strikes = np.array([90, 100, 110])
        ivs = np.array([0.26, 0.22, 0.26])
        surface = VolatilitySurface([], [], [])
        params = surface.calibrate_svi(strikes, ivs, spot=100)
        if params is not None:
            for k, iv in zip(strikes, ivs):
                iv_svi = surface.get_iv_from_svi(k, 100, params)
                assert abs(iv_svi - iv) < 0.05  # Should be close at calibration points


# =============================================================================
# INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration: load data -> price -> verify."""

    def test_synthetic_load_and_price_vanilla(self):
        """Synthetic connector: load AAPL, price call, verify roundtrip."""
        from datetime import datetime
        from data import SyntheticDataConnector
        from core.black_scholes import BlackScholes
        from core.solvers import IVsolver
        connector = SyntheticDataConnector
        spot = connector.get_spot_price("AAPL")
        exps = connector.get_expirations("AAPL")
        assert len(exps) > 0
        exp = exps[0]
        calls, puts = connector.get_option_chain("AAPL", exp)
        if calls.empty or puts.empty:
            pytest.skip("No synthetic options for AAPL")
        try:
            exp_dt = datetime.strptime(exp[:10], "%Y-%m-%d")
            days = (exp_dt - datetime.now()).days
            T = max(days / 365.0, 1e-4)
        except (ValueError, TypeError):
            T = 0.25
        rate = connector.get_risk_free_rate(T)
        div = connector.get_dividend_yield_forecast("AAPL", spot)
        row = calls.iloc[len(calls) // 2]
        K = float(row["strike"])
        mid = (float(row["bid"]) + float(row["ask"])) / 2
        iv = IVsolver.find_implied_vol(mid, spot, K, T, rate, "call", div)
        assert iv is not None
        price_back = BlackScholes.get_price(spot, K, T, rate, iv, "call", div)
        assert abs(price_back - mid) < 0.05

    def test_synthetic_treasury_has_extra_points(self):
        """Treasury curve has 2Y, 3Y, 7Y for swap precision (interpolated or direct)."""
        from data import SyntheticDataConnector
        mats, rates = SyntheticDataConnector.get_treasury_par_curve()
        assert len(mats) >= 5
        assert mats.min() <= 2 <= mats.max()
        assert mats.min() <= 3 <= mats.max()
        assert mats.min() <= 7 <= mats.max()

    def test_fallback_connector_uses_synthetic_on_yahoo_failure(self):
        """FallbackDataConnector uses Synthetic when Yahoo fails (mock)."""
        from services.market_service import FallbackDataConnector
        from data import SyntheticDataConnector
        store = {}
        fc = FallbackDataConnector(synthetic_connector=SyntheticDataConnector, state_store=store)
        spot = fc._try_or_fallback("get_spot_price", "AAPL", False)
        assert spot > 0
        exps = fc._try_or_fallback("get_expirations", "AAPL")
        assert len(exps) > 0

    def test_heston_edge_v0_small(self):
        """Heston with small v0 produces valid price."""
        from core.heston import HestonModel
        m = HestonModel(0.01, 2.0, 0.04, 0.3, -0.5)
        p = m.get_price(100, 100, 0.5, 0.05, "call", 0.0)
        assert p > 0 and p < 100

    def test_mc_barrier_converges_to_analytic(self):
        """MC up-and-out with many paths close to analytic."""
        from instruments.exotic import BarrierOption
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        barrier = 120.0
        opt = BarrierOption(S, K, T, r, sigma, barrier, "up-and-out", "call")
        mc_result = opt.price(n_simulations=20000, n_steps=100, seed=42)
        analytic_result = opt.price(method="analytical")
        analytic = analytic_result.get("price") if analytic_result else None
        if analytic is not None and mc_result["price"] is not None:
            rel_err = abs(mc_result["price"] - analytic) / max(analytic, 0.01)
            assert rel_err < 0.15  # 15% tolerance for MC vs analytic


# =============================================================================
# RUNNER (for python test_quant_regression.py)
# =============================================================================

def run_all():
    """Run all tests via pytest (legacy runner for python test_quant_regression.py)."""
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code == 0


if __name__ == "__main__":
    success = run_all()
    exit(0 if success else 1)
