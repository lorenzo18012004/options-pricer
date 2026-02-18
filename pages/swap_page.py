import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.curves import YieldCurve
from core.validation import validate_swap_params
from data.exceptions import ValidationError
from instruments import VanillaSwap, SwapCurveBuilder
from services import get_data_connector


def render_swap_pricer():
    """Interest rate swap pricer - desk-style analytics."""
    st.markdown("### Interest Rate Swap Pricer - Curve & Risk Analytics")

    c1, c2, c3 = st.columns(3)
    with c1:
        notional = st.number_input("Notional (millions)", value=10.0, step=1.0, key="sw_notional_m") * 1_000_000
        maturity = st.selectbox("Swap maturity (years)", [2, 3, 5, 7, 10, 20, 30], index=2, key="sw_mat")
        fixed_rate = st.number_input("Fixed Rate", value=0.0400, step=0.0005, format="%.4f", key="sw_fix")
    with c2:
        payment_freq = st.selectbox(
            "Payment Frequency",
            [("Annual", 1), ("Semi-Annual", 2), ("Quarterly", 4)],
            format_func=lambda x: x[0],
            key="sw_freq",
        )
        day_count = st.selectbox(
            "Day Count",
            ["30/360", "ACT/365", "ACT/360"],
            index=0,
            key="sw_day_count",
            help="30/360 (US), ACT/365, ACT/360",
        )
        position = st.radio("Position", ["payer", "receiver"], horizontal=True, key="sw_pos")
    with c3:
        run_swap = st.button("Run Swap Pricing", type="primary", use_container_width=True, key="sw_run")

    sw_key = (
        float(notional), float(maturity), float(fixed_rate), int(payment_freq[1]),
        day_count, position
    )
    if st.session_state.get("sw_key") != sw_key:
        st.session_state.pop("sw_state", None)
        st.session_state["sw_key"] = sw_key

    if not run_swap and "sw_state" not in st.session_state:
        return

    try:
        validate_swap_params(notional, fixed_rate, maturity, payment_freq[1])
    except ValidationError as e:
        st.error(str(e))
        return

    # Curve build (Yahoo or Synthetic)
    use_synthetic = st.session_state.get("data_source") == "Synthétique"
    connector = get_data_connector(use_synthetic)
    try:
        curve_nodes, curve_rates = connector.get_treasury_par_curve()
        curve = SwapCurveBuilder.build_from_market_data(
            {}, {}, {float(T): float(r) for T, r in zip(curve_nodes, curve_rates)}
        )
        curve_source_used = "Synthétique" if use_synthetic else "Yahoo Treasury (live)"
    except (ValueError, KeyError, TypeError, ImportError) as e_curve:
        st.error(f"Courbe de taux indisponible. Détails: {e_curve}")
        return

    swap = VanillaSwap(notional, fixed_rate, payment_freq[1], maturity, curve, position, day_count)
    npv = float(swap.npv())
    dv01 = float(swap.dv01())
    par_rate = float(swap.par_rate())
    mtm_bp = (fixed_rate - par_rate) * 10000.0
    pv_fixed = float(swap.swap.get_fixed_leg_pv())
    pv_float = float(swap.swap.get_floating_leg_pv())
    pvbp = dv01 / 100.0

    st.session_state["sw_state"] = {
        "curve": curve,
        "npv": npv,
        "dv01": dv01,
        "par_rate": par_rate,
        "mtm_bp": mtm_bp,
        "pv_fixed": pv_fixed,
        "pv_float": pv_float,
        "pvbp": pvbp,
        "curve_source": curve_source_used,
    }

    sw = st.session_state["sw_state"]
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("NPV", f"${sw['npv']:,.0f}")
    m2.metric("DV01", f"${sw['dv01']:,.0f}")
    m3.metric("PVBP", f"${sw['pvbp']:,.0f}")
    m4.metric("Par Rate", f"{sw['par_rate']*100:.3f}%")
    m5.metric("MTM (bp)", f"{sw['mtm_bp']:+.1f}")
    m6.metric("Fixed-Float PV", f"${(sw['pv_fixed']-sw['pv_float']):+,.0f}")

    # Quality consistency check: 1bp finite-diff vs DV01
    curve_up_1bp = YieldCurve(sw["curve"].maturities, sw["curve"].rates + 0.0001)
    swap_up = VanillaSwap(notional, fixed_rate, payment_freq[1], maturity, curve_up_1bp, position, day_count)
    pnl_1bp = float(swap_up.npv() - sw["npv"])
    dv01_gap = abs(abs(pnl_1bp) - sw["dv01"])
    quality = max(0.0, 100.0 - min(50.0, dv01_gap / max(sw["dv01"], 1e-8) * 500.0) - min(25.0, abs(sw["mtm_bp"]) * 0.1))
    q1, q2, q3 = st.columns(3)
    q1.metric("1bp Finite-Diff P&L", f"${pnl_1bp:+,.0f}")
    q2.metric("DV01 consistency gap", f"${dv01_gap:,.0f}")
    q3.metric("Swap Quality Score", f"{quality:.1f}/100")

    tabs = st.tabs(["Curve", "Scenarios", "Key-Rate Risk", "Hedging"])

    with tabs[0]:
        df_curve = pd.DataFrame({
            "Maturity_Y": sw["curve"].maturities,
            "ZeroRate_%": sw["curve"].rates * 100,
            "DF": [sw["curve"].get_discount_factor(float(t)) for t in sw["curve"].maturities],
        })
        st.dataframe(
            df_curve.style.format({"Maturity_Y": "{:.2f}", "ZeroRate_%": "{:.3f}%", "DF": "{:.6f}"}),
            use_container_width=True
        )

        t_grid = np.linspace(float(sw["curve"].maturities.min()), float(sw["curve"].maturities.max()), 120)
        z_grid = [sw["curve"].get_zero_rate(float(t)) * 100 for t in t_grid]
        f_grid = []
        for i in range(len(t_grid) - 1):
            f_grid.append(sw["curve"].get_forward_rate(float(t_grid[i]), float(t_grid[i + 1])) * 100)
        fig_curve = make_subplots(rows=1, cols=2, subplot_titles=("Zero Curve", "Forward Curve"))
        fig_curve.add_trace(go.Scatter(x=t_grid, y=z_grid, mode="lines", name="Zero"), row=1, col=1)
        fig_curve.add_trace(go.Scatter(x=t_grid[:-1], y=f_grid, mode="lines", name="Forward"), row=1, col=2)
        fig_curve.update_layout(template="plotly_white", height=360, showlegend=False)
        fig_curve.update_xaxes(title_text="Maturity (Y)", row=1, col=1)
        fig_curve.update_xaxes(title_text="Start maturity (Y)", row=1, col=2)
        fig_curve.update_yaxes(title_text="Rate (%)", row=1, col=1)
        fig_curve.update_yaxes(title_text="Rate (%)", row=1, col=2)
        st.plotly_chart(fig_curve, use_container_width=True)

    with tabs[1]:
        shifts = [-100, -50, -25, 0, 25, 50, 100]
        rows = []
        mats = sw["curve"].maturities
        for shock in shifts:
            r_par = sw["curve"].rates + shock / 10000.0
            c_par = YieldCurve(mats, r_par)
            s_par = VanillaSwap(notional, fixed_rate, payment_freq[1], maturity, c_par, position, day_count)
            npv_par = float(s_par.npv())
            rows.append({"Scenario": f"Parallel {shock:+}bp", "NPV": npv_par, "P&L": npv_par - sw["npv"]})

            prof = (mats - mats.min()) / max(mats.max() - mats.min(), 1e-8) - 0.5
            r_stp = sw["curve"].rates + (shock / 10000.0) * prof
            c_stp = YieldCurve(mats, r_stp)
            s_stp = VanillaSwap(notional, fixed_rate, payment_freq[1], maturity, c_stp, position, day_count)
            npv_stp = float(s_stp.npv())
            rows.append({"Scenario": f"Steepener {shock:+}bp", "NPV": npv_stp, "P&L": npv_stp - sw["npv"]})

        df_sc = pd.DataFrame(rows)
        st.dataframe(
            df_sc.style.format({"NPV": "${:,.0f}", "P&L": "${:+,.0f}"})
            .background_gradient(subset=["P&L"], cmap="RdYlGn"),
            use_container_width=True
        )
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(x=df_sc["Scenario"], y=df_sc["P&L"], marker_color="#3498db"))
        fig_sc.add_hline(y=0, line_color="gray")
        fig_sc.update_layout(
            title="Swap P&L by curve shock scenario",
            xaxis_title="Scenario", yaxis_title="P&L ($)",
            template="plotly_white", height=360
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[2]:
        kr_rows = []
        mats = sw["curve"].maturities
        for i, m_key in enumerate(mats):
            bumped = sw["curve"].rates.copy()
            bumped[i] += 0.0001
            c_b = YieldCurve(mats, bumped)
            s_b = VanillaSwap(notional, fixed_rate, payment_freq[1], maturity, c_b, position, day_count)
            kr = float(s_b.npv() - sw["npv"])
            kr_rows.append({"Maturity_Y": float(m_key), "KeyRateDV01": kr})
        df_kr = pd.DataFrame(kr_rows)
        st.dataframe(df_kr.style.format({"Maturity_Y": "{:.2f}", "KeyRateDV01": "${:+,.0f}"}), use_container_width=True)
        fig_kr = go.Figure()
        fig_kr.add_trace(go.Bar(x=df_kr["Maturity_Y"], y=df_kr["KeyRateDV01"], marker_color="#8e44ad"))
        fig_kr.add_hline(y=0, line_color="gray")
        fig_kr.update_layout(
            title="Key-rate DV01 profile",
            xaxis_title="Curve node maturity (Y)", yaxis_title="P&L for +1bp ($)",
            template="plotly_white", height=340
        )
        st.plotly_chart(fig_kr, use_container_width=True)

    with tabs[3]:
        hedge_tenor = st.selectbox("Hedge swap tenor (Y)", [2, 5, 10, 20, 30], index=2, key="sw_hedge_tenor")
        hedge_rate = st.number_input("Hedge fixed rate", value=sw["par_rate"], step=0.0005, format="%.4f", key="sw_hedge_rate")
        hedge_pos = st.radio("Hedge position", ["payer", "receiver"], horizontal=True, key="sw_hedge_pos")
        hedge_unit = VanillaSwap(1_000_000, hedge_rate, payment_freq[1], hedge_tenor, sw["curve"], hedge_pos, day_count)
        hedge_unit_dv01 = float(hedge_unit.dv01())
        hedge_notional = 0.0 if hedge_unit_dv01 == 0 else sw["dv01"] / hedge_unit_dv01 * 1_000_000
        h1, h2, h3 = st.columns(3)
        h1.metric("Current DV01", f"${sw['dv01']:,.0f}")
        h2.metric("Hedge DV01 / $1mm", f"${hedge_unit_dv01:,.0f}")
        h3.metric("Suggested Hedge Notional", f"${hedge_notional:,.0f}")

