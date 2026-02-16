"""
Helper functions for vanilla option pricer UI.
Extracts tab rendering logic to keep vanilla_page maintainable.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

from core import BlackScholes
from core.heston import HestonModel
from data import DataCleaner, DataConnector
from models import BinomialTree, MonteCarloPricer
from config.options import (
    MAX_SPREAD_PCT,
    IV_MIN_PCT,
    IV_MAX_PCT,
    IV_CAP_DISPLAY_PCT,
    MIN_VOLUME,
    MIN_MID_PRICE,
    BREACH_TOLERANCE_FLOOR,
    BREACH_TOLERANCE_CAP,
    BREACH_RATE_GOOD,
    BREACH_RATE_WARNING,
    MONEYNESS_SURFACE_MIN,
    MONEYNESS_SURFACE_MAX,
    SURFACE_SMOOTH_RBF,
    SURFACE_GAUSSIAN_SIGMA,
)


def render_market_summary(summary_box, df, spot, atm_idx, days_to_exp, maturity_mode,
                          rate, atm_iv, atm_strike, hv_window, hist_vol, biz_days_to_exp,
                          total_volume_exp=None, total_oi_exp=None):
    """Render Market Summary expander."""
    with summary_box:
        c1, c2, c3, c4, c5, c6 = summary_box.columns(6)
        c1.metric("Spot", f"${spot:.2f}")
        c2.metric("ATM Strike", f"${atm_strike:.0f}")
        c3.metric("Days", f"{days_to_exp}", help=f"{maturity_mode} to expiry")
        c4.metric("Rate", f"{rate*100:.2f}%",
                  help="Taux sans risque interpolé depuis la courbe Treasury US (Yahoo: ^IRX, ^FVX, ^TNX, ^TYX).")
        c5.metric("ATM IV", f"{atm_iv:.1f}%")
        c6.metric(f"Hist Vol ({hv_window}d)", f"{hist_vol*100:.1f}%",
                  help=f"{hv_window}d window matched to {biz_days_to_exp} business days to expiry")
        iv_hv_spread = atm_iv - hist_vol * 100
        c7, c8, c9, c10 = summary_box.columns(4)
        c7.metric("IV - HV Spread", f"{iv_hv_spread:+.1f}%",
                  help="Positive = options are expensive vs realized vol")
        # ATM spread (95-105% moneyness) plus représentatif que la moyenne sur tout le range
        atm_mask = (df["Moneyness"] >= 0.95) & (df["Moneyness"] <= 1.05)
        atm_spread_pct = df.loc[atm_mask, "Spread_%"].median() if atm_mask.any() else df["Spread_%"].median()
        atm_spread_dollar = df.loc[atm_mask, "Spread_$"].median() if atm_mask.any() and "Spread_$" in df.columns else (df["Ask"] - df["Bid"]).median() if "Ask" in df.columns and "Bid" in df.columns else 0
        c8.metric("ATM Spread", f"{atm_spread_pct:.1f}% (${atm_spread_dollar:.2f})",
                  help="Médiane (Ask-Bid)/Mid et Ask-Bid en $ sur 95-105% moneyness. Plus robuste aux outliers.")
        vol_val = total_volume_exp if total_volume_exp is not None else df["Volume"].sum()
        oi_val = total_oi_exp if total_oi_exp is not None else df["OpenInt"].sum()
        c9.metric("Total Volume", f"{vol_val:,.0f}",
                  help="Volume du jour sur toute l'échéance (calls+puts). Yahoo intraday.")
        c10.metric("Total OI", f"{oi_val:,.0f}",
                   help="Open Interest sur toute l'échéance (calls+puts), clôture veille.")


def render_data_quality(dq_box, df, raw_count, filtered_count, quote_ts, opt_type,
                        svi_fit_rmse_pct, day_count_basis, count_after_clean=None):
    """Render Data Quality expander."""
    with dq_box:
        retention_pct = (filtered_count / raw_count * 100) if raw_count > 0 else 0
        dq1, dq2, dq3, dq4 = dq_box.columns(4)
        dq1.metric("Raw strikes", f"{raw_count}",
                   help="Tous les strikes retournés par Yahoo pour l'échéance.")
        dq2.metric("Filtered strikes", f"{filtered_count}",
                   help="Après clean + moneyness 80-120%.")
        dq3.metric("Retention", f"{retention_pct:.1f}%",
                   help="filtered / raw. Voir le funnel ci-dessous.")
        dq4.metric("Quote timestamp", quote_ts if quote_ts else "N/A")
        chk = df[["Strike", "Mid"]].dropna().sort_values("Strike").reset_index(drop=True)
        mono_breaches = 0
        conv_breaches = 0
        if len(chk) >= 3:
            mids = chk["Mid"].values
            mono_breaches = int(np.sum(np.diff(mids) > 1e-6)) if opt_type == "call" else int(np.sum(np.diff(mids) < -1e-6))
            conv = np.diff(mids, 2)
            conv_breaches = int(np.sum(conv < -1e-4))
        dq5, dq6, dq7 = dq_box.columns(3)
        dq5.metric("Monotonicity breaches", f"{mono_breaches}")
        dq6.metric("Convexity breaches", f"{conv_breaches}")
        med_spread_dollar = df["Spread_$"].median() if "Spread_$" in df.columns else (df["Ask"] - df["Bid"]).median() if "Ask" in df.columns and "Bid" in df.columns else 0
        dq7.metric("Med spread", f"{df['Spread_%'].median():.2f}% (${med_spread_dollar:.2f})",
                   help="Médiane sur les strikes (80-120% moneyness). Plus robuste aux outliers que la moyenne.")
        dq8, dq9, dq10 = dq_box.columns(3)
        dq8.metric("SVI fit RMSE", f"{svi_fit_rmse_pct:.3f}%" if svi_fit_rmse_pct is not None else "N/A",
                   help="Si > 5%, fallback automatique sur IV marché (SVI trop bruité).")
        dq9.metric("Day count", day_count_basis)
        dq10.metric("Maturity clock", "Calendar")
        penalty = 0.0
        penalty += min(30.0, mono_breaches * 3.0)
        penalty += min(30.0, conv_breaches * 3.0)
        penalty += min(20.0, max(0.0, df["Spread_%"].median() - 8.0) * 0.8)
        if svi_fit_rmse_pct is not None:
            penalty += min(20.0, max(0.0, svi_fit_rmse_pct - 3.0) * 2.0)
        quality_score = max(0.0, 100.0 - penalty)
        qs_col1, qs_col2 = dq_box.columns(2)
        qs_col1.metric("Pricing quality score", f"{quality_score:.1f}/100")
        if quality_score >= 80:
            qs_col2.success("Quality regime: GOOD")
        elif quality_score >= 60:
            qs_col2.warning("Quality regime: ACCEPTABLE")
        else:
            qs_col2.error("Quality regime: NOISY")
        if mono_breaches == 0 and conv_breaches == 0:
            dq_box.success("No-arbitrage quick checks: clean on filtered universe.")
        else:
            dq_box.warning("Some no-arbitrage checks fail.")
        if count_after_clean is not None and raw_count > 0:
            pct_clean = count_after_clean / raw_count * 100
            pct_mny = filtered_count / count_after_clean * 100 if count_after_clean > 0 else 0
            dq_box.markdown("**Funnel de rétention :**")
            dq_box.markdown(
                f"- **1. clean_option_chain** : {raw_count} → {count_after_clean} ({pct_clean:.1f}%)  "
                f"*Exclut : bid/ask ≤ 1¢, spread > 50%, volume=0 ET OI=0*"
            )
            dq_box.markdown(
                f"- **2. filter_by_moneyness(80-120%)** : {count_after_clean} → {filtered_count} ({pct_mny:.1f}%)  "
                f"*Zone liquide ATM. Les strikes OTM lointains (ex: K=0.5×S ou 1.5×S) sont illiquides.*"
            )
            dq_box.caption(
                "Justification : Les options 80-120% moneyness concentrent la liquidité (volume, OI, spread serré). "
                "Les deep OTM ont des IV bruitées et des mid peu fiables. Pour élargir : modifier filter_by_moneyness."
            )
        dq_box.caption(
            "Mid = (Bid+Ask)/2. Yahoo Finance: bid/ask ~15min delayed → spreads plus larges qu'en temps réel."
        )


def render_chain_display(chain_box, df, option_type, ticker, exp_date):
    """Render option chain dataframe in expander."""
    with chain_box:
        fmt = {
            "Strike": "${:.2f}", "Moneyness": "{:.3f}",
            "Bid": "${:.2f}", "Ask": "${:.2f}", "Mid": "${:.2f}",
            "Spread_%": "{:.1f}%", "Spread_$": "${:.2f}",
            "IV_%": "{:.1f}%", "SVI_IV_%": "{:.1f}%", "IV_Used_%": "{:.1f}%", "HV_%": "{:.1f}%",
            "BS_Price": "${:.2f}", "Model_HV": "${:.2f}", "Mispricing": "${:+.2f}",
            "Delta": "{:+.3f}", "Gamma": "{:.5f}", "Vega": "{:.3f}",
            "Theta": "{:+.3f}", "Rho": "{:+.4f}",
            "Volume": "{:.0f}", "OpenInt": "{:.0f}"
        }
        fmt = {k: v for k, v in fmt.items() if k in df.columns}
        max_mp = max(abs(df["Mispricing"].max()), abs(df["Mispricing"].min()), 0.5)
        st.dataframe(
            df.style.format(fmt)
            .background_gradient(subset=["Mispricing"], cmap="RdYlGn", vmin=-max_mp, vmax=max_mp)
            .background_gradient(subset=["IV_%"], cmap="YlOrRd"),
            use_container_width=True, height=350
        )


def render_volatility_tab(tab, df, spot, hist_vol):
    """Render volatility smile and mispricing charts."""
    with tab:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(
                x=df["Strike"], y=df["IV_%"], mode="markers+lines",
                name="Implied Vol (Market)",
                marker=dict(size=7, color="#636EFA"), line=dict(width=2)
            ))
            fig_smile.add_hline(y=hist_vol * 100, line_dash="dash", line_color="orange",
                               annotation_text=f"Hist Vol: {hist_vol*100:.1f}%")
            fig_smile.add_vline(x=spot, line_dash="dot", line_color="red",
                               annotation_text=f"Spot ${spot:.0f}")
            fig_smile.update_layout(
                title="Volatility Smile (IV vs Strike)",
                xaxis_title="Strike", yaxis_title="Implied Volatility (%)",
                template="plotly_white", height=380
            )
            st.plotly_chart(fig_smile, use_container_width=True)

        with col_v2:
            fig_misprice = go.Figure()
            colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in df["Mispricing"]]
            fig_misprice.add_trace(go.Bar(
                x=df["Strike"], y=df["Mispricing"], name="Mispricing",
                marker_color=colors
            ))
            fig_misprice.add_hline(y=0, line_color="gray")
            fig_misprice.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_misprice.update_layout(
                title="Mispricing: Model (HV) vs Market",
                xaxis_title="Strike", yaxis_title="Mispricing ($)",
                template="plotly_white", height=380
            )
            st.plotly_chart(fig_misprice, use_container_width=True)


def render_greeks_tabs(tab, df, spot, opt_type):
    """Render Delta, Gamma, Vega, Theta, 2nd-order Greeks tabs."""
    with tab:
        tab_d, tab_g, tab_v, tab_t, tab_2nd = st.tabs(
            ["Delta", "Gamma", "Vega", "Theta", "2nd Order"]
        )

        with tab_d:
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=df["Strike"], y=df["Delta"], mode="lines+markers",
                name="Delta", line=dict(color="#3498db", width=2.5), marker=dict(size=5)
            ))
            fig_delta.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_delta.add_hline(y=0.5 if opt_type == "call" else -0.5,
                               line_dash="dash", line_color="gray",
                               annotation_text="ATM Delta")
            fig_delta.update_layout(title="Delta vs Strike",
                xaxis_title="Strike", yaxis_title="Delta",
                template="plotly_white", height=400)
            st.plotly_chart(fig_delta, use_container_width=True)

        with tab_g:
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=df["Strike"], y=df["Gamma"], mode="lines+markers",
                name="Gamma", line=dict(color="#2ecc71", width=2.5), marker=dict(size=5)
            ))
            fig_gamma.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_gamma.update_layout(title="Gamma vs Strike",
                xaxis_title="Strike", yaxis_title="Gamma",
                template="plotly_white", height=400)
            st.plotly_chart(fig_gamma, use_container_width=True)

        with tab_v:
            fig_vega = go.Figure()
            fig_vega.add_trace(go.Scatter(
                x=df["Strike"], y=df["Vega"], mode="lines+markers",
                name="Vega", line=dict(color="#e67e22", width=2.5), marker=dict(size=5)
            ))
            fig_vega.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_vega.update_layout(title="Vega vs Strike (per 1% vol)",
                xaxis_title="Strike", yaxis_title="Vega",
                template="plotly_white", height=400)
            st.plotly_chart(fig_vega, use_container_width=True)

        with tab_t:
            fig_theta = go.Figure()
            fig_theta.add_trace(go.Scatter(
                x=df["Strike"], y=df["Theta"], mode="lines+markers",
                name="Theta", line=dict(color="#e74c3c", width=2.5), marker=dict(size=5)
            ))
            fig_theta.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_theta.add_hline(y=0, line_color="gray", line_dash="dash")
            fig_theta.update_layout(title="Theta vs Strike (per day)",
                xaxis_title="Strike", yaxis_title="Theta",
                template="plotly_white", height=400)
            st.plotly_chart(fig_theta, use_container_width=True)

        with tab_2nd:
            col_2a, col_2b = st.columns(2)
            with col_2a:
                fig_vanna = go.Figure()
                fig_vanna.add_trace(go.Scatter(
                    x=df["Strike"], y=df["Vanna"], mode="lines+markers",
                    name="Vanna", line=dict(color="#9b59b6", width=2.5), marker=dict(size=5)
                ))
                fig_vanna.add_vline(x=spot, line_dash="dot", line_color="red")
                fig_vanna.add_hline(y=0, line_color="gray", line_dash="dash")
                fig_vanna.update_layout(
                    title="Vanna vs Strike (dDelta/dVol)",
                    xaxis_title="Strike", yaxis_title="Vanna",
                    template="plotly_white", height=350)
                st.plotly_chart(fig_vanna, use_container_width=True)

            with col_2b:
                fig_volga = go.Figure()
                fig_volga.add_trace(go.Scatter(
                    x=df["Strike"], y=df["Volga"], mode="lines+markers",
                    name="Volga", line=dict(color="#1abc9c", width=2.5), marker=dict(size=5)
                ))
                fig_volga.add_vline(x=spot, line_dash="dot", line_color="red")
                fig_volga.update_layout(
                    title="Volga vs Strike (dVega/dVol)",
                    xaxis_title="Strike", yaxis_title="Volga",
                    template="plotly_white", height=350)
                st.plotly_chart(fig_volga, use_container_width=True)

            fig_charm = go.Figure()
            fig_charm.add_trace(go.Scatter(
                x=df["Strike"], y=df["Charm"], mode="lines+markers",
                name="Charm", line=dict(color="#e67e22", width=2.5), marker=dict(size=5)
            ))
            fig_charm.add_vline(x=spot, line_dash="dot", line_color="red")
            fig_charm.add_hline(y=0, line_color="gray", line_dash="dash")
            fig_charm.update_layout(
                title="Charm vs Strike (dDelta/dTime per day) - Delta Decay",
                xaxis_title="Strike", yaxis_title="Charm (per day)",
                template="plotly_white", height=350)
            st.plotly_chart(fig_charm, use_container_width=True)


def render_pnl_tab(tab, df, spot, opt_type, atm_idx):
    """Render P&L at expiration chart (Long et Short)."""
    MULTIPLIER = 100
    with tab:
        col_pnl1, col_pnl2 = st.columns([1, 3])
        with col_pnl1:
            position = st.radio("Position", ["Long", "Short"], horizontal=True, key="pnl_position")
            is_long = position == "Long"
            pnl_strike = st.selectbox("Strike for P&L",
                                      df["Strike"].tolist(), index=int(atm_idx))
            pnl_row = df[df["Strike"] == pnl_strike].iloc[0]
            n_contracts = st.number_input("Contracts", value=1, min_value=1, max_value=100)

            if is_long:
                entry_price = pnl_row["Ask"]
                entry_label = "Entry (Ask)"
            else:
                entry_price = pnl_row["Bid"]
                entry_label = "Entry (Bid)"
            premium_mid = pnl_row["Mid"]
            spread_cost = (pnl_row["Ask"] - pnl_row["Bid"]) / 2

            st.metric(entry_label, f"${entry_price:.2f}")
            st.metric("Mid Price", f"${premium_mid:.2f}")
            st.metric("Spread Cost", f"${spread_cost:.2f}/share")
            if is_long:
                st.metric("Total Cost", f"${entry_price * MULTIPLIER * n_contracts:,.0f}",
                          help=f"{n_contracts} x {MULTIPLIER} x ${entry_price:.2f}")
            else:
                st.metric("Total Credit", f"${entry_price * MULTIPLIER * n_contracts:,.0f}",
                          help=f"{n_contracts} x {MULTIPLIER} x ${entry_price:.2f} (reçu)")

            if opt_type == "call":
                be = pnl_strike + entry_price
            else:
                be = pnl_strike - entry_price
            st.metric("Breakeven", f"${be:.2f}")
            if is_long:
                st.metric("Max Loss", f"${entry_price * MULTIPLIER * n_contracts:,.0f}")
            else:
                if opt_type == "call":
                    st.metric("Max Loss", "Illimité", help="Short call: perte illimitée si spot monte")
                else:
                    max_loss_put = (pnl_strike - entry_price) * MULTIPLIER * n_contracts
                    st.metric("Max Loss", f"${max_loss_put:,.0f}",
                              help="Short put: perte max si spot → 0")

        with col_pnl2:
            spot_range = np.linspace(spot * 0.7, spot * 1.3, 200)
            if opt_type == "call":
                payoff = np.maximum(spot_range - pnl_strike, 0)
            else:
                payoff = np.maximum(pnl_strike - spot_range, 0)
            if is_long:
                pnl_vals = (payoff - entry_price) * MULTIPLIER * n_contracts
            else:
                pnl_vals = (entry_price - payoff) * MULTIPLIER * n_contracts

            fig_pnl = go.Figure()
            line_color = "#3498db" if is_long else "#e74c3c"
            fill_color = "rgba(46,204,113,0.15)" if is_long else "rgba(231,76,60,0.15)"
            fig_pnl.add_trace(go.Scatter(
                x=spot_range, y=pnl_vals, mode="lines",
                name=f"P&L {position}", line=dict(color=line_color, width=3),
                fill="tozeroy", fillcolor=fill_color
            ))
            fig_pnl.add_hline(y=0, line_color="gray", line_width=1)
            fig_pnl.add_vline(x=spot, line_dash="dot", line_color="red",
                             annotation_text=f"Spot ${spot:.0f}")
            fig_pnl.add_vline(x=pnl_strike, line_dash="dash", line_color="blue",
                             annotation_text=f"K={pnl_strike:.0f}")
            fig_pnl.add_vline(x=be, line_dash="dash", line_color="green",
                             annotation_text=f"BE={be:.0f}")
            option_type = "Call" if opt_type == "call" else "Put"
            fig_pnl.update_layout(
                title=f"P&L at Expiration - {position} {n_contracts} {option_type}(s) K={pnl_strike:.0f} (x{MULTIPLIER})",
                xaxis_title="Spot at Expiration",
                yaxis_title=f"P&L ($) - {n_contracts} contract(s)",
                template="plotly_white", height=450
            )
            st.plotly_chart(fig_pnl, use_container_width=True)


def render_attribution_tab(tab, df, spot, opt_type, atm_idx, rate, T, div_yield, days_to_exp):
    """Render P&L Attribution tab."""
    with tab:
        col_at1, col_at2 = st.columns([1, 3])
        with col_at1:
            at_strike = st.selectbox("Strike", df["Strike"].tolist(),
                                     index=int(atm_idx), key="at_strike")
            at_row = df[df["Strike"] == at_strike].iloc[0]
            at_iv = at_row["IV_%"] / 100
            spot_move = st.slider("Spot move (%)", -10.0, 10.0, 2.0, 0.5)
            vol_move = st.slider("Vol move (points)", -5.0, 5.0, -1.0, 0.5)
            max_days_passed = max(1, days_to_exp - 1)
            days_move = 1 if max_days_passed <= 1 else st.slider("Days passed", 1, max_days_passed, 1)
        with col_at2:
            new_spot = spot * (1 + spot_move / 100)
            new_vol = max(0.01, at_iv + vol_move / 100)
            attrib = BlackScholes.pnl_attribution(
                S_old=spot, S_new=new_spot,
                sigma_old=at_iv, sigma_new=new_vol,
                K=at_strike, T=T, r=rate,
                option_type=opt_type, q=div_yield,
                days_passed=days_move
            )
            components = ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Unexplained"]
            values = [
                attrib["delta_pnl"], attrib["gamma_pnl"],
                attrib["vega_pnl"], attrib["theta_pnl"],
                attrib["vanna_pnl"], attrib["unexplained_pnl"]
            ]
            colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#95a5a6"]
            fig_attrib = go.Figure()
            fig_attrib.add_trace(go.Bar(
                x=components, y=values, marker_color=colors,
                text=[f"${v:+.3f}" for v in values], textposition="outside"
            ))
            fig_attrib.add_hline(y=0, line_color="gray")
            fig_attrib.add_hline(
                y=attrib["actual_pnl"], line_dash="dash", line_color="black",
                annotation_text=f"Actual P&L: ${attrib['actual_pnl']:+.3f}"
            )
            fig_attrib.update_layout(
                title=f"P&L Attribution (K={at_strike:.0f}, S: {spot:.0f}->{new_spot:.0f}, "
                      f"Vol: {at_iv*100:.1f}%->{new_vol*100:.1f}%, {days_move}d)",
                yaxis_title="P&L ($)", template="plotly_white", height=420
            )
            st.plotly_chart(fig_attrib, use_container_width=True)
            c_a1, c_a2, c_a3 = st.columns(3)
            c_a1.metric("Actual P&L", f"${attrib['actual_pnl']:+.4f}")
            c_a2.metric("Explained", f"${attrib['explained_pnl']:+.4f}")
            ratio = attrib["explanation_ratio"]
            c_a3.metric("Explanation Ratio", f"{ratio:.1f}%" if ratio is not None else "N/A")


def render_heston_tab(tab, df, spot, exp_date, opt_type, T, rate, div_yield, ticker, atm_iv):
    """Render BSM vs Heston tab."""
    with tab:
        cal_strikes = df["Strike"].values
        cal_ivs = df["IV_%"].values / 100
        cal_spreads = (df["Spread_%"].values / 100) if "Spread_%" in df.columns else None
        cal_oi = df["OpenInt"].values if "OpenInt" in df.columns else None
        heston_key = (ticker, exp_date, opt_type, len(cal_strikes), round(float(spot), 4), round(float(T), 6))
        if st.session_state.get("heston_key") != heston_key:
            st.session_state.pop("heston_result", None)
            st.session_state["heston_key"] = heston_key
        run_heston = st.button("Run Heston calibration", key="run_heston_btn")
        if run_heston:
            with st.spinner("Calibrating Heston model to market data..."):
                try:
                    heston_model, cal_info = HestonModel.calibrate(
                        cal_strikes, cal_ivs, spot, T, rate, div_yield, opt_type,
                        spreads=cal_spreads, open_interests=cal_oi,
                        moneyness_range=(0.85, 1.15), max_points=25,
                    )
                    st.session_state["heston_result"] = {"success": True, "model": heston_model, "info": cal_info}
                except (ValueError, RuntimeError, TypeError) as e_cal:
                    import logging
                    logging.getLogger(__name__).warning("Heston calibration failed: %s", e_cal)
                    st.session_state["heston_result"] = {"success": False, "error": str(e_cal)}
        heston_state = st.session_state.get("heston_result")
        if not heston_state:
            pass
        elif not heston_state.get("success", False):
            st.error(f"Calibration failed: {heston_state.get('error', 'unknown error')}")
        else:
            heston_model = heston_state["model"]
            cal_info = heston_state["info"]
            st.markdown("**Calibrated parameters:**")
            cp1, cp2, cp3, cp4, cp5 = st.columns(5)
            cp1.metric("v0 (var inst.)", f"{cal_info['v0']:.4f}",
                       help=f"Instant vol = {cal_info['current_vol']:.1f}%")
            cp2.metric("kappa", f"{cal_info['kappa']:.2f}", help="Mean-reversion speed")
            cp3.metric("theta (var LT)", f"{cal_info['theta']:.4f}",
                       help=f"Long-term vol = {cal_info['long_term_vol']:.1f}%")
            cp4.metric("xi (vol-of-vol)", f"{cal_info['xi']:.3f}")
            cp5.metric("rho (corr S-vol)", f"{cal_info['rho']:.3f}", help="Negative = leverage effect")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("RMSE IV (all)", f"{cal_info['rmse_iv_pct']:.3f}%")
            cm2.metric("Feller Condition", "OK" if cal_info["feller_satisfied"] else "Violated",
                       help="2*kappa*theta > xi^2 ensures variance stays positive")
            cm3.metric("Calibration", "Converged" if cal_info["success"] else "Warning")
            qm1, qm2, qm3 = st.columns(3)
            qm1.metric("RMSE IV (filtered)", f"{cal_info.get('rmse_iv_filtered_pct', cal_info['rmse_iv_pct']):.3f}%")
            qm2.metric("Points used", f"{cal_info.get('n_points_used', 0)}/{cal_info.get('n_points_total', len(cal_strikes))}")
            qm3.metric("Boundary hits", f"{cal_info.get('boundary_hits', 0)}")
            rmse_filtered = cal_info.get("rmse_iv_filtered_pct", cal_info["rmse_iv_pct"])
            heston_usable = (
                cal_info["success"] and rmse_filtered <= 15.0
                and cal_info.get("boundary_hits", 0) <= 1
                and cal_info.get("n_points_used", 0) >= 6
            )
            if heston_usable:
                st.success("Heston quality check: usable for analysis.")
            else:
                st.warning("Heston calibration not robust.")
            st.markdown("---")
            st.markdown("**Smile comparison: Market vs Heston**")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=cal_strikes, y=cal_ivs * 100,
                mode="markers", name="Market IV",
                marker=dict(size=9, color="#636EFA", symbol="circle")
            ))
            fig_comp.add_hline(y=atm_iv, line_dash="dash", line_color="orange",
                               annotation_text=f"BSM (flat IV = {atm_iv:.1f}%)")
            heston_ivs = cal_info["model_ivs"] if heston_usable else cal_ivs
            fig_comp.add_trace(go.Scatter(
                x=cal_strikes, y=[iv * 100 for iv in heston_ivs],
                mode="lines+markers", name="Heston IV" if heston_usable else "Fallback IV (Market)",
                line=dict(color="#EF553B", width=2.5), marker=dict(size=6)
            ))
            fig_comp.add_vline(x=spot, line_dash="dot", line_color="gray",
                               annotation_text=f"Spot ${spot:.0f}")
            fig_comp.update_layout(
                title="Implied Volatility: Market vs BSM vs Heston",
                xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
                template="plotly_white", height=420, showlegend=True
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown("**Price comparison by strike:**")
            comp_rows = []
            for i, K in enumerate(cal_strikes):
                mkt_mid = df[df["Strike"] == K]["Mid"].values
                if len(mkt_mid) == 0:
                    continue
                mkt_mid = mkt_mid[0]
                iv_mkt = cal_ivs[i]
                bsm_price = BlackScholes.get_price(spot, K, T, rate, iv_mkt, opt_type, div_yield)
                try:
                    heston_price = heston_model.get_price(spot, K, T, rate, opt_type, div_yield) if heston_usable else bsm_price
                except (ValueError, RuntimeError):
                    heston_price = bsm_price
                comp_rows.append({
                    "Strike": K, "Market Mid": mkt_mid, "BSM Price": bsm_price, "Heston Price": heston_price,
                    "BSM - Mkt": bsm_price - mkt_mid, "Heston - Mkt": heston_price - mkt_mid,
                })
            if comp_rows:
                df_comp = pd.DataFrame(comp_rows)
                st.dataframe(
                    df_comp.style.format({
                        "Strike": "${:.2f}", "Market Mid": "${:.2f}",
                        "BSM Price": "${:.4f}", "Heston Price": "${:.4f}",
                        "BSM - Mkt": "${:+.4f}", "Heston - Mkt": "${:+.4f}",
                    }).background_gradient(subset=["Heston - Mkt"], cmap="RdYlGn", vmin=-0.5, vmax=0.5),
                    use_container_width=True
                )
                if not heston_usable:
                    st.warning("Heston calibration not robust.")


def render_mc_tab(tab, df, spot, opt_type, atm_idx, rate, T, div_yield, hist_vol, ticker, exp_date):
    """Render MC vs BSM tab."""
    with tab:
        col_mc1, col_mc2 = st.columns([1, 3])
        with col_mc1:
            mc_strike = st.selectbox("Strike MC", df["Strike"].tolist(),
                                     index=int(atm_idx), key="mc_strike")
            mc_row = df[df["Strike"] == mc_strike].iloc[0]
            vol_source = st.radio("Vol source", ["Market IV", "Historical Vol"], key="mc_vol_source")
            sigma_mc = (mc_row["IV_%"] / 100) if vol_source == "Market IV" else hist_vol
            default_steps = max(25, min(252, int(T * 252)))
            n_sims = st.slider("Simulations", 2000, 50000, 12000, 2000, key="mc_sims")
            n_steps = st.slider("Time steps", 25, 252, default_steps, 25, key="mc_steps")
            mc_seed = st.number_input("Seed", value=42, min_value=0, max_value=99999, key="mc_seed")
            use_antithetic = st.checkbox("Antithetic variates", value=True, key="mc_anti")
            n_paths_plot = st.slider("Paths on chart", 10, 200, 80, 10, key="mc_paths_plot")
            run_mc = st.button("Run Monte Carlo", key="run_mc_btn")
        mc_key = (
            ticker, exp_date, opt_type, float(mc_strike), float(sigma_mc),
            int(n_sims), int(n_steps), int(mc_seed), bool(use_antithetic), int(n_paths_plot),
            float(spot), float(T), float(rate), float(div_yield)
        )
        if st.session_state.get("mc_key") != mc_key:
            st.session_state.pop("mc_result", None)
            st.session_state["mc_key"] = mc_key
        if run_mc:
            with st.spinner("Running Monte Carlo simulation..."):
                mc_pricer = MonteCarloPricer(
                    S=spot, K=mc_strike, T=T, r=rate, sigma=sigma_mc, q=div_yield,
                    n_simulations=n_sims, n_steps=n_steps, seed=int(mc_seed)
                )
                paths = mc_pricer._simulate_paths(use_antithetic=use_antithetic)
                final_prices = paths[:, -1]
                payoffs = np.maximum(final_prices - mc_strike, 0.0) if opt_type == "call" else np.maximum(mc_strike - final_prices, 0.0)
                discounted_payoffs = payoffs * np.exp(-rate * T)
                mc_price = float(np.mean(discounted_payoffs))
                mc_std = float(np.std(discounted_payoffs, ddof=1))
                mc_se = mc_std / np.sqrt(len(discounted_payoffs))
                ci95 = 1.96 * mc_se
                running = np.cumsum(discounted_payoffs) / (np.arange(len(discounted_payoffs)) + 1)
                n_points = min(120, len(running))
                idx = np.linspace(0, len(running) - 1, n_points, dtype=int)
                n_keep = min(n_paths_plot, paths.shape[0])
                paths_sample = paths[:n_keep, :]
                time_days = np.linspace(0, T * 365.0, n_steps + 1)
                pct = np.percentile(paths, [5, 10, 25, 50, 75, 90, 95], axis=0)
                st.session_state["mc_result"] = {
                    "mc_price": mc_price, "mc_se": mc_se,
                    "ci_low": mc_price - ci95, "ci_high": mc_price + ci95,
                    "discounted_payoffs": discounted_payoffs,
                    "paths_idx": idx + 1, "running_mean": running[idx],
                    "sigma_mc": sigma_mc, "paths_sample": paths_sample,
                    "time_days": time_days, "final_prices": final_prices,
                    "p05": pct[0], "p10": pct[1], "p25": pct[2], "p50": pct[3],
                    "p75": pct[4], "p90": pct[5], "p95": pct[6],
                }
        with col_mc2:
            mc_state = st.session_state.get("mc_result")
            if mc_state:
                bsm_price_mc = BlackScholes.get_price(
                    spot, mc_strike, T, rate, mc_state["sigma_mc"], opt_type, div_yield
                )
                diff = mc_state["mc_price"] - bsm_price_mc
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("BSM Price", f"${bsm_price_mc:.4f}")
                m2.metric("MC Price", f"${mc_state['mc_price']:.4f}")
                m3.metric("MC - BSM", f"${diff:+.4f}")
                m4.metric("Std Error", f"${mc_state['mc_se']:.5f}")
                st.caption(
                    f"95% CI MC: [{mc_state['ci_low']:.4f}, {mc_state['ci_high']:.4f}] | "
                    f"Vol used: {mc_state['sigma_mc']*100:.2f}%"
                )
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=mc_state["paths_idx"], y=mc_state["running_mean"],
                    mode="lines", name="MC running price", line=dict(color="#3498db", width=2),
                ))
                fig_conv.add_hline(y=bsm_price_mc, line_dash="dash", line_color="orange",
                                   annotation_text=f"BSM {bsm_price_mc:.4f}")
                fig_conv.update_layout(
                    title="Monte Carlo Convergence",
                    xaxis_title="Number of paths", yaxis_title="Option price",
                    template="plotly_white", height=320
                )
                st.plotly_chart(fig_conv, use_container_width=True)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=mc_state["discounted_payoffs"], nbinsx=60,
                    histnorm="percent",
                    name="Discounted payoff distribution",
                    marker_color="#636EFA", opacity=0.75
                ))
                fig_hist.add_vline(x=mc_state["mc_price"], line_color="#2ecc71", line_width=2,
                                   annotation_text=f"MC {mc_state['mc_price']:.4f}")
                fig_hist.add_vline(x=bsm_price_mc, line_color="#e67e22", line_width=2, line_dash="dash",
                                   annotation_text=f"BSM {bsm_price_mc:.4f}")
                fig_hist.update_layout(
                    title="MC Discounted Payoff Distribution",
                    xaxis_title="Discounted payoff", yaxis_title="% of paths",
                    template="plotly_white", height=340, barmode="overlay"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                fig_fan = go.Figure()
                t = mc_state["time_days"]
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p95"], mode="lines", line=dict(width=0), showlegend=False))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p05"], mode="lines", line=dict(width=0),
                                             fill="tonexty", fillcolor="rgba(52,152,219,0.18)", name="P05-P95"))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p90"], mode="lines", line=dict(width=0), showlegend=False))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p10"], mode="lines", line=dict(width=0),
                                             fill="tonexty", fillcolor="rgba(52,152,219,0.28)", name="P10-P90"))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p75"], mode="lines", line=dict(width=0), showlegend=False))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p25"], mode="lines", line=dict(width=0),
                                             fill="tonexty", fillcolor="rgba(52,152,219,0.40)", name="P25-P75"))
                fig_fan.add_trace(go.Scatter(x=t, y=mc_state["p50"], mode="lines",
                                             line=dict(color="#f1c40f", width=2), name="Median (P50)"))
                fig_fan.add_hline(y=mc_strike, line_dash="dash", line_color="orange",
                                  annotation_text=f"Strike {mc_strike:.0f}")
                fig_fan.add_hline(y=spot, line_dash="dot", line_color="gray",
                                  annotation_text=f"Spot {spot:.2f}")
                fig_fan.update_layout(
                    title="Monte Carlo Fan Chart (Percentile Bands)",
                    xaxis_title="Time (days)", yaxis_title="Underlying price",
                    template="plotly_white", height=360
                )
                st.plotly_chart(fig_fan, use_container_width=True)
                fig_st = go.Figure()
                fig_st.add_trace(go.Histogram(
                    x=mc_state["final_prices"], nbinsx=60,
                    histnorm="percent",
                    name="S(T) distribution", marker_color="#7f8c8d", opacity=0.8
                ))
                fig_st.add_vline(x=spot, line_dash="dot", line_color="gray",
                                 annotation_text=f"S0 {spot:.2f}")
                fig_st.add_vline(x=mc_strike, line_dash="dash", line_color="orange",
                                 annotation_text=f"K {mc_strike:.0f}")
                fig_st.update_layout(
                    title="Terminal Price Distribution S(T)",
                    xaxis_title="S(T)", yaxis_title="% of paths",
                    template="plotly_white", height=330
                )
                st.plotly_chart(fig_st, use_container_width=True)
                fig_paths = go.Figure()
                time_days = mc_state["time_days"]
                for i, path in enumerate(mc_state["paths_sample"]):
                    fig_paths.add_trace(go.Scatter(
                        x=time_days, y=path, mode="lines", line=dict(width=1),
                        opacity=0.30, showlegend=False,
                        hovertemplate=f"Path {i+1}<br>Day=%{{x:.0f}}<br>S=%{{y:.2f}}<extra></extra>"
                    ))
                fig_paths.add_hline(y=mc_strike, line_dash="dash", line_color="orange",
                                    annotation_text=f"Strike {mc_strike:.0f}")
                fig_paths.add_hline(y=spot, line_dash="dot", line_color="gray",
                                    annotation_text=f"Spot {spot:.2f}")
                fig_paths.update_layout(
                    title=f"Sample Monte Carlo Price Paths ({len(mc_state['paths_sample'])} paths)",
                    xaxis_title="Time (days)", yaxis_title="Underlying price",
                    template="plotly_white", height=360
                )
                st.plotly_chart(fig_paths, use_container_width=True)


def render_pricing_tab(tab, df, spot, T, rate, div_yield, opt_type):
    """Render EU vs US Pricing tab (American vs European)."""
    with tab:
        am_results = []
        sample_strikes = df["Strike"].tolist()[::max(1, len(df) // 8)]
        with st.spinner("Computing American prices (Binomial 1000 steps + Control Variate)..."):
            for K in sample_strikes:
                row_data = df[df["Strike"] == K].iloc[0]
                iv = row_data["IV_%"] / 100
                euro_price = BlackScholes.get_price(spot, K, T, rate, iv, opt_type, div_yield)
                try:
                    tree = BinomialTree(spot, K, T, rate, iv, opt_type, n_steps=1000, q=div_yield)
                    amer_price = tree.price_american_cv(euro_price)
                except (ValueError, RuntimeError):
                    amer_price = euro_price
                ee_premium = amer_price - euro_price
                am_results.append({
                    "Strike": K, "European (BS)": euro_price,
                    "American (Tree)": amer_price, "EE Premium": ee_premium,
                    "EE Premium %": (ee_premium / euro_price * 100) if euro_price > 0 else 0
                })
        df_am = pd.DataFrame(am_results)
        # Seuil significatif: rouge seulement si EE > $0.05 ou > 1% du prix EU
        ee_threshold = max(0.05, 0.01 * df_am["European (BS)"].max() if len(df_am) else 0.05)
        st.dataframe(
            df_am.style.format({
                "Strike": "${:.2f}", "European (BS)": "${:.4f}",
                "American (Tree)": "${:.4f}", "EE Premium": "${:+.4f}",
                "EE Premium %": "{:+.2f}%"
            }).background_gradient(subset=["EE Premium"], cmap="YlOrRd", vmin=0, vmax=ee_threshold),
            use_container_width=True
        )


def calibrate_market_from_put_call_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot_market: float,
    T: float,
    rate_init: float,
    div_yield_init: float,
    div_yield_forecast: float = 0.0,
    max_spread_pct: float = None,
    n_strikes_near_atm: int = 10,
) -> tuple:
    """
    Calibre r, q, S par regression sur la parite Put-Call: C - P = S*e^(-qT) - K*e^(-rT).
    - Utilise div_yield_forecast (historique) comme point de depart si div_yield_init = 0.
    - Contraint r proche du taux marche (Treasury) et rejette l'excedent dans q.
    """
    if max_spread_pct is None:
        max_spread_pct = MAX_SPREAD_PCT
    q_init = div_yield_init if div_yield_init > 0.001 else max(div_yield_init, div_yield_forecast)
    try:
        calls_c = DataCleaner.clean_option_chain(calls.copy(), min_bid=0.01)
        puts_c = DataCleaner.clean_option_chain(puts.copy(), min_bid=0.01)
        calls_c = calls_c[["strike", "bid", "ask"]].rename(
            columns={"bid": "call_bid", "ask": "call_ask"}
        )
        puts_c = puts_c[["strike", "bid", "ask"]].rename(
            columns={"bid": "put_bid", "ask": "put_ask"}
        )
        merged = pd.merge(calls_c, puts_c, on="strike")
        merged["call_mid"] = (merged["call_bid"] + merged["call_ask"]) / 2
        merged["put_mid"] = (merged["put_bid"] + merged["put_ask"]) / 2
        merged = merged[(merged["call_mid"] > 0) & (merged["put_mid"] > 0)]
        merged["call_spread_pct"] = (merged["call_ask"] - merged["call_bid"]) / merged["call_mid"]
        merged["put_spread_pct"] = (merged["put_ask"] - merged["put_bid"]) / merged["put_mid"]
        merged["cp_market"] = merged["call_mid"] - merged["put_mid"]
        merged["dist_to_atm"] = (merged["strike"] - spot_market).abs()
        merged = merged[
            (merged["call_spread_pct"] <= max_spread_pct) &
            (merged["put_spread_pct"] <= max_spread_pct)
        ]
        merged = merged.nsmallest(n_strikes_near_atm, "dist_to_atm")
        if len(merged) < 5:
            return spot_market, rate_init, q_init

        K_arr = merged["strike"].values.astype(float)
        cp_arr = merged["cp_market"].values.astype(float)

        # r fixe = taux marche (Treasury), optimise q et S. L'excedent va dans le yield.
        def objective_qs(x):
            q, S = x[0], x[1]
            theory = S * np.exp(-q * T) - K_arr * np.exp(-rate_init * T)
            return float(np.sum((cp_arr - theory) ** 2))

        x0_qs = [q_init, spot_market]
        bounds_qs = [
            (0.0, 0.20),
            (spot_market * 0.5, spot_market * 2.0),
        ]
        res = minimize(objective_qs, x0_qs, method="L-BFGS-B", bounds=bounds_qs)
        if res.success:
            q_impl, S_impl = res.x
            if 0 <= q_impl <= 0.20 and spot_market * 0.5 <= S_impl <= spot_market * 2:
                return float(S_impl), float(rate_init), float(q_impl)

        # Fallback: regression complete avec penalite forte sur (r - rate_init)^2
        def objective_full(x):
            r, q, S = x[0], x[1], x[2]
            theory = S * np.exp(-q * T) - K_arr * np.exp(-r * T)
            ssr = np.sum((cp_arr - theory) ** 2)
            penalty = 50000.0 * (r - rate_init) ** 2
            return float(ssr + penalty)

        x0 = [rate_init, q_init, spot_market]
        bounds = [
            (max(0.001, rate_init - 0.02), min(0.20, rate_init + 0.02)),
            (0.0, 0.20),
            (spot_market * 0.5, spot_market * 2.0),
        ]
        res = minimize(objective_full, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return spot_market, rate_init, q_init
        r_impl, q_impl, S_impl = res.x
        if not (0.001 <= r_impl <= 0.20 and 0 <= q_impl <= 0.20 and spot_market * 0.5 <= S_impl <= spot_market * 2):
            return spot_market, rate_init, q_init
        return float(S_impl), float(r_impl), float(q_impl)
    except Exception:
        return spot_market, rate_init, q_init


def render_risk_tab(tab, df, spot, opt_type, atm_idx, T, rate, div_yield, calls, puts, spot_market=None):
    """Render Risk Analysis tab (Scenario + Put-Call Parity).
    spot_market: underlying spot from ticker (for parity check). If None, uses spot (calibrated).
    """
    import logging
    logger = logging.getLogger(__name__)
    spot_underlying = spot_market if spot_market is not None else spot
    with tab:
        tab_scenario, tab_parity = st.tabs(["Scenario Analysis", "Put-Call Parity"])
        with tab_scenario:
            col_sc1, col_sc2 = st.columns([1, 3])
            with col_sc1:
                sc_strike = st.selectbox("Strike", df["Strike"].tolist(),
                                         index=int(atm_idx), key="sc_strike")
                sc_row = df[df["Strike"] == sc_strike].iloc[0]
                sc_iv = sc_row["IV_%"] / 100
                st.caption(
                    f"**Current (Spot):** ${spot_underlying:.2f} | "
                    f"Option Mid: ${sc_row['Mid']:.2f} | IV: {sc_iv*100:.1f}%"
                )
            with col_sc2:
                spot_shocks = np.linspace(-15, 15, 13)
                vol_shocks = np.linspace(-10, 10, 9)
                z_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))
                base_price = BlackScholes.get_price(spot, sc_strike, T, rate, sc_iv, opt_type, div_yield)
                for i, dv in enumerate(vol_shocks):
                    for j, ds in enumerate(spot_shocks):
                        new_spot = spot * (1 + ds / 100)
                        new_vol = max(0.01, sc_iv + dv / 100)
                        new_price = BlackScholes.get_price(
                            new_spot, sc_strike, T, rate, new_vol, opt_type, div_yield)
                        z_matrix[i, j] = new_price - base_price
                fig_scenario = go.Figure(data=go.Heatmap(
                    x=[f"{s:+.0f}%" for s in spot_shocks],
                    y=[f"{v:+.0f}%" for v in vol_shocks],
                    z=z_matrix, colorscale="RdYlGn",
                    colorbar=dict(title="P&L ($)"),
                    text=np.round(z_matrix, 2), texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                fig_scenario.update_layout(
                    title=f"P&L Heatmap: Spot vs Vol Shock (K={sc_strike:.0f})",
                    xaxis_title="Spot Change (%)", yaxis_title="Vol Change (points)",
                    template="plotly_white", height=420
                )
                st.plotly_chart(fig_scenario, use_container_width=True)
        with tab_parity:
            try:
                calls_pc = DataCleaner.clean_option_chain(calls.copy(), min_bid=0.01)
                puts_pc = DataCleaner.clean_option_chain(puts.copy(), min_bid=0.01)
                calls_pc = calls_pc[["strike", "bid", "ask", "volume", "openInterest"]].rename(
                    columns={"bid": "call_bid", "ask": "call_ask", "volume": "call_vol", "openInterest": "call_oi"})
                puts_pc = puts_pc[["strike", "bid", "ask", "volume", "openInterest"]].rename(
                    columns={"bid": "put_bid", "ask": "put_ask", "volume": "put_vol", "openInterest": "put_oi"})
                merged = pd.merge(calls_pc, puts_pc, on="strike")
                merged["call_mid"] = (merged["call_bid"] + merged["call_ask"]) / 2
                merged["put_mid"] = (merged["put_bid"] + merged["put_ask"]) / 2
                merged = merged[(merged["call_mid"] > 0) & (merged["put_mid"] > 0)]
                merged["call_spread_pct"] = (merged["call_ask"] - merged["call_bid"]) / merged["call_mid"]
                merged["put_spread_pct"] = (merged["put_ask"] - merged["put_bid"]) / merged["put_mid"]
                # Use underlying spot for moneyness & parity (consistent with ticker data)
                spot_parity = spot_underlying
                merged["moneyness"] = merged["strike"] / spot_parity  # K/S: 1=ATM, <1=ITM call, >1=ITM put
                merged["dist_to_atm"] = (merged["strike"] - spot_parity).abs()
                # Filtres auto: spread < 15%, top 10 strikes proches ATM
                merged = merged[
                    (merged["call_spread_pct"] <= MAX_SPREAD_PCT) &
                    (merged["put_spread_pct"] <= MAX_SPREAD_PCT)
                ]
                merged = merged.nsmallest(10, "dist_to_atm")
                parity_results = []
                for _, row in merged.iterrows():
                    K_pc = row["strike"]
                    call_mid = row["call_mid"]
                    put_mid = row["put_mid"]
                    market = call_mid - put_mid
                    # EU Theory: C - P = S*e^(-qT) - K*e^(-rT)
                    cp_theory_eu = spot_parity * np.exp(-div_yield * T) - K_pc * np.exp(-rate * T)
                    deviation_eu = market - cp_theory_eu
                    # Bornes americaines: lb = S*exp(-q*T)-K, ub = S-K*exp(-r*T)
                    lb_amer = spot_parity * np.exp(-div_yield * T) - K_pc
                    ub_amer = spot_parity - K_pc * np.exp(-rate * T)
                    if lb_amer > ub_amer:
                        lb_amer, ub_amer = ub_amer, lb_amer
                    breach = (market - lb_amer) if market < lb_amer else ((market - ub_amer) if market > ub_amer else 0.0)
                    # Tolerance: breach compte seulement si |breach| > demi-spread (C-P)
                    call_spread = row["call_ask"] - row["call_bid"]
                    put_spread = row["put_ask"] - row["put_bid"]
                    tolerance = min(
                        max(0.5 * (call_spread + put_spread), BREACH_TOLERANCE_FLOOR),
                        BREACH_TOLERANCE_CAP,
                    )
                    parity_results.append({
                        "Strike": K_pc, "Mny": row["moneyness"],
                        "Call Mid": call_mid, "Put Mid": put_mid,
                        "C-P Market": market, "EU Theory": cp_theory_eu,
                        "EU Dev": deviation_eu, "Amer LB": lb_amer, "Amer UB": ub_amer,
                        "Breach $": breach, "Tolerance": tolerance,
                        "Dev % Spot": (deviation_eu / spot_parity) * 100,
                    })
                if not parity_results:
                    st.info(f"Aucun strike liquide (spread < {MAX_SPREAD_PCT*100:.0f}%) trouvé pour la parité.")
                elif parity_results:
                    df_parity = pd.DataFrame(parity_results)
                    col_pc1, col_pc2 = st.columns(2)
                    with col_pc1:
                        df_display = df_parity.drop(columns=["Tolerance"], errors="ignore")
                        st.dataframe(
                            df_display.style.format({
                                "Strike": "${:.2f}", "Mny": "{:.3f}",
                                "Call Mid": "${:.2f}", "Put Mid": "${:.2f}",
                                "C-P Market": "${:+.2f}", "EU Theory": "${:+.2f}",
                                "EU Dev": "${:+.3f}", "Amer LB": "${:+.3f}",
                                "Amer UB": "${:+.3f}", "Breach $": "${:+.3f}",
                                "Dev % Spot": "{:+.3f}%"
                            }).background_gradient(subset=["EU Dev"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
                              .background_gradient(subset=["Breach $"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5),
                            use_container_width=True, height=350
                        )
                    with col_pc2:
                        fig_pc = go.Figure()
                        fig_pc.add_trace(go.Bar(
                            x=df_parity["Strike"], y=df_parity["Breach $"],
                            marker_color=[
                                "#2ecc71" if abs(d) <= tol else "#e74c3c"
                                for d, tol in zip(df_parity["Breach $"], df_parity["Tolerance"])
                            ]
                        ))
                        fig_pc.add_hline(y=0, line_color="gray")
                        fig_pc.update_layout(
                            title="American Bounds Breach (0 = inside bounds)",
                            xaxis_title="Strike", yaxis_title="Breach ($)",
                            template="plotly_white", height=350
                        )
                        st.plotly_chart(fig_pc, use_container_width=True)
                    avg_dev_eu = df_parity["EU Dev"].abs().mean()
                    # Breach compte seulement si |breach| > tolerance (demi-spread)
                    breach_count = (df_parity["Breach $"].abs() > df_parity["Tolerance"]).sum()
                    breach_rate = float(breach_count / len(df_parity) * 100)
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Filtered strikes", f"{len(df_parity)}")
                    s2.metric("Avg |EU Dev|", f"${avg_dev_eu:.3f}")
                    s3.metric("Amer bounds breach rate", f"{breach_rate:.1f}%")
                    st.caption(
                        f"Breach = hors bornes et |écart| > tolérance "
                        f"(plancher ${BREACH_TOLERANCE_FLOOR:.2f}, plafond ${BREACH_TOLERANCE_CAP:.2f}). "
                        "Mny = K/S (1=ATM, <1=ITM call, >1=ITM put). "
                        "EU Theory: C−P = S·e^(-qT) − K·e^(-rT). Spot = underlying from ticker."
                    )
                    if breach_rate < BREACH_RATE_GOOD:
                        st.success("Desk check: OK")
                    elif breach_rate < BREACH_RATE_WARNING:
                        st.warning("Desk check: some breaches")
                    else:
                        st.error("Desk check: many breaches")
            except (ValueError, KeyError, TypeError) as e:
                logger.debug("Parity check failed: %s", e)


def render_surfaces_section(surface_box, exp_options, ticker, spot, opt_type, rate, div_yield, hist_vol):
    """Render 3D Surfaces (Strike x Maturity) section."""
    import logging
    logger = logging.getLogger(__name__)
    with st.spinner("Building 3D surfaces from multiple expirations..."):
        surface_data = []
        exps_to_use = [e for e in exp_options[:8] if e["days"] > 0]
        for exp_info in exps_to_use:
            exp_d = exp_info["date"]
            biz_d = exp_info.get("biz_days", exp_info["days"])
            T_exp = biz_d / 252.0
            try:
                c_chain, p_chain = DataConnector.get_option_chain(ticker, exp_d)
                chain_exp = (c_chain if opt_type == "call" else p_chain).copy()
                chain_exp = DataCleaner.clean_option_chain(chain_exp, min_bid=0.01, max_spread_pct=0.5)
                chain_exp = DataCleaner.filter_by_moneyness(
                    chain_exp, spot, MONEYNESS_SURFACE_MIN, MONEYNESS_SURFACE_MAX
                )
                chain_exp = DataCleaner.filter_surface_quality(
                    chain_exp, min_volume=MIN_VOLUME,
                )
                for _, row in chain_exp.iterrows():
                    K = row["strike"]
                    iv = row.get("impliedVolatility", None)
                    if iv is None or iv <= 0:
                        continue
                    # IV outlier removal: exclude IV > 80% or < 5%
                    if iv > IV_MAX_PCT or iv < IV_MIN_PCT:
                        continue
                    mid = (row["bid"] + row["ask"]) / 2
                    if mid < MIN_MID_PRICE:
                        continue
                    greeks_3d = BlackScholes.get_all_greeks(
                        spot, K, T_exp, rate, iv, opt_type, div_yield)
                    surface_data.append({
                        "Strike": K, "T": T_exp, "Days": exp_info["days"],
                        "IV": iv * 100,
                        "Delta": greeks_3d["delta"], "Gamma": greeks_3d["gamma"],
                        "Vega": greeks_3d["vega"], "Theta": greeks_3d["theta"],
                        "Vanna": greeks_3d["vanna"], "Volga": greeks_3d["volga"],
                        "Charm": greeks_3d["charm"],
                    })
            except (ValueError, KeyError, TypeError):
                continue
    if len(surface_data) <= 10:
        return
    df_surf = pd.DataFrame(surface_data)
    surface_q_box = surface_box.expander("Surface Quality", expanded=False)
    calendar_violations = 0
    calendar_pairs = 0
    for _, grp in df_surf.groupby("Strike"):
        g = grp.sort_values("T")
        if len(g) < 2:
            continue
        w = (g["IV"].values / 100.0) ** 2 * g["T"].values
        dw = np.diff(w)
        calendar_violations += int(np.sum(dw < -1e-6))
        calendar_pairs += len(dw)
    calendar_violation_rate = (calendar_violations / calendar_pairs * 100.0) if calendar_pairs > 0 else 0.0
    butterfly_violations = 0
    butterfly_checks = 0
    for _, grp in df_surf.groupby("Days"):
        g = grp.sort_values("Strike")
        if len(g) < 3:
            continue
        strikes_g = g["Strike"].values
        ivs_g = g["IV"].values / 100.0
        T_g = float(g["T"].iloc[0])
        prices_g = np.array([
            BlackScholes.get_price(spot, K, T_g, rate, iv, "call", div_yield)
            for K, iv in zip(strikes_g, ivs_g)
        ], dtype=float)
        sec_diff = np.diff(prices_g, 2)
        butterfly_violations += int(np.sum(sec_diff < -1e-6))
        butterfly_checks += len(sec_diff)
    butterfly_violation_rate = (butterfly_violations / butterfly_checks * 100.0) if butterfly_checks > 0 else 0.0
    sq1, sq2, sq3, sq4 = surface_q_box.columns(4)
    sq1.metric("Surface points", f"{len(df_surf)}")
    sq2.metric("Calendar violation rate", f"{calendar_violation_rate:.2f}%")
    sq3.metric("Butterfly violation rate", f"{butterfly_violation_rate:.2f}%")
    sq4.metric("Maturities used", f"{df_surf['Days'].nunique()}")
    if calendar_violation_rate < 5 and butterfly_violation_rate < 5:
        surface_q_box.success("Surface quality: GOOD")
    elif calendar_violation_rate < 15 and butterfly_violation_rate < 15:
        surface_q_box.warning("Surface quality: ACCEPTABLE")
    else:
        surface_q_box.error("Surface quality: NOISY (consider stricter filters / smoothing).")
    strikes_unique = sorted(df_surf["Strike"].unique())
    days_unique = sorted(df_surf["Days"].unique())

    def build_surface_grid(df_s, value_col, smooth=None, iv_cap_pct=None, gaussian_sigma=None):
        """Interpolation RBF + Gaussian smoothing + plafond IV pour eviter pics."""
        from scipy.interpolate import RBFInterpolator
        from scipy.ndimage import gaussian_filter
        if smooth is None:
            smooth = SURFACE_SMOOTH_RBF
        if iv_cap_pct is None:
            iv_cap_pct = IV_CAP_DISPLAY_PCT
        if gaussian_sigma is None:
            gaussian_sigma = SURFACE_GAUSSIAN_SIGMA
        pts = np.column_stack([df_s["Strike"].values, df_s["Days"].values])
        vals = df_s[value_col].values.astype(float)
        xi = np.linspace(min(strikes_unique), max(strikes_unique), 50)
        yi = np.linspace(min(days_unique), max(days_unique), 25)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        grid_pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        try:
            rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=smooth)
            zi_grid = rbf(grid_pts).reshape(xi_grid.shape)
        except Exception:
            from scipy.interpolate import griddata
            zi_grid = griddata(
                pts, vals, (xi_grid, yi_grid), method="cubic"
            )
        zi_grid = np.nan_to_num(zi_grid, nan=0.0, posinf=0.0, neginf=0.0)
        # Gaussian smoothing for IV and Theta surfaces
        if value_col in ("IV", "Theta") and gaussian_sigma > 0:
            zi_grid = gaussian_filter(zi_grid, sigma=gaussian_sigma, mode="nearest")
        if value_col == "IV":
            zi_grid = np.clip(zi_grid, 0, iv_cap_pct)
        return xi, yi, zi_grid

    (tab_iv, tab_delta, tab_gamma, tab_vega, tab_theta,
     tab_vanna3d, tab_volga3d, tab_charm3d, tab_term) = surface_box.tabs(
        ["IV", "Delta", "Gamma", "Vega", "Theta",
         "Vanna", "Volga", "Charm", "Term Structure"]
    )
    surface_configs = [
        (tab_iv, "IV", "Implied Volatility (%)", "Viridis"),
        (tab_delta, "Delta", "Delta", "RdBu"),
        (tab_gamma, "Gamma", "Gamma", "Hot"),
        (tab_vega, "Vega", "Vega", "YlOrRd"),
        (tab_theta, "Theta", "Theta (per day)", "Blues_r"),
        (tab_vanna3d, "Vanna", "Vanna (dDelta/dVol)", "PiYG"),
        (tab_volga3d, "Volga", "Volga (dVega/dVol)", "Spectral"),
        (tab_charm3d, "Charm", "Charm (dDelta/dT per day)", "Cividis"),
    ]
    for tab, col_name, z_label, colorscale in surface_configs:
        with tab:
            try:
                xi, yi, zi = build_surface_grid(
                    df_surf, col_name,
                    iv_cap_pct=IV_CAP_DISPLAY_PCT if col_name == "IV" else 999.0,
                    gaussian_sigma=SURFACE_GAUSSIAN_SIGMA if col_name in ("IV", "Theta") else 0.0,
                )
                scene_cfg = dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Expiry",
                    zaxis_title=z_label,
                )
                if col_name == "IV":
                    scene_cfg["zaxis"] = dict(range=[0, int(IV_CAP_DISPLAY_PCT)], title=z_label)
                fig_3d = go.Figure(data=[go.Surface(
                    x=xi, y=yi, z=zi,
                    colorscale=colorscale,
                    colorbar=dict(title=z_label),
                    cmin=0 if col_name == "IV" else None,
                    cmax=IV_CAP_DISPLAY_PCT if col_name == "IV" else None,
                )])
                fig_3d.update_layout(
                    title=f"{ticker} {col_name} Surface",
                    scene=scene_cfg,
                    height=550, template="plotly_white"
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            except (ValueError, RuntimeError, KeyError) as e_surf:
                logger.warning("Surface build failed for %s: %s", col_name, e_surf)
                st.warning(f"Cannot build {col_name} surface: {e_surf}")
    with tab_term:
        df_ts = pd.DataFrame(surface_data)
        ts_points = []
        for days_val in sorted(df_ts["Days"].unique()):
            sub = df_ts[df_ts["Days"] == days_val]
            if len(sub) == 0:
                continue
            atm_row = sub.iloc[(sub["Strike"] - spot).abs().argsort()[:1]]
            ts_points.append({"Days": days_val, "ATM_IV": float(atm_row["IV"].iloc[0])})
        if ts_points:
            df_term = pd.DataFrame(ts_points)
            fig_term = go.Figure()
            fig_term.add_trace(go.Scatter(
                x=df_term["Days"], y=df_term["ATM_IV"],
                mode="lines+markers",
                marker=dict(size=10, color="#636EFA"), line=dict(width=3)
            ))
            fig_term.add_hline(y=hist_vol * 100, line_dash="dash",
                              line_color="orange",
                              annotation_text=f"Hist Vol: {hist_vol*100:.1f}%")
            fig_term.update_layout(
                title=f"{ticker} ATM IV Term Structure",
                xaxis_title="Days to Expiry",
                yaxis_title="ATM Implied Volatility (%)",
                template="plotly_white", height=450
            )
            st.plotly_chart(fig_term, use_container_width=True)
