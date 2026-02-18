"""
Onglets Volatility, Greeks, P&L, Attribution pour le pricer vanilla.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import BlackScholes


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
                title="Mispricing: BS (IV_Used) vs Market",
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
                          help=f"{n_contracts} x {MULTIPLIER} x ${entry_price:.2f} (received)")

            if opt_type == "call":
                be = pnl_strike + entry_price
            else:
                be = pnl_strike - entry_price
            st.metric("Breakeven", f"${be:.2f}")
            if is_long:
                st.metric("Max Loss", f"${entry_price * MULTIPLIER * n_contracts:,.0f}")
            else:
                if opt_type == "call":
                    st.metric("Max Loss", "Unlimited", help="Short call: unlimited loss if spot rises")
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
