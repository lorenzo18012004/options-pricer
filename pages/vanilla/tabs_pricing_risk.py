"""
Onglets EU vs US Pricing et Risk Analysis pour le pricer vanilla.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import BlackScholes
from data import DataCleaner
from models import BinomialTree

from config.options import (
    MAX_SPREAD_PCT,
    BREACH_TOLERANCE_FLOOR,
    BREACH_TOLERANCE_CAP,
    BREACH_RATE_GOOD,
    BREACH_RATE_WARNING,
)

logger = logging.getLogger(__name__)


def render_pricing_tab(tab, df, spot, T, rate, div_yield, opt_type, ticker=None, exp_date=None, connector=None):
    """Render EU vs US Pricing tab (American vs European)."""
    dividends = []
    if ticker and exp_date and connector and hasattr(connector, "get_dividends_schedule"):
        dividends = connector.get_dividends_schedule(ticker, exp_date)
    with tab:
        am_results = []
        sample_strikes = df["Strike"].tolist()[::max(1, len(df) // 8)]
        with st.spinner("Computing American prices (Binomial 1000 steps + Control Variate)..."):
            for K in sample_strikes:
                row_data = df[df["Strike"] == K].iloc[0]
                iv = row_data["IV_%"] / 100
                euro_price = BlackScholes.get_price(spot, K, T, rate, iv, opt_type, div_yield)
                try:
                    tree = BinomialTree(spot, K, T, rate, iv, opt_type, n_steps=1000, dividends=dividends, q=div_yield)
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
        ee_threshold = max(0.05, 0.01 * df_am["European (BS)"].max() if len(df_am) else 0.05)
        st.dataframe(
            df_am.style.format({
                "Strike": "${:.2f}", "European (BS)": "${:.4f}",
                "American (Tree)": "${:.4f}", "EE Premium": "${:+.4f}",
                "EE Premium %": "{:+.2f}%"
            }).background_gradient(subset=["EE Premium"], cmap="YlOrRd", vmin=0, vmax=ee_threshold),
            use_container_width=True
        )


def render_risk_tab(tab, df, spot, opt_type, atm_idx, T, rate, div_yield, calls, puts, spot_market=None):
    """Render Risk Analysis tab (Scenario + Put-Call Parity)."""
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
            with col_sc2:
                spot_shocks = np.linspace(-15, 15, 13)
                vol_shocks = np.linspace(-10, 10, 9)
                z_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))
                entry_price = float(sc_row["Mid"])
                base_price = BlackScholes.get_price(spot, sc_strike, T, rate, sc_iv, opt_type, div_yield)
                for i, dv in enumerate(vol_shocks):
                    for j, ds in enumerate(spot_shocks):
                        new_spot = spot * (1 + ds / 100)
                        new_vol = max(0.01, sc_iv + dv / 100)
                        new_price = BlackScholes.get_price(
                            new_spot, sc_strike, T, rate, new_vol, opt_type, div_yield)
                        raw_pnl = new_price - entry_price
                        z_matrix[i, j] = max(raw_pnl, -entry_price)
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
                spot_parity = spot_underlying
                merged["moneyness"] = merged["strike"] / spot_parity
                merged["dist_to_atm"] = (merged["strike"] - spot_parity).abs()
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
                    cp_theory_eu = spot_parity * np.exp(-div_yield * T) - K_pc * np.exp(-rate * T)
                    deviation_eu = market - cp_theory_eu
                    lb_amer = spot_parity * np.exp(-div_yield * T) - K_pc
                    ub_amer = spot_parity - K_pc * np.exp(-rate * T)
                    if lb_amer > ub_amer:
                        lb_amer, ub_amer = ub_amer, lb_amer
                    breach = (market - lb_amer) if market < lb_amer else ((market - ub_amer) if market > ub_amer else 0.0)
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
                    st.info(f"No liquid strike (spread < {MAX_SPREAD_PCT*100:.0f}%) found for parity.")
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
                    breach_count = (df_parity["Breach $"].abs() > df_parity["Tolerance"]).sum()
                    breach_rate = float(breach_count / len(df_parity) * 100)
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Filtered strikes", f"{len(df_parity)}")
                    s2.metric("Avg |EU Dev|", f"${avg_dev_eu:.3f}")
                    s3.metric("Amer bounds breach rate", f"{breach_rate:.1f}%")
                    st.caption(
                        f"Breach = out of bounds and |gap| > tolerance "
                        f"(floor ${BREACH_TOLERANCE_FLOOR:.2f}, cap ${BREACH_TOLERANCE_CAP:.2f}). "
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
