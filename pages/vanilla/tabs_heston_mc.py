"""
Onglets Heston et Monte Carlo pour le pricer vanilla.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import BlackScholes
from core.heston import HestonModel
from models import MonteCarloPricer

logger = logging.getLogger(__name__)


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
                    logger.warning("Heston calibration failed: %s", e_cal)
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
            cp5.metric("rho (corr S-vol)", f"{cal_info['rho']:.3f}")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("RMSE IV (all)", f"{cal_info['rmse_iv_pct']:.3f}%")
            cm2.metric("Feller Condition", "OK" if cal_info["feller_satisfied"] else "Violated")
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
                st.success("Heston OK.")
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
                    st.warning("Heston calibration failed.")


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
