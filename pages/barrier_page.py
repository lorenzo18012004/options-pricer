import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import DataCleaner
from core.validation import validate_ticker, validate_barrier_params
from data.exceptions import ValidationError
from config.options import MONEYNESS_MIN, MONEYNESS_MAX
from instruments import BarrierOption
from models import MonteCarloPricer
from services import (
    build_expiration_options,
    get_data_connector,
    load_market_snapshot,
    require_hist_vol_market_only,
)


def render_barrier_pricer():
    """Barrier Option Pricer - live market, robust diagnostics."""
    st.markdown("### Barrier Option Pricer - Live Data")

    from .tickers import POPULAR_TICKERS
    popular_tickers = POPULAR_TICKERS

    a1, a2 = st.columns([3, 1])
    with a1:
        selected = st.selectbox("Select Asset", list(popular_tickers.keys()), key="br_asset")
    if popular_tickers[selected] == "CUSTOM":
        with a2:
            ticker = st.text_input("Ticker", placeholder="AAPL", key="br_custom_ticker").upper()
            if not ticker:
                return
    else:
        ticker = popular_tickers[selected]
        with a2:
            st.metric("Symbol", ticker)

    if st.session_state.get("br_ticker") != ticker:
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("br_")]
        for k in keys_to_clear:
            if k not in ("br_asset", "br_custom_ticker"):
                st.session_state.pop(k, None)
        st.session_state["br_ticker"] = ticker

    if not st.button("Load Barrier Market Data", type="primary", use_container_width=True) and not st.session_state.get("br_loaded"):
        return

    try:
        ticker = validate_ticker(ticker)
    except ValidationError as e:
        st.error(str(e))
        return

    use_synthetic = st.session_state.get("data_source") == "Synthétique"
    connector = get_data_connector(use_synthetic)

    try:
        with st.spinner(f"Loading {ticker}..."):
            spot, expirations, _market_data, div_yield = load_market_snapshot(ticker, connector=connector)
        st.session_state["br_loaded"] = True

        exp_options = build_expiration_options(expirations, max_items=20)
        if not exp_options:
            st.error("No valid option expirations found.")
            return

        selected_exp = st.selectbox("Expiration", [e["label"] for e in exp_options], key="br_exp")
        exp_idx = [e["label"] for e in exp_options].index(selected_exp)
        exp_date = exp_options[exp_idx]["date"]
        cal_days = exp_options[exp_idx]["days"]
        biz_days = exp_options[exp_idx].get("biz_days", cal_days)
        T = max(biz_days / 252.0, 1e-6)
        rate = connector.get_risk_free_rate(T)
        hist_vol = require_hist_vol_market_only(ticker, biz_days, connector=connector)

        option_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="br_opt_type")
        barrier_type = st.selectbox(
            "Barrier Type",
            ["down-and-out", "down-and-in", "up-and-out", "up-and-in"],
            key="br_type"
        )

        with st.spinner(f"Loading {option_type} chain..."):
            calls, puts = connector.get_option_chain(ticker, exp_date)
        chain = (calls if option_type == "call" else puts).copy()
        chain = DataCleaner.clean_option_chain(chain, min_bid=0.01)
        chain = DataCleaner.filter_by_moneyness(chain, spot, MONEYNESS_MIN, MONEYNESS_MAX)
        if len(chain) == 0:
            st.error("No liquid strikes available for selected maturity.")
            return

        chain["Mid"] = (chain["bid"] + chain["ask"]) / 2
        chain = chain[chain["Mid"] > 0].sort_values("strike").reset_index(drop=True)
        if len(chain) == 0:
            st.error("No positive mid quotes.")
            return

        atm_idx = int((chain["strike"] - spot).abs().idxmin())
        strike = float(st.selectbox("Strike", chain["strike"].tolist(), index=atm_idx, key="br_strike"))
        row = chain[chain["strike"] == strike].iloc[0]

        iv_market = float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) and row["impliedVolatility"] > 0 else float(hist_vol)
        vol_source = st.radio("Vol input for model", ["Market IV", "Historical Vol"], horizontal=True, key="br_vol_source")
        sigma = iv_market if vol_source == "Market IV" else float(hist_vol)

        b1, b2, b3 = st.columns(3)
        with b1:
            if "down" in barrier_type:
                barrier_pct = st.slider("Barrier (% of spot)", 50.0, 99.5, 90.0, 0.5, key="br_barrier_down")
            else:
                barrier_pct = st.slider("Barrier (% of spot)", 100.5, 150.0, 110.0, 0.5, key="br_barrier_up")
            barrier = spot * barrier_pct / 100.0
        with b2:
            rebate = st.number_input("Rebate ($)", min_value=0.0, value=0.0, step=0.1, key="br_rebate")
        with b3:
            n_steps = st.slider("MC time steps", 25, 365, min(252, max(25, int(T * 252))), 25, key="br_steps")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_sims = st.slider("MC simulations", 5000, 120000, 30000, 5000, key="br_sims")
        with c2:
            use_antithetic = st.checkbox("Antithetic variates", value=True, key="br_anti")
        with c3:
            run_price = st.button("Price Barrier Option", type="primary", use_container_width=True, key="br_run")

        try:
            validate_barrier_params(spot, barrier, option_type, barrier_type)
        except ValidationError as e:
            st.warning(str(e))

        br_key = (
            ticker, exp_date, option_type, barrier_type, float(spot), float(strike), float(T), float(rate),
            float(sigma), float(barrier), float(rebate), int(n_sims), int(n_steps), bool(use_antithetic), float(div_yield)
        )
        if st.session_state.get("br_key") != br_key:
            st.session_state.pop("br_result", None)
            st.session_state["br_key"] = br_key

        if run_price:
            with st.spinner("Running barrier pricing and diagnostics..."):
                barrier_opt = BarrierOption(
                    spot, strike, T, rate, sigma, barrier, barrier_type, option_type,
                    rebate=rebate, q=div_yield
                )
                result = barrier_opt.price(
                    n_simulations=n_sims, n_steps=n_steps,
                    use_antithetic=use_antithetic, seed=42
                )
                comparison = barrier_opt.compare_with_vanilla(
                    n_simulations=n_sims, n_steps=n_steps,
                    use_antithetic=use_antithetic, seed=42
                )

                parity_gap = np.nan
                try:
                    comp_type = barrier_type.replace("out", "in") if "out" in barrier_type else barrier_type.replace("in", "out")
                    comp_opt = BarrierOption(
                        spot, strike, T, rate, sigma, barrier, comp_type, option_type,
                        rebate=0.0, q=div_yield
                    )
                    comp_res = comp_opt.price(
                        n_simulations=max(4000, n_sims // 2), n_steps=n_steps,
                        use_antithetic=use_antithetic, seed=7
                    )
                    vanilla_ref = comparison["vanilla_price"]
                    parity_gap = (result["price"] + comp_res["price"]) - vanilla_ref
                except (ValueError, RuntimeError, KeyError):
                    pass

                st.session_state["br_result"] = {
                    "result": result,
                    "comparison": comparison,
                    "parity_gap": parity_gap,
                    "spot": spot,
                    "strike": strike,
                    "barrier": barrier,
                    "sigma": sigma,
                    "rate": rate,
                    "T": T,
                    "option_type": option_type,
                    "barrier_type": barrier_type,
                    "div_yield": div_yield,
                    "market_mid": float(row["Mid"]),
                    "steps": n_steps,
                    "sims": n_sims,
                }

        br_state = st.session_state.get("br_result")
        if not br_state:
            return

        result = br_state["result"]
        comparison = br_state["comparison"]
        parity_gap = br_state["parity_gap"]
        market_mid = br_state["market_mid"]

        st.markdown("---")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Barrier Price", f"${result['price']:.4f}")
        m2.metric("Vanilla BS", f"${comparison['vanilla_price']:.4f}")
        m3.metric("Barrier Discount", f"${comparison['discount']:.4f}", f"{comparison['discount_pct']:.2f}%")
        m4.metric("Std Error", f"${result['std_error']:.6f}")
        m5.metric("Barrier Hit Rate", f"{result['activation_rate']*100:.2f}%")
        m6.metric("Vanilla Market Mid", f"${market_mid:.4f}")

        q_penalty = 0.0
        if result["price"] > 1e-10:
            q_penalty += min(40.0, (result["std_error"] / result["price"]) * 1000.0)
        if not np.isnan(parity_gap):
            q_penalty += min(40.0, abs(parity_gap) * 15.0)
        dist = abs(br_state["barrier"] / br_state["spot"] - 1.0) * 100.0
        if dist < 2.0:
            q_penalty += 15.0
        quality = max(0.0, 100.0 - q_penalty)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Barrier Distance", f"{dist:.2f}%")
        q2.metric("In/Out Parity Gap", f"${parity_gap:+.4f}" if not np.isnan(parity_gap) else "N/A")
        q3.metric("Quote vs Vanilla", f"${market_mid - comparison['vanilla_price']:+.4f}")
        q4.metric("Barrier Quality Score", f"{quality:.1f}/100")
        if quality >= 80:
            st.success("Barrier pricing quality: GOOD")
        elif quality >= 60:
            st.warning("Barrier pricing quality: ACCEPTABLE")
        else:
            st.error("Barrier pricing quality: NOISY")

        tab_overview, tab_sens, tab_risk, tab_diag = st.tabs(
            ["Overview", "Sensitivity", "Risk Scenarios", "Diagnostics"]
        )

        with tab_overview:
            info_df = pd.DataFrame([{
                "Ticker": ticker,
                "Expiration": exp_date,
                "Option Type": option_type,
                "Barrier Type": barrier_type,
                "Spot": br_state["spot"],
                "Strike": br_state["strike"],
                "Barrier": br_state["barrier"],
                "Rate_%": br_state["rate"] * 100,
                "DivYield_%": br_state["div_yield"] * 100,
                "Vol_%": br_state["sigma"] * 100,
                "Maturity_days": cal_days,
                "MC_sims": br_state["sims"],
                "MC_steps": br_state["steps"],
            }])
            st.dataframe(
                info_df.style.format({
                    "Spot": "${:.2f}", "Strike": "${:.2f}", "Barrier": "${:.2f}",
                    "Rate_%": "{:.2f}%", "DivYield_%": "{:.2f}%", "Vol_%": "{:.2f}%"
                }),
                use_container_width=True
            )

            x = np.linspace(br_state["spot"] * 0.6, br_state["spot"] * 1.4, 120)
            vanilla_payoff = np.maximum(x - br_state["strike"], 0.0) if option_type == "call" else np.maximum(br_state["strike"] - x, 0.0)
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=x, y=vanilla_payoff, mode="lines", name="Vanilla terminal payoff"))
            fig_pay.add_vline(x=br_state["barrier"], line_dash="dot", line_color="red", annotation_text=f"Barrier {br_state['barrier']:.1f}")
            fig_pay.add_vline(x=br_state["strike"], line_dash="dash", line_color="gray", annotation_text=f"Strike {br_state['strike']:.1f}")
            fig_pay.update_layout(
                title="Terminal payoff reference (path condition not shown)",
                xaxis_title="Spot at expiry", yaxis_title="Payoff",
                template="plotly_white", height=320
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with tab_sens:
            n_pts = st.slider("Sensitivity points", 7, 15, 10, 1, key="br_sens_pts")
            sens_sims = st.slider("Sensitivity MC sims", 2000, 20000, 7000, 1000, key="br_sens_sims")
            if "down" in barrier_type:
                b_min, b_max = br_state["spot"] * 0.60, br_state["spot"] * 0.995
            else:
                b_min, b_max = br_state["spot"] * 1.005, br_state["spot"] * 1.40
            barrier_range = np.linspace(b_min, b_max, n_pts)

            with st.spinner("Computing barrier sensitivity..."):
                prices, hit_rates = [], []
                for b in barrier_range:
                    try:
                        tmp = BarrierOption(
                            br_state["spot"], br_state["strike"], br_state["T"], br_state["rate"], br_state["sigma"],
                            float(b), barrier_type, option_type, rebate=rebate, q=br_state["div_yield"]
                        )
                        rr = tmp.price(n_simulations=sens_sims, n_steps=max(50, br_state["steps"] // 2), use_antithetic=True, seed=11)
                        prices.append(rr["price"])
                        hit_rates.append(rr.get("activation_rate", np.nan) * 100.0)
                    except (ValueError, ZeroDivisionError, RuntimeError):
                        prices.append(np.nan)
                        hit_rates.append(np.nan)

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=barrier_range, y=prices, mode="lines+markers", name="Barrier price"))
            fig_s.add_hline(y=comparison["vanilla_price"], line_dash="dash", line_color="orange", annotation_text="Vanilla BS")
            fig_s.add_vline(x=br_state["barrier"], line_dash="dot", line_color="red", annotation_text="Current barrier")
            fig_s.update_layout(
                title="Price vs Barrier level", xaxis_title="Barrier", yaxis_title="Price",
                template="plotly_white", height=340
            )
            st.plotly_chart(fig_s, use_container_width=True)

            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=barrier_range, y=hit_rates, mode="lines+markers", name="Hit rate (%)"))
            fig_hr.update_layout(
                title="Barrier hit rate vs Barrier level",
                xaxis_title="Barrier", yaxis_title="Hit rate (%)",
                template="plotly_white", height=300
            )
            st.plotly_chart(fig_hr, use_container_width=True)

        with tab_risk:
            spot_shocks = np.linspace(-15, 15, 13)
            vol_shocks = np.linspace(-10, 10, 9)
            z = np.zeros((len(vol_shocks), len(spot_shocks)))
            base_price = result["price"]
            small_sims = max(3000, n_sims // 6)
            for i, dv in enumerate(vol_shocks):
                for j, ds in enumerate(spot_shocks):
                    s_new = br_state["spot"] * (1.0 + ds / 100.0)
                    vol_new = max(0.01, br_state["sigma"] + dv / 100.0)
                    try:
                        tmp = BarrierOption(
                            s_new, br_state["strike"], br_state["T"], br_state["rate"], vol_new,
                            br_state["barrier"], barrier_type, option_type, rebate=rebate, q=br_state["div_yield"]
                        )
                        rr = tmp.price(n_simulations=small_sims, n_steps=max(50, n_steps // 2), use_antithetic=True, seed=19)
                        raw_pnl = rr["price"] - base_price
                        z[i, j] = max(raw_pnl, -base_price)  # Long: floor -premium
                    except (ValueError, ZeroDivisionError, RuntimeError):
                        z[i, j] = np.nan

            fig_risk = go.Figure(data=go.Heatmap(
                x=[f"{x:+.0f}%" for x in spot_shocks],
                y=[f"{v:+.0f}pt" for v in vol_shocks],
                z=z, colorscale="RdYlGn", colorbar=dict(title="P&L ($)")
            ))
            fig_risk.update_layout(
                title="Barrier price P&L heatmap (spot/vol shocks)",
                xaxis_title="Spot shock", yaxis_title="Vol shock",
                template="plotly_white", height=390
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with tab_diag:
            n_paths_plot = st.slider("Paths on chart", 20, 250, 80, 10, key="br_paths")
            diag_steps = max(50, min(252, n_steps))
            diag_sims = max(1000, min(5000, n_sims // 5))
            mc = MonteCarloPricer(
                br_state["spot"], br_state["strike"], br_state["T"], br_state["rate"], br_state["sigma"],
                n_simulations=diag_sims, n_steps=diag_steps, seed=123, q=br_state["div_yield"]
            )
            paths = mc._simulate_paths(use_antithetic=True)
            finals = paths[:, -1]
            barrier_hit = np.min(paths, axis=1) <= br_state["barrier"] if "down" in barrier_type else np.max(paths, axis=1) >= br_state["barrier"]
            if option_type == "call":
                payoff = np.maximum(finals - br_state["strike"], 0.0)
            else:
                payoff = np.maximum(br_state["strike"] - finals, 0.0)
            if "out" in barrier_type:
                payoff = np.where(barrier_hit, rebate, payoff)
            else:
                payoff = np.where(barrier_hit, payoff, 0.0)
            disc_payoff = payoff * np.exp(-br_state["rate"] * br_state["T"])

            fig_paths = go.Figure()
            t_days = np.linspace(0, br_state["T"] * 365.0, diag_steps + 1)
            keep = min(n_paths_plot, paths.shape[0])
            for i in range(keep):
                fig_paths.add_trace(go.Scatter(
                    x=t_days, y=paths[i], mode="lines", showlegend=False, opacity=0.35, line=dict(width=1)
                ))
            fig_paths.add_hline(y=br_state["barrier"], line_dash="dash", line_color="red", annotation_text=f"Barrier {br_state['barrier']:.2f}")
            fig_paths.add_hline(y=br_state["strike"], line_dash="dot", line_color="gray", annotation_text=f"Strike {br_state['strike']:.2f}")
            fig_paths.update_layout(
                title=f"Sample MC paths ({keep})",
                xaxis_title="Time (days)", yaxis_title="Underlying price",
                template="plotly_white", height=350
            )
            st.plotly_chart(fig_paths, use_container_width=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=disc_payoff, nbinsx=60, histnorm="percent",
                                            marker_color="#636EFA", opacity=0.8))
            fig_hist.add_vline(x=float(np.mean(disc_payoff)), line_dash="dash", line_color="black",
                               annotation_text=f"Mean {np.mean(disc_payoff):.4f}")
            fig_hist.update_layout(
                title="Discounted payoff distribution",
                xaxis_title="Discounted payoff", yaxis_title="% of paths",
                template="plotly_white", height=320
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    except ValueError as e:
        if "No options data" in str(e):
            st.error(
                "**Données d'options indisponibles.** Yahoo Finance bloque souvent les requêtes "
                "depuis les serveurs cloud (Streamlit Cloud). L'app fonctionne correctement en local : "
                "lancez `streamlit run app.py` sur votre machine."
            )
        else:
            st.error(f"Error: {str(e)}")
    except (KeyError, TypeError, ImportError) as e:
        st.error(f"Error: {str(e)}")

