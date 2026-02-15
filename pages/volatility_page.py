import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import BlackScholes
from core.heston import HestonModel
from data import DataConnector, DataCleaner
from models import BinomialTree, MonteCarloPricer
from services import (
    build_expiration_options,
    load_market_snapshot,
    require_hist_vol_market_only,
)


def _plot_pnl_profile(spots, pnls, current_spot, breakevens=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spots, y=pnls, mode="lines", name="P&L", line=dict(color="#0066cc", width=3)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=current_spot, line_dash="dot", line_color="red",
                  annotation_text="Current Spot", annotation_position="top")
    if breakevens:
        for be in breakevens:
            fig.add_vline(x=be, line_dash="dot", line_color="green", opacity=0.5)
    fig.update_layout(
        title="P&L at Expiration",
        xaxis_title="Spot Price",
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=400
    )
    return fig


def render_volatility_strategies():
    """Volatility strategies pricer: straddle + strangle."""
    st.markdown("### Volatility Strategies - Live Data")

    popular_tickers = {
        "AAPL - Apple": "AAPL", "MSFT - Microsoft": "MSFT",
        "TSLA - Tesla": "TSLA", "NVDA - NVIDIA": "NVDA",
        "GOOGL - Google": "GOOGL", "AMZN - Amazon": "AMZN",
        "META - Meta": "META", "SPY - S&P 500 ETF": "SPY",
        "QQQ - Nasdaq 100 ETF": "QQQ", "IWM - Russell 2000": "IWM",
        "GLD - Gold ETF": "GLD", "Custom": "CUSTOM"
    }

    top1, top2, top3 = st.columns([2, 2, 1])
    with top1:
        strategy_mode = st.radio("Strategy", ["Straddle", "Strangle"], horizontal=True, key="str_mode")
    with top2:
        selected = st.selectbox("Select Asset", list(popular_tickers.keys()), key="str_asset")
    if popular_tickers[selected] == "CUSTOM":
        with top3:
            ticker = st.text_input("Ticker", placeholder="AAPL", key="str_custom_ticker").upper()
            if not ticker:
                return
    else:
        ticker = popular_tickers[selected]
        with top3:
            st.metric("Symbol", ticker)

    if st.session_state.get("str_ticker") != ticker:
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("str_")]
        for k in keys_to_clear:
            if k not in ("str_asset", "str_custom_ticker", "str_mode"):
                st.session_state.pop(k, None)
        st.session_state["str_ticker"] = ticker

    if not st.button("Load Strategy Market Data", type="primary", use_container_width=True) and not st.session_state.get("str_loaded"):
        return

    try:
        with st.spinner(f"Loading {ticker}..."):
            spot, expirations, _market_data, div_yield = load_market_snapshot(ticker)
        st.session_state["str_loaded"] = True

        exp_options = build_expiration_options(expirations, max_items=20)
        if not exp_options:
            st.error("No valid option expirations found.")
            return

        selected_exp = st.selectbox("Expiration", [e["label"] for e in exp_options], key="str_exp")
        exp_idx = [e["label"] for e in exp_options].index(selected_exp)
        exp_date = exp_options[exp_idx]["date"]
        cal_days = exp_options[exp_idx]["days"]
        T = max(cal_days / 365.0, 1e-6)

        with st.spinner(f"Loading option chain for {exp_date}..."):
            calls, puts = DataConnector.get_option_chain(ticker, exp_date)
        calls = DataCleaner.clean_option_chain(calls.copy(), min_bid=0.01)
        puts = DataCleaner.clean_option_chain(puts.copy(), min_bid=0.01)
        calls = DataCleaner.filter_by_moneyness(calls, spot, 0.80, 1.25)
        puts = DataCleaner.filter_by_moneyness(puts, spot, 0.75, 1.20)
        if len(calls) == 0 or len(puts) == 0:
            st.error("Insufficient liquid calls/puts for strategy construction.")
            return

        rate = DataConnector.get_risk_free_rate(T)
        hist_vol = require_hist_vol_market_only(ticker, cal_days)

        calls_tbl = calls[["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]].copy()
        puts_tbl = puts[["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]].copy()
        calls_tbl["mid"] = (calls_tbl["bid"] + calls_tbl["ask"]) / 2
        puts_tbl["mid"] = (puts_tbl["bid"] + puts_tbl["ask"]) / 2
        calls_tbl["spread_pct"] = np.where(calls_tbl["mid"] > 0, (calls_tbl["ask"] - calls_tbl["bid"]) / calls_tbl["mid"], np.nan)
        puts_tbl["spread_pct"] = np.where(puts_tbl["mid"] > 0, (puts_tbl["ask"] - puts_tbl["bid"]) / puts_tbl["mid"], np.nan)
        calls_tbl = calls_tbl[calls_tbl["mid"] > 0].sort_values("strike").reset_index(drop=True)
        puts_tbl = puts_tbl[puts_tbl["mid"] > 0].sort_values("strike").reset_index(drop=True)
        if len(calls_tbl) == 0 or len(puts_tbl) == 0:
            st.error("No valid call/put mid prices.")
            return

        pairs = pd.merge(
            calls_tbl.rename(columns={"strike": "K", "bid": "call_bid", "ask": "call_ask", "impliedVolatility": "call_iv", "volume": "call_vol", "openInterest": "call_oi", "mid": "call_mid", "spread_pct": "call_spread_pct"}),
            puts_tbl.rename(columns={"strike": "K", "bid": "put_bid", "ask": "put_ask", "impliedVolatility": "put_iv", "volume": "put_vol", "openInterest": "put_oi", "mid": "put_mid", "spread_pct": "put_spread_pct"}),
            on="K",
            how="inner"
        ).sort_values("K").reset_index(drop=True)

        if strategy_mode == "Straddle":
            if len(pairs) == 0:
                st.error("No matching strikes between calls and puts for straddle.")
                return
            atm_idx = int((pairs["K"] - spot).abs().idxmin())
            K = st.selectbox("ATM Strike", pairs["K"].tolist(), index=atm_idx, key="str_k_atm")
            row_c = pairs[pairs["K"] == K].iloc[0]
            row_p = row_c
            K_call = float(K)
            K_put = float(K)
            strategy_label = "Straddle"
        else:
            call_candidates = calls_tbl[calls_tbl["strike"] >= spot]
            put_candidates = puts_tbl[puts_tbl["strike"] <= spot]
            if len(call_candidates) == 0 or len(put_candidates) == 0:
                st.error("Need OTM call and OTM put strikes around spot for strangle.")
                return
            c_idx = int((call_candidates["strike"] - spot).abs().idxmin())
            p_idx = int((put_candidates["strike"] - spot).abs().idxmin())
            c1, c2 = st.columns(2)
            with c1:
                K_put = float(st.selectbox("Put strike (OTM put)", put_candidates["strike"].tolist(), index=p_idx, key="str_k_put"))
            with c2:
                K_call = float(st.selectbox("Call strike (OTM call)", call_candidates["strike"].tolist(), index=c_idx, key="str_k_call"))
            if K_put >= K_call:
                st.error("Invalid strangle: need K_put < K_call.")
                return
            row_c = calls_tbl[calls_tbl["strike"] == K_call].iloc[0]
            row_p = puts_tbl[puts_tbl["strike"] == K_put].iloc[0]
            strategy_label = "Strangle"

        iv_candidates = [row_c.get("impliedVolatility", np.nan), row_p.get("impliedVolatility", np.nan)]
        iv_candidates = [x for x in iv_candidates if pd.notna(x) and x > 0]
        iv_market = float(np.mean(iv_candidates)) if iv_candidates else float(hist_vol)
        vol_source = st.radio("Vol input for model", ["Market IV avg", "Historical Vol"], horizontal=True, key="str_vol_source")
        sigma = iv_market if vol_source == "Market IV avg" else float(hist_vol)

        call_bid, call_ask, call_mid = float(row_c["bid"]), float(row_c["ask"]), float(row_c["mid"])
        put_bid, put_ask, put_mid = float(row_p["bid"]), float(row_p["ask"]), float(row_p["mid"])
        premium_market = call_mid + put_mid

        exec_box = st.expander("Execution Assumptions", expanded=False)
        ex1, ex2 = exec_box.columns([3, 2])
        with ex1:
            execution_mode = st.radio("Entry marking mode", ["Mid entry (fair)", "Ask entry (conservative)"], horizontal=True, key="str_exec_mode")
        with ex2:
            slippage_bps = st.slider("Extra slippage (bps)", 0, 100, 0, 5, key="str_exec_slippage")
        raw_entry = premium_market if execution_mode == "Mid entry (fair)" else (call_ask + put_ask)
        premium_ref = raw_entry * (1.0 + slippage_bps / 10000.0)

        call_bs = BlackScholes.get_price(spot, K_call, T, rate, sigma, "call", div_yield)
        put_bs = BlackScholes.get_price(spot, K_put, T, rate, sigma, "put", div_yield)
        premium_bsm = float(call_bs + put_bs)

        call_g = BlackScholes.get_all_greeks(spot, K_call, T, rate, sigma, "call", div_yield)
        put_g = BlackScholes.get_all_greeks(spot, K_put, T, rate, sigma, "put", div_yield)
        pos_delta = call_g["delta"] + put_g["delta"]
        pos_gamma = call_g["gamma"] + put_g["gamma"]
        pos_vega = call_g["vega"] + put_g["vega"]
        pos_theta = call_g["theta"] + put_g["theta"]
        pos_vanna = call_g["vanna"] + put_g["vanna"]
        pos_volga = call_g["volga"] + put_g["volga"]

        lower_be = K_put - premium_ref
        upper_be = K_call + premium_ref
        move_required = (premium_ref / max(spot, 1e-8)) * 100.0

        st.markdown("---")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Strategy", strategy_label)
        m2.metric("Spot", f"${spot:.2f}")
        m3.metric("K_put / K_call", f"{K_put:.1f} / {K_call:.1f}")
        m4.metric("Entry Premium Used", f"${premium_ref:.3f}")
        m5.metric("BSM - Entry", f"${premium_bsm - premium_ref:+.3f}")
        m6.metric("Move Required", f"{move_required:.2f}%")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Lower Breakeven", f"${lower_be:.2f}")
        b2.metric("Upper Breakeven", f"${upper_be:.2f}")
        b3.metric("Rate", f"{rate*100:.2f}%")
        b4.metric("Dividend Yield", f"{div_yield*100:.2f}%")
        exec_box.caption(f"Entry premium = ${raw_entry:.3f} adjusted by {slippage_bps} bps -> ${premium_ref:.3f}")

        tabs = st.tabs(["Legs & Greeks", "P&L at Expiry", "P&L Attribution", "Model Comparison", "Risk"])

        with tabs[0]:
            legs_df = pd.DataFrame([
                {
                    "Leg": "Call", "Strike": K_call, "Bid": call_bid, "Ask": call_ask, "Mid": call_mid,
                    "IV_%": row_c["impliedVolatility"] * 100 if pd.notna(row_c["impliedVolatility"]) else np.nan,
                    "Spread_%": row_c["spread_pct"] * 100 if pd.notna(row_c["spread_pct"]) else np.nan,
                    "Volume": row_c["volume"], "OpenInt": row_c["openInterest"],
                    "BSM_Price": call_bs, "Delta": call_g["delta"], "Gamma": call_g["gamma"], "Vega": call_g["vega"],
                    "Theta": call_g["theta"], "Vanna": call_g["vanna"], "Volga": call_g["volga"],
                },
                {
                    "Leg": "Put", "Strike": K_put, "Bid": put_bid, "Ask": put_ask, "Mid": put_mid,
                    "IV_%": row_p["impliedVolatility"] * 100 if pd.notna(row_p["impliedVolatility"]) else np.nan,
                    "Spread_%": row_p["spread_pct"] * 100 if pd.notna(row_p["spread_pct"]) else np.nan,
                    "Volume": row_p["volume"], "OpenInt": row_p["openInterest"],
                    "BSM_Price": put_bs, "Delta": put_g["delta"], "Gamma": put_g["gamma"], "Vega": put_g["vega"],
                    "Theta": put_g["theta"], "Vanna": put_g["vanna"], "Volga": put_g["volga"],
                },
            ])
            st.dataframe(
                legs_df.style.format({
                    "Strike": "${:.2f}", "Bid": "${:.3f}", "Ask": "${:.3f}", "Mid": "${:.3f}",
                    "IV_%": "{:.2f}%", "Spread_%": "{:.1f}%", "BSM_Price": "${:.4f}",
                    "Delta": "{:+.4f}", "Gamma": "{:.6f}", "Vega": "{:.4f}",
                    "Theta": "{:+.4f}", "Vanna": "{:+.5f}", "Volga": "{:.5f}"
                }),
                use_container_width=True
            )
            g1, g2, g3, g4, g5, g6 = st.columns(6)
            g1.metric("Position Delta", f"{pos_delta:+.4f}")
            g2.metric("Position Gamma", f"{pos_gamma:.6f}")
            g3.metric("Position Vega", f"{pos_vega:.4f}")
            g4.metric("Position Theta", f"{pos_theta:+.4f}")
            g5.metric("Position Vanna", f"{pos_vanna:+.5f}")
            g6.metric("Position Volga", f"{pos_volga:.5f}")

        with tabs[1]:
            def payoff_total(s):
                return max(s - K_call, 0.0) + max(K_put - s, 0.0)
            spots_range = np.linspace(spot * 0.60, spot * 1.40, 160)
            pnls = [payoff_total(s) - premium_ref for s in spots_range]
            fig = _plot_pnl_profile(spots_range, pnls, spot, [lower_be, upper_be])
            st.plotly_chart(fig, use_container_width=True)

            scenarios = []
            for move_pct in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
                final_spot = spot * (1 + move_pct / 100.0)
                pnl = payoff_total(final_spot) - premium_ref
                ret = (pnl / premium_ref * 100.0) if premium_ref > 0 else 0.0
                scenarios.append({"Move_%": move_pct, "Final Spot": final_spot, "P&L": pnl, "Return_%": ret})
            sc_df = pd.DataFrame(scenarios)
            st.dataframe(
                sc_df.style.format({
                    "Move_%": "{:+.0f}", "Final Spot": "${:.2f}", "P&L": "${:+.3f}", "Return_%": "{:+.1f}%"
                }).background_gradient(subset=["P&L"], cmap="RdYlGn"),
                use_container_width=True
            )

        with tabs[2]:
            c1, c2, c3 = st.columns(3)
            with c1:
                spot_shock = st.slider("Spot shock (%)", -20, 20, 0, 1, key="str_attr_spot")
            with c2:
                vol_shock = st.slider("Vol shock (pts)", -20, 20, 0, 1, key="str_attr_vol")
            with c3:
                max_days_passed = max(1, cal_days - 1)
                if max_days_passed <= 1:
                    days_passed = 1
                else:
                    days_passed = st.slider("Days passed", 1, max_days_passed, min(5, max_days_passed), 1, key="str_attr_days")

            new_spot = spot * (1.0 + spot_shock / 100.0)
            new_sigma = max(0.01, sigma + vol_shock / 100.0)
            attr_call = BlackScholes.pnl_attribution(spot, new_spot, sigma, new_sigma, K_call, T, rate, "call", div_yield, days_passed)
            attr_put = BlackScholes.pnl_attribution(spot, new_spot, sigma, new_sigma, K_put, T, rate, "put", div_yield, days_passed)

            total_actual = attr_call["actual_pnl"] + attr_put["actual_pnl"]
            total_delta = attr_call["delta_pnl"] + attr_put["delta_pnl"]
            total_gamma = attr_call["gamma_pnl"] + attr_put["gamma_pnl"]
            total_vega = attr_call["vega_pnl"] + attr_put["vega_pnl"]
            total_theta = attr_call["theta_pnl"] + attr_put["theta_pnl"]
            total_vanna = attr_call["vanna_pnl"] + attr_put["vanna_pnl"]
            total_explained = attr_call["explained_pnl"] + attr_put["explained_pnl"]
            total_unexplained = attr_call["unexplained_pnl"] + attr_put["unexplained_pnl"]
            ratio = (total_explained / total_actual * 100.0) if abs(total_actual) > 1e-12 else None

            fig_attr = go.Figure()
            comp_names = ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Unexplained"]
            comp_vals = [total_delta, total_gamma, total_vega, total_theta, total_vanna, total_unexplained]
            fig_attr.add_trace(go.Bar(
                x=comp_names, y=comp_vals,
                marker_color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#95a5a6"],
                text=[f"${v:+.3f}" for v in comp_vals], textposition="outside"
            ))
            fig_attr.add_hline(y=0, line_color="gray")
            fig_attr.add_hline(y=total_actual, line_dash="dash", line_color="black",
                               annotation_text=f"Actual P&L: ${total_actual:+.3f}")
            fig_attr.update_layout(
                title=f"{strategy_label} P&L Attribution (S {spot:.1f}->{new_spot:.1f}, vol {sigma*100:.1f}%->{new_sigma*100:.1f}%, {days_passed}d)",
                yaxis_title="P&L ($)", template="plotly_white", height=400
            )
            st.plotly_chart(fig_attr, use_container_width=True)
            a1, a2, a3 = st.columns(3)
            a1.metric("Actual P&L", f"${total_actual:+.4f}")
            a2.metric("Explained P&L", f"${total_explained:+.4f}")
            a3.metric("Explanation Ratio", f"{ratio:.1f}%" if ratio is not None else "N/A")

        with tabs[3]:
            p_rows = [
                {"Model": "Market Mid (Call+Put)", "Price": premium_market, "Vs Market": 0.0},
                {"Model": "BSM (European)", "Price": premium_bsm, "Vs Market": premium_bsm - premium_market},
            ]

            n_tree = st.slider("Binomial steps", 100, 500, 200, 50, key="str_tree_steps")
            try:
                tree_call = BinomialTree(spot, K_call, T, rate, sigma, "call", n_steps=n_tree, q=div_yield).price()
                tree_put = BinomialTree(spot, K_put, T, rate, sigma, "put", n_steps=n_tree, q=div_yield).price()
                tree_price = float(tree_call + tree_put)
                p_rows.append({"Model": "Binomial (American)", "Price": tree_price, "Vs Market": tree_price - premium_market})
            except (ValueError, RuntimeError):
                pass

            c_mc1, c_mc2 = st.columns([1, 2])
            with c_mc1:
                n_sims = st.slider("MC simulations", 2000, 50000, 12000, 2000, key="str_mc_sims")
                n_steps = st.slider("MC time steps", 25, 252, max(25, min(252, int(T * 252))), 25, key="str_mc_steps")
                use_antithetic = st.checkbox("MC antithetic variates", value=True, key="str_mc_anti")
                run_mc = st.button(f"Run MC ({strategy_label})", key="str_run_mc")

            mc_key = (ticker, exp_date, strategy_label, float(K_put), float(K_call), float(sigma), int(n_sims), int(n_steps), bool(use_antithetic))
            if st.session_state.get("str_mc_key") != mc_key:
                st.session_state.pop("str_mc_result", None)
                st.session_state["str_mc_key"] = mc_key
            if run_mc:
                with st.spinner(f"Running MC for {strategy_label.lower()}..."):
                    mc_pricer = MonteCarloPricer(S=spot, K=spot, T=T, r=rate, q=div_yield, sigma=sigma,
                                                 n_simulations=n_sims, n_steps=n_steps, seed=42)
                    paths = mc_pricer._simulate_paths(use_antithetic=use_antithetic)
                    finals = paths[:, -1]
                    discounted = (np.maximum(finals - K_call, 0.0) + np.maximum(K_put - finals, 0.0)) * np.exp(-rate * T)
                    mc_price = float(np.mean(discounted))
                    mc_std = float(np.std(discounted, ddof=1))
                    mc_se = mc_std / np.sqrt(len(discounted))
                    ci95 = 1.96 * mc_se
                    running = np.cumsum(discounted) / (np.arange(len(discounted)) + 1)
                    idx = np.linspace(0, len(running) - 1, min(120, len(running)), dtype=int)
                    st.session_state["str_mc_result"] = {
                        "price": mc_price, "se": mc_se, "ci_low": mc_price - ci95, "ci_high": mc_price + ci95,
                        "x": idx + 1, "y": running[idx], "finals": finals
                    }

            mc_state = st.session_state.get("str_mc_result")
            with c_mc2:
                if mc_state:
                    p_rows.append({"Model": "Monte Carlo (European)", "Price": mc_state["price"], "Vs Market": mc_state["price"] - premium_market})
                    cma, cmb, cmc = st.columns(3)
                    cma.metric("MC Price", f"${mc_state['price']:.4f}")
                    cmb.metric("MC Std Error", f"${mc_state['se']:.5f}")
                    cmc.metric("95% CI", f"[{mc_state['ci_low']:.4f}, {mc_state['ci_high']:.4f}]")
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(x=mc_state["x"], y=mc_state["y"], mode="lines", name="MC running"))
                    fig_conv.add_hline(y=premium_bsm, line_dash="dash", line_color="orange", annotation_text=f"BSM {premium_bsm:.4f}")
                    fig_conv.update_layout(
                        title=f"{strategy_label} Monte Carlo Convergence",
                        xaxis_title="Number of paths", yaxis_title="Price",
                        template="plotly_white", height=300
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
                else:
                    pass

            run_heston = st.button("Run Heston calibration (optional)", key="str_run_heston")
            heston_key = (ticker, exp_date, float(spot), float(T), float(rate), int(len(calls_tbl)))
            if st.session_state.get("str_heston_key") != heston_key:
                st.session_state.pop("str_heston_result", None)
                st.session_state["str_heston_key"] = heston_key
            if run_heston:
                with st.spinner("Calibrating Heston (calls smile)..."):
                    try:
                        cal_df = calls_tbl[(calls_tbl["impliedVolatility"] > 0) & (calls_tbl["mid"] > 0)]
                        strikes_h = cal_df["strike"].values
                        ivs_h = cal_df["impliedVolatility"].values
                        spreads_h = np.where(cal_df["mid"].values > 0, (cal_df["ask"].values - cal_df["bid"].values) / cal_df["mid"].values, np.nan)
                        oi_h = cal_df["openInterest"].values
                        heston_model, cal_info = HestonModel.calibrate(
                            strikes_h, ivs_h, spot, T, rate, div_yield, "call",
                            spreads=spreads_h, open_interests=oi_h, moneyness_range=(0.85, 1.15), max_points=25
                        )
                        h_call = heston_model.get_price(spot, K_call, T, rate, "call", div_yield)
                        h_put = heston_model.get_price(spot, K_put, T, rate, "put", div_yield)
                        st.session_state["str_heston_result"] = {"success": True, "price": float(h_call + h_put), "info": cal_info}
                    except (ValueError, RuntimeError, TypeError) as e_h:
                        st.session_state["str_heston_result"] = {"success": False, "error": str(e_h)}

            h_state = st.session_state.get("str_heston_result")
            if h_state:
                if h_state.get("success"):
                    p_rows.append({"Model": "Heston (European)", "Price": h_state["price"], "Vs Market": h_state["price"] - premium_market})
                    info = h_state["info"]
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Heston RMSE IV", f"{info.get('rmse_iv_filtered_pct', info.get('rmse_iv_pct', np.nan)):.3f}%")
                    h2.metric("Points used", f"{info.get('n_points_used', 0)}")
                    h3.metric("Boundary hits", f"{info.get('boundary_hits', 0)}")
                else:
                    pass

            df_models = pd.DataFrame(p_rows)
            st.dataframe(
                df_models.style.format({"Price": "${:.4f}", "Vs Market": "${:+.4f}"})
                .background_gradient(subset=["Vs Market"], cmap="RdYlGn", vmin=-1.0, vmax=1.0),
                use_container_width=True
            )
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(x=df_models["Model"], y=df_models["Price"], marker_color="#3498db"))
            fig_cmp.add_hline(y=premium_market, line_dash="dash", line_color="black", annotation_text="Market Mid")
            fig_cmp.update_layout(
                title=f"{strategy_label} Price by Model",
                xaxis_title="Model", yaxis_title="Price ($)",
                template="plotly_white", height=340
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        with tabs[4]:
            spot_shocks = np.linspace(-15, 15, 13)
            vol_shocks = np.linspace(-10, 10, 9)
            z_mat = np.zeros((len(vol_shocks), len(spot_shocks)))
            base = premium_bsm
            for i, dv in enumerate(vol_shocks):
                for j, ds in enumerate(spot_shocks):
                    s_new = spot * (1.0 + ds / 100.0)
                    v_new = max(0.01, sigma + dv / 100.0)
                    c_new = BlackScholes.get_price(s_new, K_call, T, rate, v_new, "call", div_yield)
                    p_new = BlackScholes.get_price(s_new, K_put, T, rate, v_new, "put", div_yield)
                    z_mat[i, j] = (c_new + p_new) - base
            fig_heat = go.Figure(data=go.Heatmap(
                x=[f"{x:+.0f}%" for x in spot_shocks],
                y=[f"{y:+.0f}%" for y in vol_shocks],
                z=z_mat, colorscale="RdYlGn", colorbar=dict(title="P&L ($)")
            ))
            fig_heat.update_layout(
                title=f"{strategy_label} P&L Heatmap (Spot vs Vol shocks)",
                xaxis_title="Spot shock", yaxis_title="Vol shock (pts)",
                template="plotly_white", height=390
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            hedge_box = st.expander("Delta-Hedging Simulator (Discrete)", expanded=False)
            h1, h2, h3 = hedge_box.columns(3)
            with h1:
                n_paths_hedge = st.slider("Hedge paths", 200, 5000, 1000, 200, key="str_hedge_paths")
            with h2:
                n_rehedges = st.slider("Re-hedges to expiry", 2, 80, min(20, max(2, cal_days)), 2, key="str_hedge_rebals")
            with h3:
                realized_vol_source = st.radio("Realized vol for path generation", ["Model vol", "Historical vol"], key="str_hedge_realized_vol")
            realized_sigma = sigma if realized_vol_source == "Model vol" else float(hist_vol)
            run_hedge = hedge_box.button("Run delta-hedging simulation", key="str_run_hedge")

            hedge_key = (ticker, exp_date, strategy_label, K_put, K_call, float(sigma), float(realized_sigma), int(n_paths_hedge), int(n_rehedges))
            if st.session_state.get("str_hedge_key") != hedge_key:
                st.session_state.pop("str_hedge_result", None)
                st.session_state["str_hedge_key"] = hedge_key
            if run_hedge:
                with st.spinner(f"Simulating delta-hedged vs unhedged {strategy_label.lower()}..."):
                    rng = np.random.default_rng(42)
                    dt_h = T / n_rehedges
                    zz = rng.standard_normal((n_paths_hedge, n_rehedges))
                    drift = (rate - div_yield - 0.5 * realized_sigma ** 2) * dt_h
                    diff = realized_sigma * np.sqrt(dt_h)
                    growth = np.exp(drift + diff * zz)
                    s_paths = np.zeros((n_paths_hedge, n_rehedges + 1))
                    s_paths[:, 0] = spot
                    for t_idx in range(1, n_rehedges + 1):
                        s_paths[:, t_idx] = s_paths[:, t_idx - 1] * growth[:, t_idx - 1]

                    v0_call = BlackScholes.get_price(spot, K_call, T, rate, sigma, "call", div_yield)
                    v0_put = BlackScholes.get_price(spot, K_put, T, rate, sigma, "put", div_yield)
                    v0 = v0_call + v0_put
                    d0_call = BlackScholes.get_delta(spot, K_call, T, rate, sigma, "call", div_yield)
                    d0_put = BlackScholes.get_delta(spot, K_put, T, rate, sigma, "put", div_yield)
                    d0 = d0_call + d0_put

                    unhedged_pnl = np.zeros(n_paths_hedge)
                    hedged_pnl = np.zeros(n_paths_hedge)
                    for p_idx in range(n_paths_hedge):
                        delta_prev = d0
                        hedge_pnl = 0.0
                        s_prev = s_paths[p_idx, 0]
                        v_now = v0
                        for t_idx in range(1, n_rehedges + 1):
                            s_now = s_paths[p_idx, t_idx]
                            hedge_pnl += -delta_prev * (s_now - s_prev)
                            tau = max(T - t_idx * dt_h, 0.0)
                            if tau > 1e-8:
                                c_now = BlackScholes.get_price(s_now, K_call, tau, rate, sigma, "call", div_yield)
                                p_now = BlackScholes.get_price(s_now, K_put, tau, rate, sigma, "put", div_yield)
                                v_now = c_now + p_now
                                d_call = BlackScholes.get_delta(s_now, K_call, tau, rate, sigma, "call", div_yield)
                                d_put = BlackScholes.get_delta(s_now, K_put, tau, rate, sigma, "put", div_yield)
                                delta_prev = d_call + d_put
                            else:
                                v_now = max(s_now - K_call, 0.0) + max(K_put - s_now, 0.0)
                                delta_prev = 0.0
                            s_prev = s_now
                        unhedged_pnl[p_idx] = v_now - v0
                        hedged_pnl[p_idx] = unhedged_pnl[p_idx] + hedge_pnl
                    st.session_state["str_hedge_result"] = {"unhedged": unhedged_pnl, "hedged": hedged_pnl, "realized_sigma": realized_sigma}

            hedge_state = st.session_state.get("str_hedge_result")
            if hedge_state:
                unhd = hedge_state["unhedged"]
                hdg = hedge_state["hedged"]
                hm1, hm2, hm3, hm4 = hedge_box.columns(4)
                hm1.metric("Unhedged P&L Mean", f"${float(np.mean(unhd)):+.3f}")
                hm2.metric("Hedged P&L Mean", f"${float(np.mean(hdg)):+.3f}")
                hm3.metric("Unhedged P&L Std", f"${float(np.std(unhd, ddof=1)):.3f}")
                hm4.metric("Hedged P&L Std", f"${float(np.std(hdg, ddof=1)):.3f}")
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=unhd, nbinsx=60, name="Unhedged", opacity=0.60, marker_color="#e67e22"))
                fig_h.add_trace(go.Histogram(x=hdg, nbinsx=60, name="Delta-hedged", opacity=0.60, marker_color="#3498db"))
                fig_h.update_layout(
                    title=f"Distribution of P&L: Unhedged vs Delta-Hedged {strategy_label}",
                    xaxis_title="P&L ($)", yaxis_title="Frequency", template="plotly_white", barmode="overlay", height=350
                )
                hedge_box.plotly_chart(fig_h, use_container_width=True)

            avg_spread = float(np.nanmean([row_c["spread_pct"], row_p["spread_pct"]]) * 100.0)
            avg_oi = float(np.nanmean([row_c["openInterest"], row_p["openInterest"]]))
            parity_gap = np.nan
            if strategy_label == "Straddle":
                parity_gap = abs((call_mid - put_mid) - (spot * np.exp(-div_yield * T) - K_call * np.exp(-rate * T)))
            penalty = min(30.0, max(0.0, avg_spread - 8.0) * 2.0)
            if not np.isnan(parity_gap):
                penalty += min(30.0, parity_gap * 10.0)
            if avg_oi < 100:
                penalty += 20.0
            quality_score = max(0.0, 100.0 - penalty)

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Avg Spread (2 legs)", f"{avg_spread:.2f}%")
            q2.metric("Avg Open Interest", f"{avg_oi:.0f}")
            q3.metric("Parity Gap (ATM only)", f"${parity_gap:.3f}" if not np.isnan(parity_gap) else "N/A")
            q4.metric("Quality Score", f"{quality_score:.1f}/100")

            rl1, rl2, rl3 = st.columns(3)
            rl1.metric("Entry Cost / Spot", f"{(premium_ref / max(spot, 1e-8)) * 100:.2f}%")
            rl2.metric("Call Spread", f"{(row_c['spread_pct'] * 100):.2f}%")
            rl3.metric("Put Spread", f"{(row_p['spread_pct'] * 100):.2f}%")
            if quality_score >= 80:
                st.success("Execution quality: GOOD")
            elif quality_score >= 60:
                st.warning("Execution quality: ACCEPTABLE")
            else:
                st.error("Execution quality: NOISY")

            mc_state = st.session_state.get("str_mc_result")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

