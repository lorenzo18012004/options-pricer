import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from core import BlackScholes, IVsolver
from core.validation import validate_ticker
from data.exceptions import ValidationError
from models import VolatilitySurface
from data import DataCleaner
from config.options import MAX_SPREAD_PCT, MONEYNESS_MIN, MONEYNESS_MAX
from services import (
    build_expiration_options,
    get_data_connector,
    load_market_snapshot,
    require_hist_vol_market_only,
)
from .vanilla_helpers import (
    calibrate_market_from_put_call_parity,
    render_volatility_tab,
    render_greeks_tabs,
    render_pnl_tab,
    render_attribution_tab,
    render_heston_tab,
    render_mc_tab,
    render_pricing_tab,
    render_risk_tab,
    render_surfaces_section,
    render_market_summary,
    render_data_quality,
    render_chain_display,
)


def render_vanilla_option_pricer():
    """Vanilla Option Pricing - Complete Analysis with Live Data"""
    st.markdown("### Vanilla Option Pricer - Live Data")

    # --- Asset & Expiration Selection ---
    from .tickers import POPULAR_TICKERS
    popular_tickers = POPULAR_TICKERS

    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected = st.selectbox("Select Asset", list(popular_tickers.keys()))
    if popular_tickers[selected] == "CUSTOM":
        with col_sel2:
            ticker = st.text_input("Ticker", placeholder="AAPL").upper()
            if not ticker:
                return
    else:
        ticker = popular_tickers[selected]
        with col_sel2:
            st.metric("Symbol", ticker)

    if not st.button("Load Market Data", type="primary", use_container_width=True) and "vp_data" not in st.session_state:
        return

    try:
        ticker = validate_ticker(ticker)
    except ValidationError as e:
        st.error(str(e))
        return

    use_synthetic = st.session_state.get("data_source") == "Synthetic"
    connector = get_data_connector(use_synthetic)

    try:
        # --- Fetch all data ---
        with st.spinner(f"Loading {ticker}..."):
            spot, expirations, _market_data, div_yield = load_market_snapshot(ticker, connector=connector)

        st.session_state["vp_data"] = True

        # --- Expiration picker (show all, no pre-filtering) ---
        exp_options = build_expiration_options(expirations, max_items=20)

        if not exp_options:
            st.error("No valid expirations")
            return

        selected_exp = st.selectbox("Expiration", [e["label"] for e in exp_options])
        sel_idx = [e["label"] for e in exp_options].index(selected_exp)
        exp_date = exp_options[sel_idx]["date"]
        cal_days_to_exp = exp_options[sel_idx]["days"]
        biz_days_to_exp = exp_options[sel_idx].get("biz_days", cal_days_to_exp)
        if biz_days_to_exp is None or biz_days_to_exp < 1:
            try:
                today_np = np.datetime64(datetime.now().date())
                exp_np = np.datetime64(datetime.strptime(exp_date, "%Y-%m-%d").date())
                biz_days_to_exp = max(1, int(np.busday_count(today_np, exp_np)))
            except (ValueError, TypeError):
                biz_days_to_exp = max(1, cal_days_to_exp)

        # Convention equity options : tout sur 252 (jours ouvrés, vol annualisee)
        day_count_basis = "ACT/252"
        maturity_mode = "Business days"
        days_to_exp = max(1, biz_days_to_exp)
        T = days_to_exp / 252.0

        # Fetch rate, vol, chain (spot refetch apres chain pour sync avec quotes)
        with st.spinner("Loading options data..."):
            rate_init = connector.get_risk_free_rate(T)
            if T > 1.0:
                rate_init = max(rate_init, 0.035)
            hist_vol = require_hist_vol_market_only(ticker, max(1, biz_days_to_exp), connector=connector)
            calls, puts, spot_market = connector.get_option_chain_with_synced_spot(ticker, exp_date)
            div_yield_forecast = connector.get_dividend_yield_forecast(ticker, spot_market)
            spot, rate, div_yield = calibrate_market_from_put_call_parity(
                calls, puts, spot_market, T, rate_init, div_yield,
                div_yield_forecast=div_yield_forecast,
                max_spread_pct=MAX_SPREAD_PCT, n_strikes_near_atm=10,
            )
        st.success(
            f"**Market Calibrated** | Implied Rate: {rate*100:.2f}% | "
            f"Implied Yield: {div_yield*100:.2f}% | Spot: ${spot:.2f}"
        )
        if T > 90 / 252.0 and div_yield < 0.001:
            st.warning("**Check Dividend Data** — T > 90 days and Yield = 0%. Verify dividend data.")
        if biz_days_to_exp < 14:
            hv_window = 10
        elif biz_days_to_exp < 45:
            hv_window = 20
        elif biz_days_to_exp < 90:
            hv_window = 60
        elif biz_days_to_exp < 180:
            hv_window = 120
        else:
            hv_window = 252

        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
        opt_type = option_type.lower()
        chain = (calls if opt_type == "call" else puts).copy()
        raw_chain = chain.copy()
        raw_count = len(raw_chain)
        # Volume et OI sur TOUTE l'échéance (calls+puts), pas seulement les strikes affichés
        total_volume_exp = int(calls["volume"].fillna(0).sum() + puts["volume"].fillna(0).sum())
        total_oi_exp = int(calls["openInterest"].fillna(0).sum() + puts["openInterest"].fillna(0).sum())
        quote_ts = None
        if "lastTradeDate" in raw_chain.columns and len(raw_chain) > 0:
            try:
                quote_ts = str(pd.to_datetime(raw_chain["lastTradeDate"]).max())
            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Quote timestamp unavailable: %s", e)
                quote_ts = None
        chain_after_clean = DataCleaner.clean_option_chain(chain, min_bid=0.01)
        count_after_clean = len(chain_after_clean)
        chain = DataCleaner.filter_by_moneyness(chain_after_clean, spot, MONEYNESS_MIN, MONEYNESS_MAX)
        filtered_count = len(chain)

        if len(chain) == 0:
            st.warning("No liquid options found")
            return

        svi_params = None
        svi_fit_rmse_pct = None
        use_svi_for_pricing = True  # Auto: SVI si fit OK (RMSE <= 5%), sinon IV brute
        svi_mask = chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0.01) & (chain["impliedVolatility"] < 3.0)
        if svi_mask.sum() >= 5:
            try:
                svi_model = VolatilitySurface([], [], [])
                strikes_svi = chain.loc[svi_mask, "strike"].values.astype(float)
                ivs_svi = chain.loc[svi_mask, "impliedVolatility"].values.astype(float)
                svi_params = svi_model.calibrate_svi(strikes_svi, ivs_svi, spot)
                if svi_params is not None:
                    iv_fit = np.array([svi_model.get_iv_from_svi(k, spot, svi_params) for k in strikes_svi], dtype=float)
                    svi_fit_rmse_pct = float(np.sqrt(np.mean((iv_fit - ivs_svi) ** 2)) * 100)
            except (ValueError, RuntimeError) as e:
                logger.warning("SVI calibration failed: %s", e)
                svi_params = None

        results = []
        for _, row in chain.iterrows():
            K = row["strike"]
            bid, ask = row["bid"], row["ask"]
            # Mid = (Bid+Ask)/2 (priorité). Fallback Last si bid/ask nuls (rare après clean)
            mid = (bid + ask) / 2 if (bid + ask) > 0 else row.get("lastPrice", 0)
            if mid <= 0:
                continue

            market_iv = row.get("impliedVolatility", None)
            if market_iv is None or market_iv <= 0:
                market_iv = IVsolver.find_implied_vol(mid, spot, K, T, rate, opt_type, div_yield)
            if market_iv is None or market_iv <= 0:
                continue

            svi_iv = None
            if svi_params is not None:
                try:
                    svi_iv = float(svi_model.get_iv_from_svi(K, spot, svi_params))
                    svi_iv = float(np.clip(svi_iv, 0.01, 3.0))
                except (ValueError, TypeError, KeyError):
                    svi_iv = None

            # Ne pas utiliser SVI si le fit est mauvais (RMSE > 5% = bruit excessif)
            svi_usable = (
                use_svi_for_pricing
                and svi_iv is not None
                and (svi_fit_rmse_pct is None or svi_fit_rmse_pct <= 5.0)
            )
            iv_used = svi_iv if svi_usable else market_iv

            theoretical_hv = BlackScholes.get_price(spot, K, T, rate, hist_vol, opt_type, div_yield)
            greeks = BlackScholes.get_all_greeks(spot, K, T, rate, iv_used, opt_type, div_yield)

            exec_cost = (ask - bid) / 2

            spread_dollar = ask - bid
            results.append({
                "Strike": K, "Moneyness": K / spot,
                "Bid": bid, "Ask": ask, "Mid": mid,
                "Spread_%": ((ask - bid) / mid * 100) if mid > 0 else 0,
                "Spread_$": spread_dollar,
                "IV_%": market_iv * 100, "SVI_IV_%": (svi_iv * 100) if svi_iv is not None else np.nan,
                "IV_Used_%": iv_used * 100, "HV_%": hist_vol * 100,
                "BS_Price": greeks["price"], "Theoretical_HV": theoretical_hv,
                "Mispricing": greeks["price"] - mid,
                "Cost_Ask": ask * 100,
                "Exec_Cost": exec_cost * 100,
                "Delta": greeks["delta"], "Gamma": greeks["gamma"],
                "Vega": greeks["vega"], "Theta": greeks["theta"],
                "Rho": greeks["rho"],
                "Vanna": greeks["vanna"], "Volga": greeks["volga"],
                "Charm": greeks["charm"],
                "Volume": row.get("volume", 0), "OpenInt": row.get("openInterest", 0)
            })

        if not results:
            st.warning("Could not compute IVs")
            return
        df = pd.DataFrame(results)

        # Correction convexité (butterfly) : si violations, remplacer IV par SVI pour ces strikes
        if opt_type == "call" and svi_params is not None and svi_usable:
            strikes_arr = df["Strike"].values
            iv_arr = df["IV_Used_%"].values / 100.0
            prices = np.array([
                BlackScholes.get_price(spot, K, T, rate, iv, "call", div_yield)
                for K, iv in zip(strikes_arr, iv_arr)
            ])
            sec_diff = np.diff(prices, 2)
            viol_idx = np.where(sec_diff < -1e-6)[0] + 1
            for idx in viol_idx:
                try:
                    svi_iv_fix = float(svi_model.get_iv_from_svi(strikes_arr[idx], spot, svi_params))
                    svi_iv_fix = float(np.clip(svi_iv_fix, 0.01, 3.0))
                    df.loc[df.index[idx], "IV_Used_%"] = svi_iv_fix * 100
                    g = BlackScholes.get_all_greeks(spot, strikes_arr[idx], T, rate, svi_iv_fix, opt_type, div_yield)
                    df.loc[df.index[idx], "BS_Price"] = g["price"]
                    df.loc[df.index[idx], "Mispricing"] = g["price"] - df.loc[df.index[idx], "Mid"]
                    for gk in ["Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga", "Charm"]:
                        df.loc[df.index[idx], gk] = g[gk.lower()]
                except (ValueError, TypeError, KeyError):
                    pass

        summary_box = st.expander("Market Summary", expanded=True)
        atm_idx = (df["Moneyness"] - 1.0).abs().idxmin()
        atm_iv = df.loc[atm_idx, "IV_%"]
        atm_strike = df.loc[atm_idx, "Strike"]
        render_market_summary(
            summary_box, df, spot, atm_idx, days_to_exp, maturity_mode,
            rate, atm_iv, atm_strike, hv_window, hist_vol, biz_days_to_exp,
            total_volume_exp=total_volume_exp, total_oi_exp=total_oi_exp
        )

        dq_box = st.expander("Data Quality", expanded=False)
        render_data_quality(
            dq_box, df, raw_count, filtered_count, quote_ts, opt_type,
            svi_fit_rmse_pct, day_count_basis,
            count_after_clean=count_after_clean,
        )

        chain_box = st.expander(f"{option_type} Chain - {ticker} {exp_date}", expanded=False)
        render_chain_display(chain_box, df, option_type, ticker, exp_date)

        analysis_box = st.expander("Analysis", expanded=True)
        tab_vol, tab_greeks, tab_pnl, tab_attrib, tab_heston, tab_mc, tab_pricing, tab_risk = analysis_box.tabs([
            "Volatility", "Greeks", "P&L", "P&L Attribution",
            "BSM vs Heston", "MC vs BSM", "EU vs US Pricing", "Risk Analysis"
        ])

        render_volatility_tab(tab_vol, df, spot, hist_vol)
        render_greeks_tabs(tab_greeks, df, spot, opt_type)
        render_pnl_tab(tab_pnl, df, spot, opt_type, atm_idx)
        render_attribution_tab(tab_attrib, df, spot, opt_type, atm_idx, rate, T, div_yield, days_to_exp)

        render_heston_tab(tab_heston, df, spot, exp_date, opt_type, T, rate, div_yield, ticker, atm_iv)

        render_mc_tab(tab_mc, df, spot, opt_type, atm_idx, rate, T, div_yield, hist_vol, ticker, exp_date)

        render_pricing_tab(tab_pricing, df, spot, T, rate, div_yield, opt_type, ticker=ticker, exp_date=exp_date, connector=connector)

        render_risk_tab(tab_risk, df, spot, opt_type, atm_idx, T, rate, div_yield, calls, puts, spot_market=spot_market)

        surface_box = st.expander("3D Surfaces (Strike x Maturity)", expanded=False)
        render_surfaces_section(surface_box, exp_options, ticker, spot, opt_type, rate, div_yield, hist_vol, connector=connector)

    except ValueError as e:
        if "No options data" in str(e):
            st.error(
                "**Options data unavailable.** Yahoo Finance often blocks requests "
                "from cloud servers (Streamlit Cloud). The app works correctly locally: "
                "run `streamlit run app.py` on your machine."
            )
        else:
            st.error(f"Error: {str(e)}")
        logger.exception("Vanilla pricer error")
    except (KeyError, TypeError, ImportError) as e:
        logger.exception("Vanilla pricer error")
        st.error(f"Data error: {str(e)}")
    except Exception as e:
        from data.exceptions import DataError, NetworkError
        if isinstance(e, (DataError, NetworkError)):
            st.error(f"Market/data error: {str(e)}")
        else:
            st.error(f"Error: {str(e)}")
            import traceback
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
        logger.exception("Vanilla pricer error")
