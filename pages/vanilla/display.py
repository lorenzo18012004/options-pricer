"""
Affichage des métriques et tableaux pour le pricer vanilla.
"""

import numpy as np
import streamlit as st


def render_market_summary(summary_box, df, spot, atm_idx, days_to_exp, maturity_mode,
                          rate, atm_iv, atm_strike, hv_window, hist_vol, biz_days_to_exp,
                          total_volume_exp=None, total_oi_exp=None):
    """Render Market Summary expander."""
    with summary_box:
        c1, c2, c3, c4, c5, c6 = summary_box.columns(6)
        c1.metric("Spot", f"${spot:.2f}")
        c2.metric("ATM Strike", f"${atm_strike:.0f}")
        c3.metric("Days", f"{days_to_exp}", help=f"{maturity_mode} to expiry")
        c4.metric("Rate", f"{rate*100:.2f}%")
        c5.metric("ATM IV", f"{atm_iv:.1f}%")
        c6.metric(f"Hist Vol ({hv_window}d)", f"{hist_vol*100:.1f}%")
        iv_hv_spread = atm_iv - hist_vol * 100
        c7, c8, c9, c10 = summary_box.columns(4)
        c7.metric("IV - HV Spread", f"{iv_hv_spread:+.1f}%")
        atm_mask = (df["Moneyness"] >= 0.95) & (df["Moneyness"] <= 1.05)
        atm_spread_pct = df.loc[atm_mask, "Spread_%"].median() if atm_mask.any() else df["Spread_%"].median()
        atm_spread_dollar = df.loc[atm_mask, "Spread_$"].median() if atm_mask.any() and "Spread_$" in df.columns else (df["Ask"] - df["Bid"]).median() if "Ask" in df.columns and "Bid" in df.columns else 0
        c8.metric("ATM Spread", f"{atm_spread_pct:.1f}% (${atm_spread_dollar:.2f})")
        vol_val = total_volume_exp if total_volume_exp is not None else df["Volume"].sum()
        oi_val = total_oi_exp if total_oi_exp is not None else df["OpenInt"].sum()
        c9.metric("Total Volume", f"{vol_val:,.0f}")
        c10.metric("Total OI", f"{oi_val:,.0f}")


def render_data_quality(dq_box, df, raw_count, filtered_count, quote_ts, opt_type,
                        svi_fit_rmse_pct, day_count_basis, count_after_clean=None):
    """Render Data Quality expander."""
    with dq_box:
        retention_pct = (filtered_count / raw_count * 100) if raw_count > 0 else 0
        dq1, dq2, dq3, dq4 = dq_box.columns(4)
        dq1.metric("Raw strikes", f"{raw_count}")
        dq2.metric("Filtered strikes", f"{filtered_count}")
        dq3.metric("Retention", f"{retention_pct:.1f}%")
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
        dq7.metric("Med spread", f"{df['Spread_%'].median():.2f}% (${med_spread_dollar:.2f})")
        dq8, dq9, dq10 = dq_box.columns(3)
        dq8.metric("SVI fit RMSE", f"{svi_fit_rmse_pct:.3f}%" if svi_fit_rmse_pct is not None else "N/A")
        dq9.metric("Day count", day_count_basis)
        dq10.metric("Maturity clock", "Calendar")
        penalty = 0.0
        penalty += min(30.0, mono_breaches * 3.0)
        penalty += min(30.0, conv_breaches * 3.0)
        penalty += min(20.0, max(0.0, df["Spread_%"].median() - 8.0) * 0.8)
        if svi_fit_rmse_pct is not None:
            penalty += min(20.0, max(0.0, svi_fit_rmse_pct - 3.0) * 2.0)
        quality_score = max(0.0, 100.0 - penalty)
        dq_box.metric("Pricing quality score", f"{quality_score:.1f}/100")


def render_chain_display(chain_box, df, option_type, ticker, exp_date):
    """Render option chain dataframe in expander."""
    with chain_box:
        fmt = {
            "Strike": "${:.2f}", "Moneyness": "{:.3f}",
            "Bid": "${:.2f}", "Ask": "${:.2f}", "Mid": "${:.2f}",
            "Spread_%": "{:.1f}%", "Spread_$": "${:.2f}",
            "IV_%": "{:.1f}%", "SVI_IV_%": "{:.1f}%", "IV_Used_%": "{:.1f}%", "HV_%": "{:.1f}%",
            "BS_Price": "${:.2f}", "Theoretical_HV": "${:.2f}", "Mispricing": "${:+.2f}",
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
