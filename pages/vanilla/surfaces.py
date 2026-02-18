"""
Surfaces 3D (Strike x Maturity) pour le pricer vanilla.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import BlackScholes
from data import DataCleaner
from models.surfaces import VolatilitySurface

from config.options import (
    MONEYNESS_SURFACE_MIN,
    MONEYNESS_SURFACE_MAX,
    IV_MIN_PCT,
    IV_MAX_PCT,
    MIN_VOLUME,
    MIN_MID_PRICE,
    SURFACE_SMOOTH_RBF,
    SURFACE_GAUSSIAN_SIGMA,
    IV_CAP_DISPLAY_PCT,
)

logger = logging.getLogger(__name__)


def render_surfaces_section(surface_box, exp_options, ticker, spot, opt_type, rate, div_yield, hist_vol, connector=None):
    """Render 3D Surfaces (Strike x Maturity) section."""
    from data import DataConnector
    if connector is None:
        connector = DataConnector
    with st.spinner("Building 3D surfaces from multiple expirations..."):
        surface_data = []
        exps_to_use = [e for e in exp_options[:15] if e["days"] > 0]
        for exp_info in exps_to_use:
            exp_d = exp_info["date"]
            biz_d = exp_info.get("biz_days", exp_info["days"])
            T_exp = biz_d / 252.0
            try:
                c_chain, p_chain = connector.get_option_chain(ticker, exp_d)
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
    extrap_strikes = np.round(np.linspace(spot * MONEYNESS_SURFACE_MIN, spot * MONEYNESS_SURFACE_MAX, 25), 2)
    extrap_rows = []
    for days_val in df_surf["Days"].unique():
        grp = df_surf[df_surf["Days"] == days_val].sort_values("Strike")
        if len(grp) < 5:
            continue
        T_g = float(grp["T"].iloc[0])
        k_min, k_max = grp["Strike"].min(), grp["Strike"].max()
        try:
            svi_model = VolatilitySurface([], [], [])
            svi_p = svi_model.calibrate_svi(
                grp["Strike"].values, grp["IV"].values / 100.0, spot
            )
            if svi_p is not None:
                for K in extrap_strikes:
                    if K < k_min or K > k_max:
                        iv_ext = svi_model.get_iv_from_svi(K, spot, svi_p) * 100
                        g_ext = BlackScholes.get_all_greeks(spot, K, T_g, rate, iv_ext / 100.0, opt_type, div_yield)
                        extrap_rows.append({
                            "Strike": K, "T": T_g, "Days": days_val,
                            "IV": iv_ext,
                            "Delta": g_ext["delta"], "Gamma": g_ext["gamma"],
                            "Vega": g_ext["vega"], "Theta": g_ext["theta"],
                            "Vanna": g_ext["vanna"], "Volga": g_ext["volga"],
                            "Charm": g_ext["charm"],
                        })
        except (ValueError, RuntimeError, TypeError):
            pass
    if extrap_rows:
        df_surf = pd.concat([df_surf, pd.DataFrame(extrap_rows)], ignore_index=True)
        df_surf = df_surf.drop_duplicates(subset=["Strike", "Days"], keep="first")
        df_surf = df_surf.sort_values(["Days", "Strike"]).reset_index(drop=True)
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
        """Interpolation RBF + Gaussian smoothing + plafond IV."""
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
        except (ValueError, ImportError, RuntimeError):
            from scipy.interpolate import griddata
            zi_grid = griddata(
                pts, vals, (xi_grid, yi_grid), method="cubic"
            )
        zi_grid = np.nan_to_num(zi_grid, nan=0.0, posinf=0.0, neginf=0.0)
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
                    z_min, z_max = float(zi.min()), float(zi.max())
                    z_span = max(z_max - z_min, 5.0)
                    z_lo = max(0, z_min - 0.15 * z_span)
                    z_hi = min(IV_CAP_DISPLAY_PCT, z_max + 0.15 * z_span)
                    scene_cfg["zaxis"] = dict(range=[z_lo, z_hi], title=z_label)
                    cmin, cmax = z_lo, z_hi
                else:
                    cmin, cmax = None, None
                fig_3d = go.Figure(data=[go.Surface(
                    x=xi, y=yi, z=zi,
                    colorscale=colorscale,
                    colorbar=dict(title=z_label),
                    cmin=cmin if col_name == "IV" else None,
                    cmax=cmax if col_name == "IV" else None,
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
