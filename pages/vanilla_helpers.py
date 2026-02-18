"""
Façade pour le pricer vanilla - réexporte depuis pages.vanilla.
Maintient la compatibilité avec vanilla_page et autres imports.
"""

from services.vanilla_service import calibrate_market_from_put_call_parity

from .vanilla import (
    render_market_summary,
    render_data_quality,
    render_chain_display,
    render_volatility_tab,
    render_greeks_tabs,
    render_pnl_tab,
    render_attribution_tab,
    render_heston_tab,
    render_mc_tab,
    render_pricing_tab,
    render_risk_tab,
    render_surfaces_section,
)

__all__ = [
    "calibrate_market_from_put_call_parity",
    "render_market_summary",
    "render_data_quality",
    "render_chain_display",
    "render_volatility_tab",
    "render_greeks_tabs",
    "render_pnl_tab",
    "render_attribution_tab",
    "render_heston_tab",
    "render_mc_tab",
    "render_pricing_tab",
    "render_risk_tab",
    "render_surfaces_section",
]
