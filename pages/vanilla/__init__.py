"""
Sous-modules pour le pricer vanilla.
Découpe de vanilla_helpers pour maintenabilité (< 400 lignes par fichier).
"""

from .display import render_market_summary, render_data_quality, render_chain_display
from .tabs_charts import (
    render_volatility_tab,
    render_greeks_tabs,
    render_pnl_tab,
    render_attribution_tab,
)
from .tabs_heston_mc import render_heston_tab, render_mc_tab
from .tabs_pricing_risk import render_pricing_tab, render_risk_tab
from .surfaces import render_surfaces_section

__all__ = [
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
