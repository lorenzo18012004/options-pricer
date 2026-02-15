"""
PROJET PRICER - App entrypoint (clean router).
"""

import logging
from pathlib import Path

import streamlit as st

from pages import (
    render_vanilla_option_pricer,
    render_volatility_strategies,
    render_barrier_pricer,
    render_swap_pricer,
)

# Basic runtime logging for diagnostics in local execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


st.set_page_config(
    page_title="Options Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    .metric-container {background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .price-display {font-size: 2rem; font-weight: 700; color: #0066cc;}
    .greek-label {font-size: 0.9rem; color: #666; font-weight: 500;}
    .greek-value {font-size: 1.3rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.dialog("Welcome", width="small")
def welcome_modal():
    st.markdown("**Lorenzo Philippe**")
    st.caption(
        "M1 Portfolio Management, IAE Paris Est. Seeking a 4–6 month internship in market finance "
        "(front/middle/back) from April. Open to international."
    )

    st.markdown("**About this pricer**")
    st.caption(
        "Built from scratch. Data: Yahoo Finance (spot, options, vol, Treasury). "
        "Conventions: ACT/365, calendar days."
    )
    st.caption(
        "Vanilla: BSM, Heston, Monte Carlo, binomial (American). "
        "Volatility: straddle, strangle. Barrier: MC + Brownian bridge. "
        "Swap: bootstrap, NPV, DV01, hedge sizing."
    )

    st.caption("Questions or feedback? Reach out on LinkedIn.")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/lorenzo-philippe-9584a82b1", use_container_width=True)
    with col2:
        cv_path = Path(__file__).parent / "static" / "cv.pdf"
        if cv_path.exists():
            with open(cv_path, "rb") as f:
                st.download_button("CV", data=f.read(), file_name="Lorenzo_PHILIPPE_CV.pdf", mime="application/pdf", use_container_width=True)
    if st.button("Get started", type="primary", use_container_width=True):
        st.session_state.welcome_shown = True
        st.rerun()


def main():
    if "welcome_shown" not in st.session_state:
        welcome_modal()

    st.markdown('<p class="main-header">Options Pricer</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Product Selection")
        product = st.selectbox(
            "Choose Product Type",
            [
                "Vanilla Options",
                "Volatility Strategies",
                "Barrier Options",
                "Interest Rate Swap",
            ],
        )

        st.markdown("---")
        st.markdown("**Lorenzo Philippe**")
        st.markdown(
            "I built this pricer from scratch. It covers vanilla options, volatility strategies, "
            "barrier options, and interest rate swaps, all with live market data."
        )
        st.markdown(
            "I am a student in M1 Portfolio Management at IAE Paris Est. "
            "I am seeking a 4–6 month internship in market finance (front, middle, or back office) from April, "
            "and I am open to international opportunities."
        )
        st.link_button("LinkedIn", "https://www.linkedin.com/in/lorenzo-philippe-9584a82b1", use_container_width=True)
        cv_path = Path(__file__).parent / "static" / "cv.pdf"
        if cv_path.exists():
            with open(cv_path, "rb") as f:
                st.download_button("Download CV", data=f.read(), file_name="Lorenzo_PHILIPPE_CV.pdf", mime="application/pdf", use_container_width=True)

    if product == "Vanilla Options":
        render_vanilla_option_pricer()
    elif product == "Volatility Strategies":
        render_volatility_strategies()
    elif product == "Barrier Options":
        render_barrier_pricer()
    elif product == "Interest Rate Swap":
        render_swap_pricer()


if __name__ == "__main__":
    main()
