import streamlit as st
from ui.styles import inject_global_css, set_page_config
from data.loaders import load_config, load_data
from ui.components import sidebar, header
from pages.tab_top import render_tab as render_tab_top
from pages.tab_detail import render_tab as render_tab_detail
from pages.tab_explore import render_tab as render_tab_explore
from pages.tab_history import render_tab as render_tab_history


def run():
    set_page_config()
    inject_global_css()

    config = load_config()
    df_feats, df_today, df_hist = load_data()

    # Sidebar (retourne filtres sélectionnés)
    sectors_sel, regions_sel = sidebar(df_feats, df_today)

    header()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Opportunités",
        "🔍 Analyse Détaillée",
        "📊 Exploration",
        "📈 Historique",
    ])

    with tab1:
        render_tab_top(df_today, df_feats, sectors_sel, regions_sel)

    with tab2:
        render_tab_detail(df_today, config, sectors_sel, regions_sel)

    with tab3:
        render_tab_explore(df_feats, sectors_sel, regions_sel)

    with tab4:
        render_tab_history(df_hist, df_today, sectors_sel, regions_sel)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <h3 style="color: #667eea; margin: 0;">💡 AI Invest Assistant v2.0</h3>
        <p style="margin: 1rem 0 0 0; color: #666;">
        Analyse quantitative multi-facteurs pour l'investissement intelligent<br>
        <code>compute_scores.py --profile [growth|value|blend|turnaround]</code>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
