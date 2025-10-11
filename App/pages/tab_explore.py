# App/pages/tab_explore.py
import streamlit as st
import numpy as np
import plotly.express as px
from data.loaders import apply_filters
from ui.components import show_column_legend_clear


def render_tab(df_feats, sectors_sel, regions_sel):
    if df_feats.empty:
        st.info("features_latest.csv introuvable. Exécutez compute_features_auto.py")
        return
    
    st.markdown("### 🧭 Exploration des Relations entre Indicateurs")
    
    base_cols = ["ticker", "name", "sector", "region", "style", "asof_date"]
    feature_cols = [c for c in df_feats.columns if c not in base_cols]
    
    if len(feature_cols) < 2:
        st.warning("Pas assez d'indicateurs pour créer un graphique de comparaison")
        return
    
    with st.expander("📖 Comprendre les indicateurs", expanded=False):
        show_column_legend_clear(df_feats)
    
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox("📊 Indicateur X (horizontal)", feature_cols, index=0)
    with c2:
        y = st.selectbox("📊 Indicateur Y (vertical)", feature_cols, index=min(1, len(feature_cols) - 1))
    with c3:
        color_by = st.selectbox(
            "🎨 Colorer par",
            ["Aucun"] + [c for c in ["sector", "region", "style"] if c in df_feats.columns]
        )
    
    filt = apply_filters(df_feats, sectors_sel, regions_sel)
    if filt.empty:
        st.warning("Aucune donnée après application des filtres")
        return
    
    plot_cols = [x, y]
    if color_by != "Aucun" and color_by in filt.columns:
        plot_cols.append(color_by)
    for c in ("ticker", "name"):
        if c in filt.columns:
            plot_cols.append(c)
    
    plot_df = filt[plot_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    
    if plot_df.empty:
        st.warning("Données insuffisantes après nettoyage (NaN/Inf)")
        return
    
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=(color_by if color_by != "Aucun" else None),
        hover_data=[c for c in ["ticker", "name"] if c in plot_df.columns],
        title=f"Relation entre {x} et {y}",
        labels={x: x, y: y}
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(gridcolor='lightgray', showgrid=True),
        yaxis=dict(gridcolor='lightgray', showgrid=True)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if len(plot_df) > 2:
        corr = plot_df[x].corr(plot_df[y])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Nombre de points", len(plot_df))
        with c2:
            st.metric("Corrélation", f"{corr:.3f}")
        with c3:
            strength = "Forte" if abs(corr) > 0.7 else "Moyenne" if abs(corr) > 0.4 else "Faible"
            direction = "positive" if corr > 0 else "négative"
            st.info(f"Corrélation {strength} {direction}")