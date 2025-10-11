# App/pages/tab_top.py
import streamlit as st
import plotly.graph_objects as go
from data.loaders import apply_filters, dataframe_to_percent
from ui.components import show_column_legend_clear


def render_tab(df_today, df_feats, sectors_sel, regions_sel):
    if df_today.empty:
        st.warning("⚠️ Aucune donnée disponible. Lancez `compute_features_auto.py` puis `compute_scores.py`")
        return
    
    top = apply_filters(df_today, sectors_sel, regions_sel)
    if top.empty:
        st.warning("Aucun résultat avec les filtres actuels")
        return
    
    st.markdown("### 📊 Indicateurs Clés")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Titres Analysés</div><div class='metric-value'>{len(top)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Meilleur Score</div><div class='metric-value'>{top['score'].max():.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Score Médian</div><div class='metric-value'>{top['score'].median():.1f}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Score Moyen</div><div class='metric-value'>{top['score'].mean():.1f}</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏆 Top 10 des Meilleures Opportunités")
    
    top10 = top.nlargest(10, 'score')
    colors = ['#43e97b' if x > 70 else '#4facfe' if x > 50 else '#ffa502' for x in top10['score']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top10['ticker'],
        x=top10['score'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=top10['score'].round(1),
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Score",
        yaxis_title="",
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=80, r=80, t=20, b=40)
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<span class="performance-indicator perf-positive">🟢 > 70: Excellente opportunité</span>', unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="performance-indicator perf-neutral">🟡 50-70: Opportunité moyenne</span>', unsafe_allow_html=True)
    with c3:
        st.markdown('<span class="performance-indicator perf-negative">🔴 < 50: À surveiller</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📖 Comprendre les indicateurs du tableau", expanded=False):
        show_column_legend_clear(top)
    
    st.markdown("### 📋 Tableau Complet des Résultats")
    all_cols = top.columns.tolist()
    default_cols = ['ticker', 'name', 'score', 'sector', 'region'] if all(c in all_cols for c in ['ticker', 'name', 'score', 'sector', 'region']) else all_cols[:5]
    display_cols = st.multiselect("Colonnes à afficher:", all_cols, default=default_cols)
    
    if display_cols:
        top_display = dataframe_to_percent(top[display_cols], digits=2)
        st.dataframe(
            top_display.sort_values("score", ascending=False) if "score" in display_cols else top_display,
            use_container_width=True,
            height=500
        )
    else:
        st.info("Sélectionnez au moins une colonne à afficher")