# App/pages/tab_history.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data.loaders import apply_filters


def render_tab(df_hist, df_today, sectors_sel, regions_sel):
    """Affiche l'historique des scores et leur évolution"""
    
    if df_hist.empty:
        st.info("📊 Aucun historique disponible. Les données s'accumuleront au fil du temps.")
        return
    
    st.markdown("### 📈 Évolution Historique des Scores")
    
    # Appliquer les filtres
    filt = apply_filters(df_hist, sectors_sel, regions_sel)
    
    if filt.empty:
        st.warning("Aucune donnée après application des filtres")
        return
    
    # Vérifier que la colonne 'date' existe
    if "date" not in filt.columns:
        st.error("La colonne 'date' est manquante dans l'historique")
        return
    
    # Convertir les dates
    filt["date"] = pd.to_datetime(filt["date"], format="%d-%m-%y", errors="coerce")
    filt = filt.dropna(subset=["date"])
    
    if filt.empty:
        st.warning("Aucune date valide dans l'historique")
        return
    
    # Sélection du ticker
    st.markdown("### 🔍 Sélectionner un Titre")
    tickers_available = sorted(filt["ticker"].unique().tolist())
    
    if not tickers_available:
        st.warning("Aucun ticker disponible")
        return
    
    ticker_sel = st.selectbox(
        "Choisir un titre pour voir son évolution :",
        tickers_available,
        help="Sélectionnez un ticker pour visualiser l'évolution de son score dans le temps"
    )
    
    # Filtrer pour le ticker sélectionné
    ticker_data = filt[filt["ticker"] == ticker_sel].sort_values("date")
    
    if ticker_data.empty:
        st.warning(f"Aucune donnée historique pour {ticker_sel}")
        return
    
    # Afficher les informations du ticker
    latest = ticker_data.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Ticker", ticker_sel)
    with c2:
        if "name" in latest:
            st.metric("Nom", latest["name"])
    with c3:
        st.metric("Score Actuel", f"{latest['score']:.1f}")
    with c4:
        if len(ticker_data) >= 2:
            score_change = latest["score"] - ticker_data.iloc[-2]["score"]
            st.metric("Évolution", f"{score_change:+.1f}", delta=f"{score_change:+.1f}")
    
    st.markdown("---")
    
    # Graphique d'évolution du score
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ticker_data["date"],
        y=ticker_data["score"],
        mode="lines+markers",
        name="Score",
        line=dict(color="#667eea", width=3),
        marker=dict(size=8, color="#667eea", line=dict(width=2, color="white"))
    ))
    
    # Lignes de référence
    fig.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.5, 
                  annotation_text="Excellente opportunité")
    fig.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.5,
                  annotation_text="Opportunité moyenne")
    
    fig.update_layout(
        title=f"Évolution du Score de {ticker_sel}",
        xaxis_title="Date",
        yaxis_title="Score",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques sur la période
    st.markdown("### 📊 Statistiques sur la Période")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Score Minimum", f"{ticker_data['score'].min():.1f}")
    with c2:
        st.metric("Score Maximum", f"{ticker_data['score'].max():.1f}")
    with c3:
        st.metric("Score Moyen", f"{ticker_data['score'].mean():.1f}")
    with c4:
        st.metric("Nombre de Points", len(ticker_data))
    
    # Tableau détaillé
    with st.expander("📋 Voir le tableau détaillé", expanded=False):
        display_cols = ["date", "score", "score_adj", "sector", "region"]
        display_cols = [c for c in display_cols if c in ticker_data.columns]
        st.dataframe(
            ticker_data[display_cols].sort_values("date", ascending=False),
            use_container_width=True,
            height=400
        )