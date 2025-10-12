# App/pages/tab_top.py
# -----------------------------------------------------------
# Tableau de bord principal : Top opportunités + pondérations interactives
# -----------------------------------------------------------

import streamlit as st
import plotly.graph_objects as go
from data.loaders import apply_filters, dataframe_to_percent, load_config
from ui.components import show_column_legend_clear


def render_tab(df_today, df_feats, sectors_sel, regions_sel):
    """Onglet principal : top 10, tableau complet, sliders de pondération."""

    # -------------------- Vérification données --------------------
    if df_today.empty:
        st.warning("⚠️ Aucune donnée disponible. Lancez `compute_features_auto.py` puis `compute_scores.py`")
        return

    # Appliquer les filtres
    top = apply_filters(df_today, sectors_sel, regions_sel)
    if top.empty:
        st.warning("Aucun résultat avec les filtres actuels")
        return

    # -------------------- Chargement config & sliders --------------------
    cfg = load_config()
    profiles = cfg.get("profiles", {})
    profile_names = list(profiles.keys()) or ["growth"]

    with st.sidebar:
        st.markdown("### ⚙️ Pondérations")

        base_profile = st.selectbox(
            "Profil de base",
            profile_names,
            index=(profile_names.index("growth") if "growth" in profile_names else 0)
        )
        base_weights = profiles.get(base_profile, {}).get("weights", {})

        # Gestion toggle ou checkbox selon version Streamlit
        try:
            use_custom = st.toggle(
                "Activer pondérations personnalisées",
                value=False,
                help="Permet d’ajuster les poids des catégories ci-dessous."
            )
        except Exception:
            use_custom = st.checkbox(
                "Activer pondérations personnalisées",
                value=False,
                help="Permet d’ajuster les poids des catégories ci-dessous."
            )

        # Sliders
        w_mom = st.slider("Momentum", 0.0, 1.0, float(base_weights.get("momentum", 0.2)), 0.01, disabled=not use_custom)
        w_val = st.slider("Valuation", 0.0, 1.0, float(base_weights.get("valuation", 0.2)), 0.01, disabled=not use_custom)
        w_growth = st.slider("Growth", 0.0, 1.0, float(base_weights.get("growth", 0.2)), 0.01, disabled=not use_custom)
        w_quality = st.slider("Quality", 0.0, 1.0, float(base_weights.get("quality", 0.2)), 0.01, disabled=not use_custom)
        w_risk = st.slider("Risk", 0.0, 1.0, float(base_weights.get("risk", 0.2)), 0.01, disabled=not use_custom)

        if use_custom:
            tot = max(1e-9, w_mom + w_val + w_growth + w_quality + w_risk)
            weights = {
                "momentum": w_mom / tot,
                "valuation": w_val / tot,
                "growth": w_growth / tot,
                "quality": w_quality / tot,
                "risk": w_risk / tot,
            }
        else:
            weights = {
                "momentum": float(base_weights.get("momentum", 0.2)),
                "valuation": float(base_weights.get("valuation", 0.2)),
                "growth": float(base_weights.get("growth", 0.2)),
                "quality": float(base_weights.get("quality", 0.2)),
                "risk": float(base_weights.get("risk", 0.2)),
            }

    # -------------------- Calcul du score custom --------------------
    z_cols = ["z_momentum", "z_valuation", "z_growth", "z_quality", "z_risk"]
    missing = [c for c in z_cols if c not in top.columns]

    if missing:
        st.warning(f"Certaines colonnes de z-score manquent : {missing}. Lance d'abord compute_scores.py.")
        current_score_col = "score" if "score" in top.columns else None
    else:
        if use_custom:
            zdf = top[z_cols].fillna(0.0).copy()
            top["score_custom"] = (
                float(weights["momentum"]) * zdf["z_momentum"] +
                float(weights["valuation"]) * zdf["z_valuation"] +
                float(weights["growth"]) * zdf["z_growth"] +
                float(weights["quality"]) * zdf["z_quality"] +
                float(weights["risk"]) * zdf["z_risk"]
            )
            current_score_col = "score_custom"
        else:
            current_score_col = "score" if "score" in top.columns else None

    if current_score_col is None:
        st.warning("Aucun score disponible à afficher.")
        return

    # -------------------- Poids actifs --------------------
    st.caption(
        "**Pondérations actives** : " +
        " | ".join([f"{k}:{float(weights.get(k,0)):.2f}" for k in ["momentum","valuation","growth","quality","risk"]])
    )

    # -------------------- Indicateurs clés --------------------
    st.markdown("###  Indicateurs Clés")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Titres Analysés</div>"
            f"<div class='metric-value'>{len(top)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Meilleur Score</div>"
            f"<div class='metric-value'>{top[current_score_col].max():.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Score Médian</div>"
            f"<div class='metric-value'>{top[current_score_col].median():.1f}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Score Moyen</div>"
            f"<div class='metric-value'>{top[current_score_col].mean():.1f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- Top 10 --------------------
    st.markdown("###  Top 10 des Meilleures Opportunités")

    top10 = top.nlargest(10, current_score_col)
    colors = ['#43e97b' if x > 70 else '#4facfe' if x > 50 else '#ffa502' for x in top10[current_score_col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top10['ticker'],
        x=top10[current_score_col],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=top10[current_score_col].round(1),
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
        st.markdown('<span class="performance-indicator perf-positive"> > 70: Excellente opportunité</span>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="performance-indicator perf-neutral"> 50-70: Opportunité moyenne</span>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<span class="performance-indicator perf-negative"> < 50: À surveiller</span>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- Légende + Tableau --------------------
    with st.expander(" Comprendre les indicateurs du tableau", expanded=False):
        show_column_legend_clear(top)

    st.markdown("###  Tableau Complet des Résultats")

    all_cols = top.columns.tolist()
    default_cols = [c for c in ['ticker', 'name', current_score_col, 'sector', 'region'] if c in all_cols]

    display_cols = st.multiselect("Colonnes à afficher :", all_cols, default=default_cols)

    if display_cols:
        top_display = dataframe_to_percent(top[display_cols], digits=2)
        if current_score_col in display_cols:
            top_display = top_display.sort_values(current_score_col, ascending=False)
        st.dataframe(top_display, use_container_width=True, height=500)
    else:
        st.info("Sélectionnez au moins une colonne à afficher")
