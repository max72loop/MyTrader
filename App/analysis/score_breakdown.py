# App/analysis/score_breakdown.py
import streamlit as st
import plotly.graph_objects as go


def display_score_breakdown(row, config):
    """Affiche la décomposition détaillée du score d'un titre"""
    st.markdown("### 🎯 Décomposition du Score")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Score Final</div>
                <div class="metric-value">{row['score']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("#### 📊 Composition par Catégorie")
    
    z_cols = [c for c in row.index if c.startswith('z_')]
    if z_cols and 'profiles' in config:
        profile_name = row.get('profile', 'growth')
        weights = config['profiles'].get(profile_name, {}).get('weights', {})
        
        cols = st.columns(len(z_cols))
        for i, zcol in enumerate(z_cols):
            cat_name = zcol.replace('z_', '')
            z_val = row[zcol] if zcol in row.index and row[zcol] == row[zcol] else 0
            weight = float(weights.get(cat_name, 0))
            contribution = z_val * weight
            
            with cols[i]:
                st.markdown(f"**{cat_name.upper()}**")
                st.progress(min(abs(contribution) / 3, 1.0))
                st.caption(f"Z-score: {z_val:.2f}")
                st.caption(f"Poids: {weight * 100:.0f}%")
                st.caption(f"→ Contribution: {contribution:.2f}")
    
    if 'score_adj' in row.index and row['score_adj'] == row['score_adj'] and row['score_adj'] != 0:
        adj_val = row['score_adj']
        adj_class = "perf-positive" if adj_val > 0 else "perf-negative"
        adj_icon = "✅" if adj_val > 0 else "⚠️"
        st.markdown(
            f"""
            <div class="performance-indicator {adj_class}">
                {adj_icon} Ajustement: {adj_val:+.2f} points
            </div>
            """,
            unsafe_allow_html=True,
        )


def radar_from_row(row):
    """Crée un graphique radar des z-scores pour un titre"""
    z_cols = [c for c in row.index if c.startswith('z_')]
    if not z_cols:
        return None
    
    categories = [c.replace('z_', '').upper() for c in z_cols]
    values = [row[c] if c in row.index and row[c] == row[c] else 0 for c in z_cols]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=row.get('ticker', 'TICKER'),
        fillcolor='rgba(102,126,234,0.3)',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                gridcolor='lightgray'
            )
        ),
        showlegend=False,
        height=500,
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return fig