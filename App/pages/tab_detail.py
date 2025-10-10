# app/pages/tab_detail.py
import streamlit as st
from ..data.loaders import apply_filters
from ..analysis.score_breakdown import display_score_breakdown, radar_from_row


def render_tab(df_today, config, sectors_sel, regions_sel):
if df_today.empty:
st.warning("⚠️ Aucune donnée disponible")
return
filtered = apply_filters(df_today, sectors_sel, regions_sel)
if filtered.empty:
st.warning("Aucun titre avec les filtres actuels")
return
st.markdown("### 🔍 Sélectionner un Titre")
ticker_options = filtered.sort_values('score', ascending=False)['ticker'].tolist()
def _label(t):
try:
n = filtered[filtered['ticker']==t]['name'].iloc[0]
s = filtered[filtered['ticker']==t]['score'].iloc[0]
return f"{t} - {n} (Score: {s:.1f})"
except Exception:
return t
ticker_sel = st.selectbox("Choisir un titre:", ticker_options, format_func=_label)
if not ticker_sel:
return
row = filtered[filtered['ticker']==ticker_sel].iloc[0]
c1,c2,c3 = st.columns([2,1,1])
with c1:
st.markdown(f"## {ticker_sel}")
if 'name' in row: st.markdown(f"**{row['name']}**")
badges = []
if 'sector' in row: badges.append(f"📁 {row['sector']}")
if 'region' in row: badges.append(f"🌍 {row['region']}")
if 'style' in row: badges.append(f"📊 {row['style']}")
st.caption(" | ".join(badges))
with c2:
sv = row['score']
sc = "perf-positive" if sv>70 else "perf-neutral" if sv>50 else "perf-negative"
st.markdown(f"<div class='performance-indicator {sc}' style='font-size:1.2rem;padding:1rem;'><div style='font-size:.8rem;opacity:.8;'>SCORE GLOBAL</div><div style='font-size:2rem;font-weight:bold;'>{sv:.1f}</div></div>", unsafe_allow_html=True)
with c3:
if 'score_adj' in row and row['score_adj'] == row['score_adj']:
adj = row['score_adj']
icon = "✅" if adj>0 else "⚠️" if adj<0 else "➖"
cls = "perf-positive" if adj>0 else "perf-negative" if adj<0 else "perf-neutral"
st.markdown(f"<div class='performance-indicator {cls}' style='font-size:1rem;padding:1rem;'><div style='font-size:.8rem;opacity:.8;'>AJUSTEMENT</div><div style='font-size:1.5rem;font-weight:bold;'>{icon} {adj:+.1f}</div></div>", unsafe_allow_html=True)
st.markdown("---")
display_score_breakdown(row, config)
st.markdown("---")
st.markdown("### 📊 Profil Multi-Dimensionnel")
fig = radar_from_row(row)
if fig is not None:
st.plotly_chart(fig, use_container_width=True)
st.info("💡 **Interprétation:** Plus la zone est étendue, meilleur est le profil global. Les valeurs positives sont favorables.")
else:
st.info("Pas de données z-score disponibles pour le graphique radar")
st.markdown("---")
st.markdown("### 📊 Détails des Indicateurs")
with st.expander("🔍 Voir tous les indicateurs", expanded=True):
# Regroupement simple par préfixes/colonnes
cols = [c for c in row.index if c not in ("ticker","name","sector","region","style","score","score_adj","profile")]
for c in cols:
st.metric(c, row[c] if row[c]==row[c] else "N/A")