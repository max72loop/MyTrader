# app/ui/components.py
st.markdown(
f"<span class='category-badge {CATEGORY_INFO[cat]['badge_class']}'>{CATEGORY_INFO[cat]['nom']}</span>",
unsafe_allow_html=True,
)


def show_column_legend_clear(df: pd.DataFrame):
st.markdown("### 📖 Guide des Indicateurs")
categories: dict[str, list[str]] = {}
for col in df.columns:
info = COLUMN_DESCRIPTIONS.get(col, {"categorie": "autre"})
categories.setdefault(info.get("categorie","autre"), []).append(col)
for cat_key in ["score","momentum","valuation","growth","quality","risk","identité","évolution","autre"]:
if cat_key in categories:
if cat_key in CATEGORY_INFO:
cat_info = CATEGORY_INFO[cat_key]
st.markdown(f"#### {cat_info['nom']}")
st.caption(cat_info['desc'])
else:
st.markdown(f"#### {cat_key.capitalize()}")
cols = st.columns(min(5, len(categories[cat_key])))
for i, col_name in enumerate(categories[cat_key]):
with cols[i % 5]:
show_info_tooltip(col_name)
st.markdown("---")


def sidebar(df_feats: pd.DataFrame, df_today: pd.DataFrame):
with st.sidebar:
st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=AI+Invest", use_container_width=True)
st.markdown("# 📊 Tableau de Bord")
st.markdown("---")
if not df_today.empty:
st.markdown("### 📈 Vue d'ensemble")
c1, c2 = st.columns(2)
with c1: st.metric("Titres", len(df_today))
with c2: st.metric("Score Moy", f"{df_today['score'].mean():.1f}")
if 'profile' in df_today.columns:
st.success(f"**Profil:** {df_today['profile'].iloc[0].upper()}")
st.markdown("---")
st.markdown("### 🔍 Filtres")
sectors = sorted(df_feats["sector"].dropna().unique().tolist()) if not df_feats.empty and "sector" in df_feats.columns else []
sectors_sel = st.multiselect("📁 Secteurs", sectors, default=sectors[:3] if len(sectors)>=3 else sectors, key="sectors")
regions = sorted(df_feats["region"].dropna().unique().tolist()) if not df_feats.empty and "region" in df_feats.columns else []
regions_sel = st.multiselect("🌍 Régions", regions, default=[], key="regions")
st.markdown("---")
if st.button("🔄 Actualiser", use_container_width=True):
from ..data.loaders import load_data
load_data.clear()
st.rerun()
with st.expander("❓ Aide Rapide"):
st.markdown("1. Filtrez par secteur/région ")
st.markdown("2. Consultez le classement ")
st.markdown("3. Analysez un titre en détail ")
st.markdown("4. Suivez l'évolution")
return sectors_sel, regions_sel


def header():
st.markdown("""
<div class="section-header">
<h2>🎯 AI Invest Assistant - Analyse Quantitative</h2>
</div>
""", unsafe_allow_html=True)
st.info("💡 **Astuce:** Cliquez sur les boutons 💡 pour comprendre chaque indicateur en détail")