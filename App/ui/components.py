# App/ui/components.py
import streamlit as st


def sidebar(df_feats, df_today):
    """Barre latérale avec explications et filtres"""
    with st.sidebar:
        st.markdown("### 📖 Guide des Indicateurs")
        st.markdown("- **Momentum** : tendance du cours sur 6 à 12 mois")
        st.markdown("- **Volatilité** : amplitude moyenne des variations du titre")
        st.markdown("- **Value** : niveau de valorisation (PER, EV/EBITDA, etc.)")
        st.markdown("- **Quality** : rentabilité, marges, dette, ROIC, etc.")
        st.markdown("- **Growth** : croissance du chiffre d'affaires et du bénéfice")
        st.markdown("- **Turnaround** : amélioration récente des fondamentaux")
        st.markdown("---")
        
        # Filtres
        st.markdown("### 🔍 Filtres")
        
        sectors_sel = []
        regions_sel = []
        
        # Filtre secteur
        if "sector" in df_today.columns:
            all_sectors = sorted(df_today["sector"].dropna().unique().tolist())
            if all_sectors:
                sectors_sel = st.multiselect(
                    "Secteurs",
                    options=all_sectors,
                    default=all_sectors,
                    help="Sélectionnez les secteurs à analyser"
                )
        
        # Filtre région
        if "region" in df_today.columns:
            all_regions = sorted(df_today["region"].dropna().unique().tolist())
            if all_regions:
                regions_sel = st.multiselect(
                    "Régions",
                    options=all_regions,
                    default=all_regions,
                    help="Sélectionnez les régions à analyser"
                )
        
        st.markdown("---")
        st.markdown("💡 *Les scores combinent ces indicateurs pour classer les actions selon ton profil d'investisseur.*")
    
    return sectors_sel, regions_sel


def header():
    """En-tête principal du dashboard"""
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 16px; margin-bottom: 2rem; color: white; box-shadow: 0 4px 12px rgba(102,126,234,0.3);">
            <h1 style="margin: 0; font-size: 2.5rem;">📊 AI Invest Assistant</h1>
            <p style="margin: 1rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Analyse quantitative multi-facteurs pour l'investissement intelligent
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_column_legend_clear(df):
    """Affiche la légende des colonnes disponibles dans le DataFrame"""
    legends = {
        "mom_3m": "📈 Momentum 3 mois - Performance sur 3 mois (%)",
        "mom_6m": "📈 Momentum 6 mois - Performance sur 6 mois (%)",
        "beta_spx": "📊 Beta S&P500 - Sensibilité au marché",
        "volatility_30d": "📉 Volatilité - Risque annualisé",
        "pe_ttm": "💰 PER - Price/Earnings ratio",
        "fcf_yield": "💵 FCF Yield - Free Cash Flow / Market Cap (%)",
        "pb_ratio": "📊 P/B - Price to Book ratio",
        "revenue_yoy": "📈 Croissance CA - Revenus année sur année (%)",
        "eps_yoy_1y": "💹 Croissance EPS - Bénéfice par action YoY (%)",
        "gross_margin_ttm": "💼 Marge Brute - Rentabilité brute (%)",
        "oper_margin": "💼 Marge Opérationnelle (%)",
        "roe": "💰 ROE - Return on Equity (%)",
        "roa": "💰 ROA - Return on Assets (%)",
        "roic_ttm": "💰 ROIC - Return on Invested Capital (%)",
        "net_debt_ebitda": "⚠️ Dette Nette/EBITDA - Levier financier",
        "debt_equity": "⚠️ Dette/Equity - Endettement",
        "score": "🎯 Score Global - Score composite",
        "z_momentum": "📊 Z-Score Momentum",
        "z_valuation": "📊 Z-Score Valorisation",
        "z_growth": "📊 Z-Score Croissance",
        "z_quality": "📊 Z-Score Qualité",
        "z_risk": "📊 Z-Score Risque",
    }
    
    available_cols = [col for col in legends.keys() if col in df.columns]
    
    if available_cols:
        st.markdown("#### 📚 Légende des Indicateurs")
        for col in available_cols:
            st.markdown(f"**{col}** : {legends[col]}")
    else:
        st.info("Aucun indicateur disponible pour afficher la légende")