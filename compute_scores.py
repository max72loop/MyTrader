"""
compute_scores.py - VERSION AMÉLIORÉE
--------------------------------------
Lit `config.yaml`, prend le dernier fichier de features avec MÉTRIQUES EN %,
calcule les z-scores par catégorie, agrège selon un profil (pondérations),
applique bonus/malus AMÉLIORÉS, puis écrit :

- data/scores_today.csv
- data/scores_history.csv

AMÉLIORATIONS :
- Support de toutes les nouvelles métriques en %
- Bonus/malus basés sur les seuils de %
- Calcul de confiance du score
- Filtrage de qualité

Usage :
    python compute_scores.py --config config.yaml --data_dir data --profile growth
"""

import argparse
import datetime as dt
import glob
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


# ---------- Utils ----------
def load_latest_features(data_dir: str = "data") -> str:
    """Retourne le chemin du dernier fichier features_today_*.csv (trié par nom/date)."""
    files = sorted(glob.glob(os.path.join(data_dir, "features_today_*.csv")))
    if not files:
        raise SystemExit("Aucun fichier 'features_today_*.csv' trouvé. Lance d'abord compute_features_auto.py.")
    return files[-1]


def winsorize(s: pd.Series, p: float = 0.02) -> pd.Series:
    """Coupe les extrêmes pour limiter l'influence des outliers."""
    if s.isna().all():
        return s
    try:
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s


def zscore(s: pd.Series, inverse: bool = False) -> pd.Series:
    """
    Z-score robuste (winsorisé), 0 si std=0 ou série vide.
    Si inverse=True, inverse le signe (pour métriques où plus bas = mieux)
    """
    s = pd.to_numeric(s, errors="coerce")
    s = winsorize(s.astype(float))
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - mu) / sd
    if inverse:
        z = -z
    return z


def compute_category_scores(df: pd.DataFrame, feature_columns: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Calcule le score par catégorie comme moyenne simple des z-scores de ses colonnes présentes.
    Certaines métriques sont inversées (ex: volatilité, dette).
    """
    cat_scores = {}
    
    # Métriques à inverser (plus bas = mieux)
    inverse_metrics = {
        'pe_ttm', 'pe_forward', 'peg_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda',
        'volatility_30d', 'debt_to_equity', 'debt_to_assets', 'net_debt_ebitda',
        'beta_spx'  # Beta plus faible = moins volatil
    }
    
    for cat, cols in feature_columns.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            cat_scores[cat] = pd.Series(0.0, index=df.index)
            continue
        
        zs_list = []
        for c in present:
            inverse = c in inverse_metrics
            zs_list.append(zscore(df[c], inverse=inverse))
        
        zs = pd.concat(zs_list, axis=1)
        cat_scores[cat] = zs.mean(axis=1)
    
    return pd.DataFrame(cat_scores)


def calculate_data_completeness(df: pd.DataFrame, all_feature_cols: List[str]) -> pd.Series:
    """Calcule le % de données disponibles pour chaque ticker."""
    # Ne garder que les colonnes qui existent réellement dans le DataFrame
    existing_cols = [c for c in all_feature_cols if c in df.columns]
    if not existing_cols:
        return pd.Series(0.0, index=df.index)
    
    available = df[existing_cols].notna().sum(axis=1)
    total = len(existing_cols)
    return (available / total) * 100


def apply_advanced_rules(df: pd.DataFrame, penalties: dict, bonuses: dict) -> pd.Series:
    """
    Applique bonus/malus AMÉLIORÉS basés sur les métriques en %.
    Retourne une série d'ajustements.
    """
    adj = pd.Series(0.0, index=df.index, dtype=float)

    # ========== PÉNALITÉS ==========
    if penalties:
        # Croissance négative
        if 'negative_revenue_growth' in penalties and 'revenue_yoy' in df.columns:
            mask = pd.to_numeric(df['revenue_yoy'], errors='coerce') < 0
            adj.loc[mask.fillna(False)] += penalties['negative_revenue_growth']
        
        if 'negative_eps_growth' in penalties and 'eps_yoy' in df.columns:
            mask = pd.to_numeric(df['eps_yoy'], errors='coerce') < 0
            adj.loc[mask.fillna(False)] += penalties['negative_eps_growth']
        
        # Leverage élevé
        thr = penalties.get('high_debt_to_equity_threshold')
        pts = penalties.get('high_debt_to_equity_points', -8)
        if thr and 'debt_to_equity' in df.columns:
            mask = pd.to_numeric(df['debt_to_equity'], errors='coerce') > thr
            adj.loc[mask.fillna(False)] += pts
        
        thr = penalties.get('high_net_debt_ebitda_threshold')
        pts = penalties.get('high_net_debt_ebitda_points', -6)
        if thr and 'net_debt_ebitda' in df.columns:
            mask = pd.to_numeric(df['net_debt_ebitda'], errors='coerce') > thr
            adj.loc[mask.fillna(False)] += pts
        
        # Faible liquidité
        thr = penalties.get('low_current_ratio_threshold')
        pts = penalties.get('low_current_ratio_points', -5)
        if thr and 'current_ratio' in df.columns:
            mask = pd.to_numeric(df['current_ratio'], errors='coerce') < thr
            adj.loc[mask.fillna(False)] += pts
        
        # Marges faibles
        thr = penalties.get('low_operating_margin_threshold')
        pts = penalties.get('low_operating_margin_points', -4)
        if thr and 'operating_margin' in df.columns:
            mask = pd.to_numeric(df['operating_margin'], errors='coerce') < thr
            adj.loc[mask.fillna(False)] += pts
        
        # Volatilité excessive
        thr = penalties.get('high_volatility_threshold')
        pts = penalties.get('high_volatility_points', -3)
        if thr and 'volatility_30d' in df.columns:
            mask = pd.to_numeric(df['volatility_30d'], errors='coerce') > thr
            adj.loc[mask.fillna(False)] += pts
        
        # PE élevé sans croissance
        pe_thr = penalties.get('high_pe_low_growth_pe_threshold')
        gr_thr = penalties.get('high_pe_low_growth_growth_threshold')
        pts = penalties.get('high_pe_low_growth_points', -6)
        if pe_thr and gr_thr and 'pe_ttm' in df.columns and 'eps_cagr_3y' in df.columns:
            mask = (pd.to_numeric(df['pe_ttm'], errors='coerce') > pe_thr) & \
                   (pd.to_numeric(df['eps_cagr_3y'], errors='coerce') < gr_thr)
            adj.loc[mask.fillna(False)] += pts

    # ========== BONUS ==========
    if bonuses:
        # Forte croissance revenus
        thr = bonuses.get('strong_revenue_cagr_3y_threshold')
        pts = bonuses.get('strong_revenue_cagr_3y_points', 4)
        if thr and 'revenue_cagr_3y' in df.columns:
            mask = pd.to_numeric(df['revenue_cagr_3y'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Forte croissance EPS
        thr = bonuses.get('strong_eps_cagr_3y_threshold')
        pts = bonuses.get('strong_eps_cagr_3y_points', 5)
        if thr and 'eps_cagr_3y' in df.columns:
            mask = pd.to_numeric(df['eps_cagr_3y'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Excellentes marges opérationnelles
        thr = bonuses.get('excellent_operating_margin_threshold')
        pts = bonuses.get('excellent_operating_margin_points', 4)
        if thr and 'operating_margin' in df.columns:
            mask = pd.to_numeric(df['operating_margin'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Excellente marge FCF
        thr = bonuses.get('excellent_fcf_margin_threshold')
        pts = bonuses.get('excellent_fcf_margin_points', 3)
        if thr and 'fcf_margin' in df.columns:
            mask = pd.to_numeric(df['fcf_margin'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Excellente ROE
        thr = bonuses.get('excellent_roe_threshold')
        pts = bonuses.get('excellent_roe_points', 5)
        if thr and 'roe' in df.columns:
            mask = pd.to_numeric(df['roe'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Excellent ROIC
        thr = bonuses.get('excellent_roic_threshold')
        pts = bonuses.get('excellent_roic_points', 4)
        if thr and 'roic' in df.columns:
            mask = pd.to_numeric(df['roic'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # PEG attractif
        thr = bonuses.get('attractive_peg_threshold')
        pts = bonuses.get('attractive_peg_points', 3)
        if thr and 'peg_ratio' in df.columns:
            mask = (pd.to_numeric(df['peg_ratio'], errors='coerce') <= thr) & \
                   (pd.to_numeric(df['peg_ratio'], errors='coerce') > 0)
            adj.loc[mask.fillna(False)] += pts
        
        # FCF Yield élevé
        thr = bonuses.get('high_fcf_yield_threshold')
        pts = bonuses.get('high_fcf_yield_points', 3)
        if thr and 'fcf_yield' in df.columns:
            mask = pd.to_numeric(df['fcf_yield'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Dividende élevé
        thr = bonuses.get('high_dividend_yield_threshold')
        pts = bonuses.get('high_dividend_yield_points', 2)
        if thr and 'dividend_yield' in df.columns:
            mask = pd.to_numeric(df['dividend_yield'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Momentum fort
        thr = bonuses.get('strong_mom_6m_threshold')
        pts = bonuses.get('strong_mom_6m_points', 3)
        if thr and 'mom_6m' in df.columns:
            mask = pd.to_numeric(df['mom_6m'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Excellente liquidité
        thr = bonuses.get('excellent_current_ratio_threshold')
        pts = bonuses.get('excellent_current_ratio_points', 2)
        if thr and 'current_ratio' in df.columns:
            mask = pd.to_numeric(df['current_ratio'], errors='coerce') >= thr
            adj.loc[mask.fillna(False)] += pts
        
        # Dette faible
        thr = bonuses.get('low_debt_to_equity_threshold')
        pts = bonuses.get('low_debt_to_equity_points', 3)
        if thr and 'debt_to_equity' in df.columns:
            mask = pd.to_numeric(df['debt_to_equity'], errors='coerce') <= thr
            adj.loc[mask.fillna(False)] += pts

    return adj


# ---------- Main ----------
def main(config_path: str = "config.yaml", data_dir: str = "data", profile: str = "growth") -> None:
    # Charger la config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feature_columns = cfg.get("feature_columns", {})
    profiles = cfg.get("profiles", {})
    penalties = cfg.get("penalties", {})
    bonuses = cfg.get("bonuses", {})
    scores_history_file = cfg.get("scores_history_file", os.path.join(data_dir, "scores_history.csv"))

    if profile not in profiles:
        raise SystemExit(f"Profil inconnu '{profile}'. Profils disponibles : {list(profiles.keys())}")

    print(f"\n🎯 Profil sélectionné : {profile}")
    print(f"📝 Description : {profiles[profile].get('description', 'N/A')}")

    latest_file = load_latest_features(data_dir)
    print(f"📂 Fichier de features : {os.path.basename(latest_file)}")
    
    feats = pd.read_csv(latest_file)
    if "ticker" not in feats.columns:
        raise SystemExit("Le fichier de features ne contient pas la colonne 'ticker'.")

    feats = feats.dropna(subset=["ticker"]).reset_index(drop=True)
    print(f"📊 {len(feats)} tickers à analyser")

    # Calculer complétude des données
    all_feature_cols = [c for cols in feature_columns.values() for c in cols]
    feats['data_completeness'] = calculate_data_completeness(feats, all_feature_cols)

    # Scores par catégorie
    print(f"\n⚙️  Calcul des z-scores par catégorie...")
    cat_scores = compute_category_scores(feats, feature_columns)

    # Agrégation pondérée par profil
    print(f"⚙️  Agrégation selon profil {profile}...")
    weights = profiles[profile].get("weights", {})
    agg = pd.Series(0.0, index=feats.index, dtype=float)
    
    print(f"\n📊 Pondérations appliquées :")
    for cat, w in weights.items():
        if cat in cat_scores.columns:
            agg += float(w) * cat_scores[cat]
            print(f"   • {cat.capitalize()}: {w*100:.0f}%")

    # Bonus / Malus
    print(f"\n⚙️  Application des bonus/malus...")
    adj = apply_advanced_rules(feats, penalties, bonuses)
    total = agg + adj

    # Normalisation du score sur 0-100
    score_min, score_max = total.min(), total.max()
    if score_max > score_min:
        total_normalized = ((total - score_min) / (score_max - score_min)) * 100
    else:
        total_normalized = pd.Series(50.0, index=total.index)

    # Sortie
    base_cols = [c for c in ["ticker", "name", "sector", "region", "style"] if c in feats.columns]
    out = feats[base_cols].copy()

    for c in cat_scores.columns:
        out[f"z_{c}"] = cat_scores[c].round(3)

    out["score_raw"] = agg.round(3)
    out["score_adj"] = adj.round(3)
    out["score"] = total_normalized.round(2)
    out["data_completeness"] = feats["data_completeness"].round(1)
    out["profile"] = profile

    # Date à partir du nom du fichier de features
    m = re.search(r"features_today_(\d{2}-\d{2}-\d{2})\.csv$", os.path.basename(latest_file))
    asof = m.group(1) if m else dt.datetime.now().strftime("%d-%m-%y")
    out["date"] = asof

    # Écrit scores_today
    out_today = os.path.join(data_dir, "scores_today.csv")
    out.sort_values("score", ascending=False).to_csv(out_today, index=False, encoding="utf-8")

    # Met à jour l'historique
    if os.path.exists(scores_history_file):
        hist = pd.read_csv(scores_history_file)
        # Évite les doublons pour la même date et profil
        hist = hist[~((hist["date"] == asof) & (hist["profile"] == profile))]
        hist = pd.concat([hist, out], ignore_index=True)
    else:
        hist = out

    hist.to_csv(scores_history_file, index=False, encoding="utf-8")

    # Statistiques
    print(f"\n✅ Scores calculés et sauvegardés !")
    print(f"📁 scores_today.csv → {out_today}")
    print(f"📁 scores_history.csv → {scores_history_file}")
    
    print(f"\n📊 Statistiques des scores :")
    print(f"   • Score moyen : {out['score'].mean():.1f}")
    print(f"   • Score médian : {out['score'].median():.1f}")
    print(f"   • Score max : {out['score'].max():.1f}")
    print(f"   • Score min : {out['score'].min():.1f}")
    
    print(f"\n📊 Distribution :")
    excellent = (out['score'] >= 70).sum()
    good = ((out['score'] >= 50) & (out['score'] < 70)).sum()
    fair = ((out['score'] >= 30) & (out['score'] < 50)).sum()
    poor = (out['score'] < 30).sum()
    
    print(f"   • Excellente (≥70) : {excellent} ({excellent/len(out)*100:.1f}%)")
    print(f"   • Bonne (50-70) : {good} ({good/len(out)*100:.1f}%)")
    print(f"   • Moyenne (30-50) : {fair} ({fair/len(out)*100:.1f}%)")
    print(f"   • Faible (<30) : {poor} ({poor/len(out)*100:.1f}%)")
    
    print(f"\n🎯 Top 5 opportunités :")
    top5 = out.nlargest(5, 'score')[['ticker', 'name', 'score', 'data_completeness']]
    for idx, row in top5.iterrows():
        print(f"   {row['ticker']:6s} - {row['name'][:30]:30s} Score: {row['score']:5.1f} (Données: {row['data_completeness']:.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--profile", default="growth",
                    help="Profil de pondération : growth, value, quality, blend, turnaround, dividend")
    args = ap.parse_args()
    main(config_path=args.config, data_dir=args.data_dir, profile=args.profile)