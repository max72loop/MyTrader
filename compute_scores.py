"""
compute_scores.py
-----------------
Lit `config.yaml`, prend le dernier fichier de features (`data/features_today_DD-MM-YY.csv`),
calcule les z-scores par catégorie, agrège selon un profil (pondérations),
applique bonus/malus, puis écrit :

- data/scores_today.csv
- data/scores_history.csv (historique cumulé, configurable dans config.yaml)

Usage :
    python compute_scores.py --config config.yaml --data_dir data --profile growth

Dépendances :
    pip install pandas numpy pyyaml
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


def zscore(s: pd.Series) -> pd.Series:
    """Z-score robuste (winsorisé), 0 si std=0 ou série vide."""
    s = pd.to_numeric(s, errors="coerce")
    s = winsorize(s.astype(float))
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def compute_category_scores(df: pd.DataFrame, feature_columns: Dict[str, List[str]]) -> pd.DataFrame:
    """Calcule le score par catégorie comme moyenne simple des z-scores de ses colonnes présentes."""
    cat_scores = {}
    for cat, cols in feature_columns.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            # Si aucune colonne de la catégorie n’existe, on met 0 (ne pénalise pas)
            cat_scores[cat] = pd.Series(0.0, index=df.index)
            continue
        zs = pd.concat([zscore(df[c]) for c in present], axis=1)
        cat_scores[cat] = zs.mean(axis=1)
    return pd.DataFrame(cat_scores)


def apply_rules(df: pd.DataFrame, penalties: dict, bonuses: dict) -> pd.Series:
    """Applique bonus/malus simples définis dans config.yaml. Retourne une série d'ajustements."""
    adj = pd.Series(0.0, index=df.index, dtype=float)

    # --- Penalties ---
    if penalties:
        # Leverage élevé
        thr = penalties.get("high_leverage_threshold")
        pts = penalties.get("high_leverage_points", -3)
        if thr is not None and "debt_equity" in df.columns:
            mask = pd.to_numeric(df["debt_equity"], errors="coerce") > thr
            adj.loc[mask.fillna(False)] += pts

        # (exemple extensible) Mauvaise surprise earning (placeholder si tu ajoutes une colonne 'surprise_eps_pct')
        miss_thr = penalties.get("earnings_miss_threshold_pct")
        miss_pts = penalties.get("earnings_miss_points", -5)
        if miss_thr is not None and "surprise_eps_pct" in df.columns:
            mask = pd.to_numeric(df["surprise_eps_pct"], errors="coerce") <= miss_thr
            adj.loc[mask.fillna(False)] += miss_pts

    # --- Bonuses ---
    if bonuses:
        # Forte croissance EPS YoY
        thr = bonuses.get("eps_growth_strong_threshold_pct")
        pts = bonuses.get("eps_growth_strong_points", 2)
        if thr is not None and "eps_yoy_1y" in df.columns:
            mask = pd.to_numeric(df["eps_yoy_1y"], errors="coerce") >= thr
            adj.loc[mask.fillna(False)] += pts

        # Achat d’initiés (si tu ajoutes une colonne bool 'insider_buy')
        if "insider_buy_points" in bonuses and "insider_buy" in df.columns:
            mask = df["insider_buy"].astype(str).str.lower().isin(["1", "true", "yes"])
            adj.loc[mask.fillna(False)] += bonuses["insider_buy_points"]

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

    latest_file = load_latest_features(data_dir)
    feats = pd.read_csv(latest_file)
    if "ticker" not in feats.columns:
        raise SystemExit("Le fichier de features ne contient pas la colonne 'ticker'.")

    feats = feats.dropna(subset=["ticker"]).reset_index(drop=True)

    # Scores par catégorie
    cat_scores = compute_category_scores(feats, feature_columns)

    # Agrégation pondérée par profil
    weights = profiles[profile].get("weights", {})
    agg = pd.Series(0.0, index=feats.index, dtype=float)
    for cat, w in weights.items():
        if cat in cat_scores.columns:
            agg += float(w) * cat_scores[cat]

    # Bonus / Malus
    adj = apply_rules(feats, penalties, bonuses)
    total = agg + adj

    # Sortie
    base_cols = [c for c in ["ticker", "name", "sector", "region", "style"] if c in feats.columns]
    out = feats[base_cols].copy()

    for c in cat_scores.columns:
        out[f"z_{c}"] = cat_scores[c].round(3)

    out["score_raw"] = agg.round(3)
    out["score_adj"] = adj.round(3)
    out["score"] = total.round(3)
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
        # Évite les doublons pour la même date
        hist = hist[hist["date"] != asof]
        hist = pd.concat([hist, out], ignore_index=True)
    else:
        hist = out

    hist.to_csv(scores_history_file, index=False, encoding="utf-8")

    print(f"✅ scores_today -> {out_today}")
    print(f"🕒 history -> {scores_history_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--profile", default="growth",
                    help="Profil de pondération (doit exister dans config.yaml:profiles)")
    args = ap.parse_args()
    main(config_path=args.config, data_dir=args.data_dir, profile=args.profile)
