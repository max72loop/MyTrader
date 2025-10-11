# App/data/loaders.py
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Charge la configuration depuis config.yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_dir="data"):
    """Charge les fichiers de données nécessaires au dashboard"""
    data_path = Path(data_dir)
    
    # Charger features_latest.csv
    features_file = data_path / "features_latest.csv"
    if features_file.exists():
        df_feats = pd.read_csv(features_file)
    else:
        df_feats = pd.DataFrame()
    
    # Charger scores_today.csv
    scores_file = data_path / "scores_today.csv"
    if scores_file.exists():
        df_today = pd.read_csv(scores_file)
    else:
        df_today = pd.DataFrame()
    
    # Charger scores_history.csv
    history_file = data_path / "scores_history.csv"
    if history_file.exists():
        df_hist = pd.read_csv(history_file)
    else:
        df_hist = pd.DataFrame()
    
    return df_feats, df_today, df_hist


def apply_filters(df, sectors_sel=None, regions_sel=None):
    """Applique les filtres de secteur et région au DataFrame"""
    filtered = df.copy()
    
    if sectors_sel and len(sectors_sel) > 0 and "sector" in df.columns:
        filtered = filtered[filtered["sector"].isin(sectors_sel)]
    
    if regions_sel and len(regions_sel) > 0 and "region" in df.columns:
        filtered = filtered[filtered["region"].isin(regions_sel)]
    
    return filtered


def dataframe_to_percent(df, digits=2):
    """Formate les colonnes numériques du DataFrame"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].round(digits)
    return df_copy