# app/data/loaders.py
import os
import yaml
import pandas as pd
import streamlit as st


from .formatting import dataframe_to_percent, is_percent_column, to_percent_str, looks_like_fraction


__all__ = [
"load_config","load_data","apply_filters",
"dataframe_to_percent","is_percent_column","to_percent_str","looks_like_fraction",
]


def load_config():
try:
if os.path.exists("config.yaml"):
with open("config.yaml","r",encoding="utf-8") as f:
return yaml.safe_load(f)
except Exception as e:
st.sidebar.error(f"Erreur config: {e}")
return {}


@st.cache_data
def load_data():
feats_latest = os.path.join("data","features_latest.csv")
scores_today = os.path.join("data","scores_today.csv")
scores_hist = os.path.join("data","scores_history.csv")


df_feats = pd.read_csv(feats_latest) if os.path.exists(feats_latest) else pd.DataFrame()
df_today = pd.read_csv(scores_today) if os.path.exists(scores_today) else pd.DataFrame()
df_hist = pd.read_csv(scores_hist) if os.path.exists(scores_hist) else pd.DataFrame()
return df_feats, df_today, df_hist


def apply_filters(df, sectors_sel, regions_sel):
if df is None or df.empty:
return df
mask = pd.Series(True, index=df.index)
if sectors_sel and "sector" in df.columns:
mask &= df["sector"].isin(sectors_sel)
if regions_sel and "region" in df.columns:
mask &= df["region"].isin(regions_sel)
return df[mask].copy()