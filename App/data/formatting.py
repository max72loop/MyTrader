# app/data/formatting.py
import pandas as pd
from typing import List


PERCENT_COLS_HINTS = [
"return","ret","perf","change","delta","var","pct","yoy","qoq","mom",
"growth","taux","ratio","yield","winrate","margin","marge","roic","roe",
"roa","score_change","d_score","volatility","fcf_yield",
]


def looks_like_fraction(series: pd.Series) -> bool:
s = pd.to_numeric(series, errors="coerce").dropna()
if len(s) == 0:
return False
frac_like = (s >= -1.5) & (s <= 1.5)
return frac_like.mean() > 0.8


def is_percent_column(col_name: str, series: pd.Series) -> bool:
name_hit = any(h in col_name.lower() for h in PERCENT_COLS_HINTS)
frac_hit = looks_like_fraction(series)
return name_hit or frac_hit


def to_percent_str(x, digits: int = 2):
try:
if pd.isna(x):
return ""
if isinstance(x, str):
return x
if -1.5 <= float(x) <= 1.5:
return f"{float(x)*100:.{digits}f}%"
return f"{float(x):.{digits}f}%"
except Exception:
return str(x)


def dataframe_to_percent(df: pd.DataFrame, extra_percent_cols: List[str] | None = None, digits: int = 2) -> pd.DataFrame:
if df is None or df.empty:
return df
extra_percent_cols = extra_percent_cols or []
df2 = df.copy()
for col in df2.columns:
if col in extra_percent_cols or is_percent_column(col, df2[col]):
df2[col] = df2[col].apply(lambda v: to_percent_str(v, digits))
return df2