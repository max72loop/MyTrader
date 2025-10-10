"""
compute_features_auto.py
------------------------
Télécharge les données de marché & fondamentaux pour les tickers listés
dans companies.csv, calcule des features standardisées et génère :

- data/features_today_DD-MM-YY.csv
- data/features_latest.csv (alias du plus récent)

Usage :
    python compute_features_auto.py --companies companies.csv --outdir data

Dépendances :
    pip install yfinance pandas numpy statsmodels tqdm
"""

import argparse
import datetime as dt
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# --------- Utils ---------
def safe_div(a, b):
    try:
        if a is None or b is None:
            return np.nan
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def momentum(ticker: str, months: int = 3) -> float:
    """Rendement simple sur ~22 jours boursiers * months (approx)."""
    try:
        period = "1y" if months > 6 else "6mo"
        px = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )["Close"]
        if len(px) < 22 * months:
            return np.nan
        ret = (px.iloc[-1] / px.iloc[-22 * months]) - 1.0
        return float(ret * 100.0)  # en %
    except Exception:
        return np.nan


def annualized_vol(ticker: str, period: str = "6mo") -> float:
    """Volatilité annualisée = std(ret) * sqrt(252)."""
    try:
        px = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )["Close"]
        r = px.pct_change().dropna()
        return float(r.std() * np.sqrt(252))
    except Exception:
        return np.nan


def compute_beta(ticker: str, ref: str = "^GSPC", period: str = "1y") -> float:
    """Beta par régression OLS vs indice ref (par défaut S&P 500)."""
    try:
        px = yf.download([ticker, ref], period=period, interval="1d", auto_adjust=True, progress=False)
        # MultiIndex -> garder Close
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"]
        px = px.dropna(how="any")
        r_t = px[ticker].pct_change().dropna()
        r_m = px[ref].pct_change().dropna()
        df = pd.concat([r_t, r_m], axis=1).dropna()
        df.columns = ["r_t", "r_m"]
        if len(df) < 30:
            return np.nan
        X = sm.add_constant(df["r_m"].values)
        y = df["r_t"].values
        model = sm.OLS(y, X).fit()
        return float(model.params[1])
    except Exception:
        return np.nan


# --------- Fundamentals (via yfinance) ---------
def fundamentals_features(ticker: str) -> dict:
    """
    Récupère des comptes (ttm approx via quarterly) et calcule des ratios.
    NB : yfinance peut retourner des champs manquants selon les titres.
    """
    feats = {
        # Momentum / Marché
        "mom_3m": np.nan,
        "mom_6m": np.nan,
        "beta_spx": np.nan,
        "volatility_30d": np.nan,
        # Valorisation
        "pe_ttm": np.nan,
        "fcf_yield": np.nan,
        "pb_ratio": np.nan,
        # Croissance
        "revenue_yoy": np.nan,
        "eps_yoy_1y": np.nan,
        # Qualité
        "gross_margin_ttm": np.nan,
        "oper_margin": np.nan,
        "roe": np.nan,
        "roa": np.nan,
        "roic_ttm": np.nan,
        # Risque / Leverage
        "net_debt_ebitda": np.nan,
        "debt_equity": np.nan,
    }

    try:
        tk = yf.Ticker(ticker)
        # fast_info (peut manquer certains champs)
        finfo = getattr(tk, "fast_info", None)
        shares = getattr(finfo, "shares", None) if finfo is not None else None
        mcap = getattr(finfo, "market_cap", None) if finfo is not None else None
        last_price = getattr(finfo, "last_price", None) if finfo is not None else None

        # États financiers (peuvent être vides selon titre)
        fin_q = tk.quarterly_financials  # Income Statement trimestriel
        bs_q = tk.quarterly_balance_sheet
        cf_q = tk.quarterly_cashflow
        fin_a = tk.financials  # annuel (parfois plus fourni)

        # ---------- Marges ----------
        revenue_ttm = np.nan
        operating_income_ttm = np.nan
        gross_profit_ttm = np.nan
        net_income_ttm = np.nan

        # Utilise trimestriel si dispo, sinon annuel
        try:
            if fin_q is not None and not fin_q.empty:
                # Somme des 4 derniers trimestres
                def _sum_last4(df, row):
                    if row in df.index:
                        return df.loc[row].iloc[:4].sum(min_count=1)
                    return np.nan

                revenue_ttm = _sum_last4(fin_q, "Total Revenue")
                operating_income_ttm = _sum_last4(fin_q, "Operating Income")
                gross_profit_ttm = _sum_last4(fin_q, "Gross Profit")
                net_income_ttm = _sum_last4(fin_q, "Net Income")
            elif fin_a is not None and not fin_a.empty:
                # Si on n'a que l'annuel, prend dernière colonne
                def _last(df, row):
                    if row in df.index:
                        return df.loc[row].iloc[0]
                    return np.nan

                revenue_ttm = _last(fin_a, "Total Revenue")
                operating_income_ttm = _last(fin_a, "Operating Income")
                gross_profit_ttm = _last(fin_a, "Gross Profit")
                net_income_ttm = _last(fin_a, "Net Income")
        except Exception:
            pass

        # Gross margin & Operating margin
        if pd.notna(revenue_ttm) and revenue_ttm != 0:
            if pd.notna(gross_profit_ttm):
                feats["gross_margin_ttm"] = safe_div(gross_profit_ttm, revenue_ttm) * 100.0
            if pd.notna(operating_income_ttm):
                feats["oper_margin"] = safe_div(operating_income_ttm, revenue_ttm) * 100.0

        # ---------- Valorisation ----------
        # PE (TTM approximé via net income / shares)
        if pd.notna(net_income_ttm) and shares and last_price:
            eps_ttm = safe_div(net_income_ttm, shares)
            feats["pe_ttm"] = safe_div(last_price, eps_ttm)

        # P/B : market cap / book equity
        try:
            if bs_q is not None and not bs_q.empty and mcap:
                if "Total Stockholder Equity" in bs_q.index:
                    equity_last = bs_q.loc["Total Stockholder Equity"].iloc[0]
                    feats["pb_ratio"] = safe_div(mcap, equity_last)
        except Exception:
            pass

        # FCF yield : FCF TTM / market cap
        try:
            if cf_q is not None and not cf_q.empty and mcap:
                if "Free Cash Flow" in cf_q.index:
                    fcf_ttm = cf_q.loc["Free Cash Flow"].iloc[:4].sum(min_count=1)
                    feats["fcf_yield"] = safe_div(fcf_ttm, mcap) * 100.0
        except Exception:
            pass

        # ---------- Croissance (YoY approx) ----------
        # Revenus YoY (annuel si dispo sinon trimestriel à n-4)
        try:
            if fin_a is not None and not fin_a.empty and "Total Revenue" in fin_a.index and fin_a.shape[1] >= 2:
                rev_latest = fin_a.loc["Total Revenue"].iloc[0]
                rev_prev = fin_a.loc["Total Revenue"].iloc[1]
                feats["revenue_yoy"] = safe_div(rev_latest, rev_prev) * 100.0 - 100.0
            elif fin_q is not None and not fin_q.empty and "Total Revenue" in fin_q.index and fin_q.shape[1] >= 5:
                rev_latest = fin_q.loc["Total Revenue"].iloc[0]
                rev_prev = fin_q.loc["Total Revenue"].iloc[4]  # ~n-4 trimestres
                feats["revenue_yoy"] = safe_div(rev_latest, rev_prev) * 100.0 - 100.0
        except Exception:
            pass

        # EPS YoY (EPS ~ NetIncome / shares)
        try:
            if shares and pd.notna(shares):
                if fin_a is not None and not fin_a.empty and "Net Income" in fin_a.index and fin_a.shape[1] >= 2:
                    eps_latest = safe_div(fin_a.loc["Net Income"].iloc[0], shares)
                    eps_prev = safe_div(fin_a.loc["Net Income"].iloc[1], shares)
                    feats["eps_yoy_1y"] = safe_div(eps_latest, eps_prev) * 100.0 - 100.0
                elif fin_q is not None and not fin_q.empty and "Net Income" in fin_q.index and fin_q.shape[1] >= 5:
                    eps_latest = safe_div(fin_q.loc["Net Income"].iloc[0], shares)
                    eps_prev = safe_div(fin_q.loc["Net Income"].iloc[4], shares)
                    feats["eps_yoy_1y"] = safe_div(eps_latest, eps_prev) * 100.0 - 100.0
        except Exception:
            pass

        # ---------- Leverage & Qualité ----------
        total_debt = np.nan
        cash = np.nan
        total_assets = np.nan
        equity = np.nan
        try:
            if bs_q is not None and not bs_q.empty:
                if "Total Debt" in bs_q.index:
                    total_debt = bs_q.loc["Total Debt"].iloc[0]
                if "Cash And Cash Equivalents" in bs_q.index:
                    cash = bs_q.loc["Cash And Cash Equivalents"].iloc[0]
                if "Total Assets" in bs_q.index:
                    total_assets = bs_q.loc["Total Assets"].iloc[0]
                if "Total Stockholder Equity" in bs_q.index:
                    equity = bs_q.loc["Total Stockholder Equity"].iloc[0]

            net_debt = (total_debt if pd.notna(total_debt) else 0.0) - (cash if pd.notna(cash) else 0.0)
            ebitda_proxy = operating_income_ttm  # proxy
            feats["net_debt_ebitda"] = safe_div(net_debt, ebitda_proxy)
            feats["debt_equity"] = safe_div(total_debt, equity)

            if pd.notna(net_income_ttm):
                feats["roa"] = safe_div(net_income_ttm, total_assets) * 100.0
                feats["roe"] = safe_div(net_income_ttm, equity) * 100.0

            # ROIC ~ NOPAT / (Debt + Equity - Cash)
            tax_rate = 0.21
            nopat = (operating_income_ttm * (1 - tax_rate)) if pd.notna(operating_income_ttm) else np.nan
            invested_capital = (total_debt if pd.notna(total_debt) else 0.0) + (equity if pd.notna(equity) else 0.0) - (cash if pd.notna(cash) else 0.0)
            feats["roic_ttm"] = safe_div(nopat, invested_capital) * 100.0
        except Exception:
            pass

    except Exception:
        # On laisse tout à NaN si une erreur générale survient
        pass

    return feats


# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", default="companies.csv", help="Chemin vers companies.csv")
    ap.add_argument("--outdir", default="data", help="Dossier de sortie (data)")
    ap.add_argument("--ref", default="^GSPC", help="Indice de référence pour le beta")
    ap.add_argument("--sleep", type=float, default=0.4, help="Pause (sec) entre tickers pour Yahoo")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Lire companies.csv
    dfc = pd.read_csv(args.companies)
    required_cols = ["ticker", "name", "sector", "region", "style"]
    missing = [c for c in required_cols if c not in dfc.columns]
    if missing:
        raise SystemExit(f"companies.csv incomplet, colonnes manquantes: {missing}")

    tickers = dfc["ticker"].dropna().astype(str).unique().tolist()

    rows = []
    print(f"→ Téléchargement & calcul des features pour {len(tickers)} tickers ...")

    for t in tqdm(tickers):
        row = {"ticker": t}

        # Marché
        row["mom_3m"] = momentum(t, months=3)
        row["mom_6m"] = momentum(t, months=6)
        row["beta_spx"] = compute_beta(t, ref=args.ref, period="1y")
        row["volatility_30d"] = annualized_vol(t, period="6mo")

        # Fondamentaux
        feats = fundamentals_features(t)
        # On remplace les clés marché par celles calculées au-dessus (au cas où)
        feats["mom_3m"] = row["mom_3m"]
        feats["mom_6m"] = row["mom_6m"]
        feats["beta_spx"] = row["beta_spx"]
        feats["volatility_30d"] = row["volatility_30d"]

        out = {"ticker": t}
        out.update(feats)
        rows.append(out)

        time.sleep(max(0.0, args.sleep))

    feats_df = pd.DataFrame(rows)

    # Merge métadonnées (name, sector, region, style)
    feats_df = feats_df.merge(dfc, on="ticker", how="left")

    # Sauvegardes
    today_str = dt.datetime.now().strftime("%d-%m-%y")
    out_file = outdir / f"features_today_{today_str}.csv"
    feats_df["asof_date"] = dt.datetime.now(dt.timezone.utc).date().isoformat()
    feats_df.to_csv(out_file, index=False, encoding="utf-8")
    feats_df.to_csv(outdir / "features_latest.csv", index=False, encoding="utf-8")

    print(f"✅ Features sauvegardées : {out_file}")
    print(f"🔗 Alias mis à jour : {outdir / 'features_latest.csv'}")


if __name__ == "__main__":
    main()
