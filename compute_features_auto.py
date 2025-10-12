"""
compute_features_auto.py - VERSION AMÉLIORÉE
---------------------------------------------
Télécharge les données de marché & fondamentaux pour les tickers listés
dans companies.csv, calcule des features standardisées avec PLUS de pourcentages
et génère :

- data/features_today_DD-MM-YY.csv
- data/features_latest.csv (alias du plus récent)

NOUVEAUTÉS :
- ROE, ROA, ROIC en %
- Marges (brute, opérationnelle, nette) en %
- Croissance revenues et EPS sur 1, 3, 5 ans en %
- Rendement du dividende en %
- Payout ratio en %
- Asset turnover ratio
- Current ratio et Quick ratio
- Free Cash Flow margin en %

Usage :
    python compute_features_auto.py --companies companies.csv --outdir data
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


def cagr(start_val, end_val, periods):
    """Calcule le CAGR (Compound Annual Growth Rate) en %"""
    try:
        if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0 or end_val <= 0 or periods <= 0:
            return np.nan
        return (pow(end_val / start_val, 1 / periods) - 1) * 100
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
    """Volatilité annualisée = std(ret) * sqrt(252) en %."""
    try:
        px = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )["Close"]
        r = px.pct_change().dropna()
        return float(r.std() * np.sqrt(252) * 100)  # en %
    except Exception:
        return np.nan


def compute_beta(ticker: str, ref: str = "^GSPC", period: str = "1y") -> float:
    """Beta par régression OLS vs indice ref (par défaut S&P 500)."""
    try:
        px = yf.download([ticker, ref], period=period, interval="1d", auto_adjust=True, progress=False)
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


# --------- Fundamentals AMÉLIORÉS ---------
def fundamentals_features(ticker: str) -> dict:
    """
    Récupère des comptes (ttm approx via quarterly) et calcule des ratios.
    VERSION AMÉLIORÉE avec plus de pourcentages.
    """
    feats = {
        # === Momentum / Marché ===
        "mom_1m": np.nan,
        "mom_3m": np.nan,
        "mom_6m": np.nan,
        "mom_12m": np.nan,
        "beta_spx": np.nan,
        "volatility_30d": np.nan,
        
        # === Valorisation ===
        "pe_ttm": np.nan,
        "pe_forward": np.nan,
        "peg_ratio": np.nan,
        "ps_ratio": np.nan,  # Price/Sales
        "pb_ratio": np.nan,
        "ev_ebitda": np.nan,
        "fcf_yield": np.nan,  # %
        "earnings_yield": np.nan,  # %
        "dividend_yield": np.nan,  # %
        
        # === Croissance (en %) ===
        "revenue_yoy": np.nan,
        "revenue_cagr_3y": np.nan,
        "revenue_cagr_5y": np.nan,
        "eps_yoy": np.nan,
        "eps_cagr_3y": np.nan,
        "eps_cagr_5y": np.nan,
        "ebitda_yoy": np.nan,
        "fcf_yoy": np.nan,
        
        # === Qualité / Marges (en %) ===
        "gross_margin": np.nan,
        "operating_margin": np.nan,
        "net_margin": np.nan,
        "ebitda_margin": np.nan,
        "fcf_margin": np.nan,
        "roe": np.nan,  # %
        "roa": np.nan,  # %
        "roic": np.nan,  # %
        "roc": np.nan,  # Return on Capital %
        
        # === Efficacité ===
        "asset_turnover": np.nan,
        "inventory_turnover": np.nan,
        "receivables_turnover": np.nan,
        
        # === Risque / Leverage ===
        "debt_to_equity": np.nan,
        "debt_to_assets": np.nan,  # %
        "net_debt_ebitda": np.nan,
        "interest_coverage": np.nan,
        "current_ratio": np.nan,
        "quick_ratio": np.nan,
        
        # === Dividendes ===
        "payout_ratio": np.nan,  # %
        "dividend_growth_5y": np.nan,  # % CAGR
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        finfo = getattr(tk, "fast_info", None)
        
        # Données de base
        shares = info.get("sharesOutstanding") or (getattr(finfo, "shares", None) if finfo else None)
        mcap = info.get("marketCap") or (getattr(finfo, "market_cap", None) if finfo else None)
        last_price = info.get("currentPrice") or (getattr(finfo, "last_price", None) if finfo else None)
        
        # États financiers
        fin_q = tk.quarterly_financials
        fin_a = tk.financials
        bs_q = tk.quarterly_balance_sheet
        bs_a = tk.balance_sheet
        cf_q = tk.quarterly_cashflow
        cf_a = tk.cashflow

        # ========== MOMENTUM ==========
        feats["mom_1m"] = momentum(ticker, months=1)
        feats["mom_3m"] = momentum(ticker, months=3)
        feats["mom_6m"] = momentum(ticker, months=6)
        feats["mom_12m"] = momentum(ticker, months=12)

        # ========== VALORISATION ==========
        feats["pe_ttm"] = info.get("trailingPE")
        feats["pe_forward"] = info.get("forwardPE")
        feats["peg_ratio"] = info.get("pegRatio")
        feats["ps_ratio"] = info.get("priceToSalesTrailing12Months")
        feats["pb_ratio"] = info.get("priceToBook")
        feats["ev_ebitda"] = info.get("enterpriseToEbitda")
        
        # Dividend Yield
        feats["dividend_yield"] = info.get("dividendYield")
        if feats["dividend_yield"] and not pd.isna(feats["dividend_yield"]):
            feats["dividend_yield"] *= 100  # Convertir en %
        
        # Earnings Yield = 1/PE
        if feats["pe_ttm"] and feats["pe_ttm"] > 0:
            feats["earnings_yield"] = (1 / feats["pe_ttm"]) * 100

        # ========== REVENUS & CROISSANCE ==========
        def _sum_last4(df, row):
            if df is not None and not df.empty and row in df.index:
                return df.loc[row].iloc[:4].sum(min_count=1)
            return np.nan
        
        def _get_annual(df, row, col_idx=0):
            if df is not None and not df.empty and row in df.index and df.shape[1] > col_idx:
                return df.loc[row].iloc[col_idx]
            return np.nan

        # Revenus TTM
        revenue_ttm = _sum_last4(fin_q, "Total Revenue") if fin_q is not None else _get_annual(fin_a, "Total Revenue")
        
        # Croissance Revenus
        if fin_a is not None and not fin_a.empty and "Total Revenue" in fin_a.index:
            rev_history = fin_a.loc["Total Revenue"].dropna().sort_index(ascending=False)
            if len(rev_history) >= 2:
                feats["revenue_yoy"] = ((rev_history.iloc[0] / rev_history.iloc[1]) - 1) * 100
            if len(rev_history) >= 4:
                feats["revenue_cagr_3y"] = cagr(rev_history.iloc[3], rev_history.iloc[0], 3)
            if len(rev_history) >= 6:
                feats["revenue_cagr_5y"] = cagr(rev_history.iloc[5], rev_history.iloc[0], 5)

        # Net Income TTM
        net_income_ttm = _sum_last4(fin_q, "Net Income") if fin_q is not None else _get_annual(fin_a, "Net Income")
        
        # Croissance EPS
        if fin_a is not None and not fin_a.empty and "Net Income" in fin_a.index and shares:
            ni_history = fin_a.loc["Net Income"].dropna().sort_index(ascending=False)
            if len(ni_history) >= 2:
                eps0 = safe_div(ni_history.iloc[0], shares)
                eps1 = safe_div(ni_history.iloc[1], shares)
                feats["eps_yoy"] = ((eps0 / eps1) - 1) * 100 if eps1 and eps1 > 0 else np.nan
            if len(ni_history) >= 4:
                eps0 = safe_div(ni_history.iloc[0], shares)
                eps3 = safe_div(ni_history.iloc[3], shares)
                feats["eps_cagr_3y"] = cagr(eps3, eps0, 3)
            if len(ni_history) >= 6:
                eps0 = safe_div(ni_history.iloc[0], shares)
                eps5 = safe_div(ni_history.iloc[5], shares)
                feats["eps_cagr_5y"] = cagr(eps5, eps0, 5)

        # EBITDA TTM
        ebitda_ttm = _sum_last4(fin_q, "EBITDA") if fin_q is not None else _get_annual(fin_a, "EBITDA")
        if fin_a is not None and not fin_a.empty and "EBITDA" in fin_a.index:
            eb_history = fin_a.loc["EBITDA"].dropna().sort_index(ascending=False)
            if len(eb_history) >= 2:
                feats["ebitda_yoy"] = ((eb_history.iloc[0] / eb_history.iloc[1]) - 1) * 100

        # Operating Income TTM
        operating_income_ttm = _sum_last4(fin_q, "Operating Income") if fin_q is not None else _get_annual(fin_a, "Operating Income")
        
        # Gross Profit TTM
        gross_profit_ttm = _sum_last4(fin_q, "Gross Profit") if fin_q is not None else _get_annual(fin_a, "Gross Profit")

        # ========== MARGES (en %) ==========
        if revenue_ttm and revenue_ttm > 0:
            if gross_profit_ttm:
                feats["gross_margin"] = (gross_profit_ttm / revenue_ttm) * 100
            if operating_income_ttm:
                feats["operating_margin"] = (operating_income_ttm / revenue_ttm) * 100
            if net_income_ttm:
                feats["net_margin"] = (net_income_ttm / revenue_ttm) * 100
            if ebitda_ttm:
                feats["ebitda_margin"] = (ebitda_ttm / revenue_ttm) * 100

        # ========== FREE CASH FLOW ==========
        fcf_ttm = _sum_last4(cf_q, "Free Cash Flow") if cf_q is not None else _get_annual(cf_a, "Free Cash Flow")
        
        if fcf_ttm and mcap and mcap > 0:
            feats["fcf_yield"] = (fcf_ttm / mcap) * 100
        
        if fcf_ttm and revenue_ttm and revenue_ttm > 0:
            feats["fcf_margin"] = (fcf_ttm / revenue_ttm) * 100
        
        # Croissance FCF
        if cf_a is not None and not cf_a.empty and "Free Cash Flow" in cf_a.index:
            fcf_history = cf_a.loc["Free Cash Flow"].dropna().sort_index(ascending=False)
            if len(fcf_history) >= 2:
                feats["fcf_yoy"] = ((fcf_history.iloc[0] / fcf_history.iloc[1]) - 1) * 100

        # ========== BILAN ==========
        total_assets = _get_annual(bs_a, "Total Assets") if bs_a is not None else _get_annual(bs_q, "Total Assets")
        total_equity = _get_annual(bs_a, "Total Stockholder Equity") if bs_a is not None else _get_annual(bs_q, "Total Stockholder Equity")
        total_debt = _get_annual(bs_a, "Total Debt") if bs_a is not None else _get_annual(bs_q, "Total Debt")
        cash = _get_annual(bs_a, "Cash And Cash Equivalents") if bs_a is not None else _get_annual(bs_q, "Cash And Cash Equivalents")
        current_assets = _get_annual(bs_a, "Current Assets") if bs_a is not None else _get_annual(bs_q, "Current Assets")
        current_liabilities = _get_annual(bs_a, "Current Liabilities") if bs_a is not None else _get_annual(bs_q, "Current Liabilities")
        inventory = _get_annual(bs_a, "Inventory") if bs_a is not None else _get_annual(bs_q, "Inventory")
        receivables = _get_annual(bs_a, "Receivables") if bs_a is not None else _get_annual(bs_q, "Receivables")

        # ========== RENTABILITÉ (en %) ==========
        if net_income_ttm:
            if total_equity and total_equity > 0:
                feats["roe"] = (net_income_ttm / total_equity) * 100
            if total_assets and total_assets > 0:
                feats["roa"] = (net_income_ttm / total_assets) * 100

        # ROIC = NOPAT / Invested Capital
        if operating_income_ttm:
            tax_rate = 0.21  # Approximation
            nopat = operating_income_ttm * (1 - tax_rate)
            invested_capital = (total_debt if total_debt else 0) + (total_equity if total_equity else 0) - (cash if cash else 0)
            if invested_capital > 0:
                feats["roic"] = (nopat / invested_capital) * 100

        # Return on Capital
        if ebitda_ttm and total_assets and total_assets > 0:
            feats["roc"] = (ebitda_ttm / total_assets) * 100

        # ========== EFFICACITÉ ==========
        if revenue_ttm and total_assets and total_assets > 0:
            feats["asset_turnover"] = revenue_ttm / total_assets
        
        if revenue_ttm and inventory and inventory > 0:
            feats["inventory_turnover"] = revenue_ttm / inventory
        
        if revenue_ttm and receivables and receivables > 0:
            feats["receivables_turnover"] = revenue_ttm / receivables

        # ========== LEVERAGE & LIQUIDITÉ ==========
        if total_debt and total_equity and total_equity > 0:
            feats["debt_to_equity"] = total_debt / total_equity
        
        if total_debt and total_assets and total_assets > 0:
            feats["debt_to_assets"] = (total_debt / total_assets) * 100
        
        net_debt = (total_debt if total_debt else 0) - (cash if cash else 0)
        if ebitda_ttm and ebitda_ttm > 0:
            feats["net_debt_ebitda"] = net_debt / ebitda_ttm
        
        # Interest Coverage
        interest_expense = _get_annual(fin_a, "Interest Expense") if fin_a is not None else np.nan
        if operating_income_ttm and interest_expense and interest_expense < 0:
            feats["interest_coverage"] = operating_income_ttm / abs(interest_expense)
        
        # Current & Quick Ratio
        if current_assets and current_liabilities and current_liabilities > 0:
            feats["current_ratio"] = current_assets / current_liabilities
            quick_assets = current_assets - (inventory if inventory else 0)
            feats["quick_ratio"] = quick_assets / current_liabilities

        # ========== DIVIDENDES ==========
        dividend_per_share = info.get("dividendRate")
        if dividend_per_share and shares and net_income_ttm and net_income_ttm > 0:
            total_dividends = dividend_per_share * shares
            feats["payout_ratio"] = (total_dividends / net_income_ttm) * 100
        
        # Croissance dividende 5 ans
        div_growth = info.get("fiveYearAvgDividendYield")
        if div_growth:
            feats["dividend_growth_5y"] = div_growth * 100

    except Exception as e:
        print(f"⚠️  Erreur pour {ticker}: {str(e)}")
        pass

    return feats


# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", default="companies.csv", help="Chemin vers companies.csv")
    ap.add_argument("--outdir", default="data", help="Dossier de sortie (data)")
    ap.add_argument("--ref", default="^GSPC", help="Indice de référence pour le beta")
    ap.add_argument("--sleep", type=float, default=0.5, help="Pause (sec) entre tickers pour Yahoo")
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
    print(f"→ Téléchargement & calcul des features AMÉLIORÉES pour {len(tickers)} tickers ...")
    print(f"📊 Nouvelles métriques : marges %, croissance CAGR %, ratios de liquidité, etc.")

    for t in tqdm(tickers, desc="Processing"):
        row = {"ticker": t}

        # Marché
        row["beta_spx"] = compute_beta(t, ref=args.ref, period="1y")
        row["volatility_30d"] = annualized_vol(t, period="6mo")

        # Fondamentaux améliorés
        feats = fundamentals_features(t)
        row.update(feats)

        rows.append(row)
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

    print(f"\n✅ Features améliorées sauvegardées : {out_file}")
    print(f"🔗 Alias mis à jour : {outdir / 'features_latest.csv'}")
    print(f"\n📊 Métriques calculées par ticker :")
    print(f"   • Momentum : 1m, 3m, 6m, 12m")
    print(f"   • Valorisation : PE, PEG, PS, PB, EV/EBITDA, FCF Yield, Dividend Yield")
    print(f"   • Croissance : Revenue YoY/CAGR 3y/5y, EPS YoY/CAGR 3y/5y")
    print(f"   • Marges : Brute, Opérationnelle, Nette, EBITDA, FCF (toutes en %)")
    print(f"   • Rentabilité : ROE, ROA, ROIC, ROC (toutes en %)")
    print(f"   • Liquidité : Current Ratio, Quick Ratio, Interest Coverage")
    print(f"   • Efficacité : Asset/Inventory/Receivables Turnover")


if __name__ == "__main__":
    main()