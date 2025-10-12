"""
backtest.py
-----------
Système de backtesting pour valider la performance des scores MyTrader.

Teste si les actions avec les meilleurs scores surperforment réellement le marché.

Usage:
    python backtest.py --start 2022-01-01 --end 2024-01-01 --profile growth --top 10
    python backtest.py --start 2023-01-01 --benchmark ^GSPC --rebalance monthly
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import yaml

# Import des fonctions de scoring
from compute_scores import compute_category_scores, apply_advanced_rules


class Backtester:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.results = []
        self.trades = []
    
    def download_historical_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Télécharge les données historiques pour tous les tickers."""
        print(f"\n📥 Téléchargement des données historiques...")
        print(f"   Période : {start_date} à {end_date}")
        print(f"   Tickers : {len(tickers)}")
        
        all_data = {}
        failed = []
        
        for ticker in tqdm(tickers, desc="Downloading"):
            try:
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    auto_adjust=True
                )
                if not df.empty and len(df) > 20:
                    all_data[ticker] = df['Close']
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
        
        if failed:
            print(f"   ⚠️  {len(failed)} tickers échoués : {', '.join(failed[:5])}...")
        
        # Créer un DataFrame avec tous les prix
        prices_df = pd.DataFrame(all_data)
        print(f"   ✅ {len(prices_df.columns)} tickers téléchargés avec succès")
        
        return prices_df
    
    def calculate_historical_scores(
        self,
        prices_df: pd.DataFrame,
        date: pd.Timestamp,
        lookback_months: int = 12,
        profile: str = "growth"
    ) -> pd.DataFrame:
        """
        Calcule les scores pour une date donnée en utilisant
        les données disponibles jusqu'à cette date.
        """
        # Simulation : on ne peut pas recalculer tous les fondamentaux historiques
        # On va utiliser le momentum et la volatilité comme proxy
        
        end_date = date
        start_date = date - pd.DateOffset(months=lookback_months)
        
        # Filtrer les prix jusqu'à cette date
        historical_prices = prices_df.loc[:end_date, :]
        
        if len(historical_prices) < 60:  # Besoin de min 60 jours
            return pd.DataFrame()
        
        scores = []
        
        for ticker in historical_prices.columns:
            try:
                px = historical_prices[ticker].dropna()
                if len(px) < 60:
                    continue
                
                # Calculer momentum
                mom_3m = ((px.iloc[-1] / px.iloc[-63]) - 1) * 100 if len(px) >= 63 else np.nan
                mom_6m = ((px.iloc[-1] / px.iloc[-126]) - 1) * 100 if len(px) >= 126 else np.nan
                
                # Calculer volatilité
                returns = px.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Score simple basé sur momentum et volatilité
                # Plus le momentum est élevé et la volatilité faible, meilleur le score
                momentum_score = (mom_3m if not np.isnan(mom_3m) else 0) + (mom_6m if not np.isnan(mom_6m) else 0)
                volatility_penalty = -volatility if not np.isnan(volatility) else 0
                
                # Score normalisé sur 0-100
                raw_score = momentum_score + (volatility_penalty * 0.5)
                
                scores.append({
                    'ticker': ticker,
                    'date': date,
                    'score': raw_score,
                    'mom_3m': mom_3m,
                    'mom_6m': mom_6m,
                    'volatility': volatility,
                    'price': px.iloc[-1]
                })
            
            except Exception as e:
                continue
        
        df_scores = pd.DataFrame(scores)
        
        if not df_scores.empty:
            # Normaliser les scores sur 0-100
            min_score = df_scores['score'].min()
            max_score = df_scores['score'].max()
            if max_score > min_score:
                df_scores['score'] = ((df_scores['score'] - min_score) / (max_score - min_score)) * 100
            else:
                df_scores['score'] = 50.0
        
        return df_scores
    
    def run_backtest(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        profile: str = "growth",
        top_n: int = 10,
        rebalance_freq: str = "monthly",
        initial_capital: float = 100000.0,
        benchmark: str = "^GSPC"
    ) -> Dict:
        """
        Execute le backtest complet.
        """
        print(f"\n{'='*80}")
        print(f"🚀 BACKTESTING MyTrader - Profil {profile.upper()}")
        print(f"{'='*80}")
        print(f"📅 Période : {start_date} à {end_date}")
        print(f"💰 Capital initial : ${initial_capital:,.0f}")
        print(f"🎯 Stratégie : Top {top_n} tickers")
        print(f"🔄 Rééquilibrage : {rebalance_freq}")
        print(f"📊 Benchmark : {benchmark}")
        print(f"{'='*80}\n")
        
        # Télécharger les données
        prices_df = self.download_historical_data(tickers, start_date, end_date)
        
        if prices_df.empty:
            print("❌ Aucune donnée téléchargée")
            return {}
        
        # Télécharger le benchmark
        print(f"\n📥 Téléchargement du benchmark {benchmark}...")
        benchmark_data = yf.download(benchmark, start=start_date, end=end_date, progress=False, auto_adjust=True)
        benchmark_prices = benchmark_data['Close']
        
        # Générer les dates de rééquilibrage
        if rebalance_freq == "monthly":
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif rebalance_freq == "quarterly":
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        elif rebalance_freq == "weekly":
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        else:  # daily
            rebalance_dates = prices_df.index
        
        # Filtrer pour garder seulement les dates où on a des données
        rebalance_dates = [d for d in rebalance_dates if d in prices_df.index]
        
        print(f"\n🔄 {len(rebalance_dates)} dates de rééquilibrage")
        
        # Variables de portfolio
        portfolio_value = initial_capital
        cash = initial_capital
        holdings = {}  # {ticker: shares}
        portfolio_history = []
        benchmark_history = []
        
        # Pour chaque date de rééquilibrage
        for i, rebal_date in enumerate(tqdm(rebalance_dates, desc="Backtesting")):
            
            # Calculer les scores à cette date
            scores_df = self.calculate_historical_scores(
                prices_df,
                rebal_date,
                lookback_months=12,
                profile=profile
            )
            
            if scores_df.empty:
                continue
            
            # Sélectionner le top N
            top_picks = scores_df.nlargest(top_n, 'score')
            
            # Liquider les positions actuelles
            if holdings:
                for ticker, shares in holdings.items():
                    if ticker in prices_df.columns and rebal_date in prices_df.index:
                        price = prices_df.loc[rebal_date, ticker]
                        if not np.isnan(price):
                            cash += shares * price
                holdings = {}
            
            # Acheter les nouvelles positions (répartition égale)
            allocation_per_stock = cash / len(top_picks)
            
            for _, row in top_picks.iterrows():
                ticker = row['ticker']
                if ticker in prices_df.columns and rebal_date in prices_df.index:
                    price = prices_df.loc[rebal_date, ticker]
                    if not np.isnan(price) and price > 0:
                        shares = allocation_per_stock / price
                        holdings[ticker] = shares
                        cash -= shares * price
                        
                        # Enregistrer le trade
                        self.trades.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'score': row['score']
                        })
            
            # Calculer la valeur du portfolio
            portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in prices_df.columns and rebal_date in prices_df.index:
                    price = prices_df.loc[rebal_date, ticker]
                    if not np.isnan(price):
                        portfolio_value += shares * price
            
            # Enregistrer l'historique
            portfolio_history.append({
                'date': rebal_date,
                'value': portfolio_value,
                'cash': cash,
                'num_holdings': len(holdings)
            })
            
            # Benchmark value
            if rebal_date in benchmark_prices.index:
                bench_value = (benchmark_prices.loc[rebal_date] / benchmark_prices.iloc[0]) * initial_capital
                benchmark_history.append({
                    'date': rebal_date,
                    'value': bench_value
                })
        
        # Convertir en DataFrames
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        benchmark_df = pd.DataFrame(benchmark_history).set_index('date')
        
        # Calculer les métriques de performance
        results = self.calculate_performance_metrics(
            portfolio_df,
            benchmark_df,
            initial_capital
        )
        
        # Ajouter les DataFrames aux résultats
        results['portfolio_history'] = portfolio_df
        results['benchmark_history'] = benchmark_df
        results['trades'] = pd.DataFrame(self.trades)
        
        return results
    
    def calculate_performance_metrics(
        self,
        portfolio_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """Calcule les métriques de performance."""
        
        # Returns
        portfolio_return = ((portfolio_df['value'].iloc[-1] / initial_capital) - 1) * 100
        benchmark_return = ((benchmark_df['value'].iloc[-1] / initial_capital) - 1) * 100
        
        # Annualized returns
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = days / 365.25
        
        portfolio_cagr = (pow(portfolio_df['value'].iloc[-1] / initial_capital, 1 / years) - 1) * 100
        benchmark_cagr = (pow(benchmark_df['value'].iloc[-1] / initial_capital, 1 / years) - 1) * 100
        
        # Volatilité
        portfolio_returns = portfolio_df['value'].pct_change().dropna()
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        
        benchmark_returns = benchmark_df['value'].pct_change().dropna()
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assume risk-free rate of 2%)
        risk_free_rate = 2.0
        portfolio_sharpe = (portfolio_cagr - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        benchmark_sharpe = (benchmark_cagr - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Max Drawdown
        portfolio_cummax = portfolio_df['value'].cummax()
        portfolio_drawdown = ((portfolio_df['value'] - portfolio_cummax) / portfolio_cummax) * 100
        max_drawdown = portfolio_drawdown.min()
        
        benchmark_cummax = benchmark_df['value'].cummax()
        benchmark_drawdown = ((benchmark_df['value'] - benchmark_cummax) / benchmark_cummax) * 100
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Win rate
        positive_periods = (portfolio_returns > 0).sum()
        total_periods = len(portfolio_returns)
        win_rate = (positive_periods / total_periods) * 100 if total_periods > 0 else 0
        
        # Alpha (excès de rendement)
        alpha = portfolio_cagr - benchmark_cagr
        
        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_df['value'].iloc[-1],
            'total_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'cagr': portfolio_cagr,
            'benchmark_cagr': benchmark_cagr,
            'alpha': alpha,
            'volatility': portfolio_volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'years': years
        }
    
    def print_results(self, results: Dict):
        """Affiche les résultats du backtest."""
        print(f"\n{'='*80}")
        print(f"📊 RÉSULTATS DU BACKTEST")
        print(f"{'='*80}\n")
        
        print(f"💰 PERFORMANCE FINANCIÈRE")
        print(f"{'-'*80}")
        print(f"   Capital initial        : ${results['initial_capital']:>15,.0f}")
        print(f"   Valeur finale          : ${results['final_value']:>15,.0f}")
        print(f"   Gain/Perte             : ${results['final_value'] - results['initial_capital']:>15,.0f}")
        print(f"   Return total           : {results['total_return']:>14,.2f}%")
        print(f"   Return benchmark       : {results['benchmark_return']:>14,.2f}%")
        print(f"   🎯 ALPHA               : {results['alpha']:>14,.2f}%")
        
        print(f"\n📈 RENDEMENTS ANNUALISÉS")
        print(f"{'-'*80}")
        print(f"   CAGR Strategy          : {results['cagr']:>14,.2f}%")
        print(f"   CAGR Benchmark         : {results['benchmark_cagr']:>14,.2f}%")
        print(f"   Période                : {results['years']:>14,.1f} ans")
        
        print(f"\n⚠️  RISQUE")
        print(f"{'-'*80}")
        print(f"   Volatilité Strategy    : {results['volatility']:>14,.2f}%")
        print(f"   Volatilité Benchmark   : {results['benchmark_volatility']:>14,.2f}%")
        print(f"   Max Drawdown Strategy  : {results['max_drawdown']:>14,.2f}%")
        print(f"   Max Drawdown Benchmark : {results['benchmark_max_drawdown']:>14,.2f}%")
        
        print(f"\n🎯 RATIOS")
        print(f"{'-'*80}")
        print(f"   Sharpe Ratio Strategy  : {results['sharpe_ratio']:>14,.2f}")
        print(f"   Sharpe Ratio Benchmark : {results['benchmark_sharpe']:>14,.2f}")
        print(f"   Win Rate               : {results['win_rate']:>14,.1f}%")
        print(f"   Nombre de trades       : {results['num_trades']:>15,}")
        
        # Conclusion
        print(f"\n{'='*80}")
        if results['alpha'] > 0:
            print(f"✅ STRATÉGIE GAGNANTE : +{results['alpha']:.2f}% vs benchmark")
        else:
            print(f"❌ STRATÉGIE PERDANTE : {results['alpha']:.2f}% vs benchmark")
        print(f"{'='*80}\n")
    
    def plot_results(self, results: Dict, output_file: str = "backtest_results.html"):
        """Génère des graphiques interactifs des résultats."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            portfolio_df = results['portfolio_history']
            benchmark_df = results['benchmark_history']
            
            # Créer les subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Performance du Portfolio vs Benchmark',
                    'Drawdown',
                    'Distribution des Trades'
                ),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # 1. Performance
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['value'],
                    name='Strategy',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_df['value'],
                    name='Benchmark',
                    line=dict(color='#fa709a', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. Drawdown
            portfolio_cummax = portfolio_df['value'].cummax()
            drawdown = ((portfolio_df['value'] - portfolio_cummax) / portfolio_cummax) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='#fa709a', width=1)
                ),
                row=2, col=1
            )
            
            # 3. Distribution des scores des trades
            if 'trades' in results and not results['trades'].empty:
                trades_df = results['trades']
                fig.add_trace(
                    go.Histogram(
                        x=trades_df['score'],
                        name='Score Distribution',
                        nbinsx=20,
                        marker_color='#667eea'
                    ),
                    row=3, col=1
                )
            
            # Layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                title_text=f"Backtest Results - Alpha: {results['alpha']:.2f}%",
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=3, col=1)
            
            # Sauvegarder
            fig.write_html(output_file)
            print(f"📊 Graphiques sauvegardés : {output_file}")
            
        except ImportError:
            print("⚠️  Plotly non installé, graphiques non générés")


def main():
    parser = argparse.ArgumentParser(description="Backtest MyTrader strategies")
    parser.add_argument("--start", default="2022-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-01", help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--profile", default="growth", help="Profil de stratégie")
    parser.add_argument("--top", type=int, default=10, help="Nombre de top stocks")
    parser.add_argument("--rebalance", default="monthly", choices=['daily', 'weekly', 'monthly', 'quarterly'], help="Fréquence de rééquilibrage")
    parser.add_argument("--capital", type=float, default=100000, help="Capital initial")
    parser.add_argument("--benchmark", default="^GSPC", help="Benchmark (ticker Yahoo)")
    parser.add_argument("--companies", default="companies.csv", help="Fichier companies.csv")
    parser.add_argument("--output", default="backtest_results.html", help="Fichier de sortie graphiques")
    
    args = parser.parse_args()
    
    # Charger la liste des tickers
    companies_df = pd.read_csv(args.companies)
    tickers = companies_df['ticker'].dropna().tolist()
    
    # Créer le backtester
    backtester = Backtester()
    
    # Exécuter le backtest
    results = backtester.run_backtest(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        profile=args.profile,
        top_n=args.top,
        rebalance_freq=args.rebalance,
        initial_capital=args.capital,
        benchmark=args.benchmark
    )
    
    if results:
        # Afficher les résultats
        backtester.print_results(results)
        
        # Générer les graphiques
        backtester.plot_results(results, output_file=args.output)
        
        # Sauvegarder les résultats détaillés
        results_file = f"backtest_results_{args.profile}_{args.start}_{args.end}.csv"
        if 'portfolio_history' in results:
            results['portfolio_history'].to_csv(results_file)
            print(f"💾 Résultats détaillés sauvegardés : {results_file}")


if __name__ == "__main__":
    main()