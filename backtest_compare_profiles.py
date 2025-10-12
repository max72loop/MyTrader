"""
backtest_compare_profiles.py
----------------------------
Compare la performance de tous les profils en backtest

Usage:
    python backtest_compare_profiles.py --start 2022-01-01 --end 2024-01-01
"""

import argparse
import pandas as pd
from backtest import Backtester


def compare_all_profiles(
    tickers,
    start_date,
    end_date,
    top_n=10,
    rebalance_freq="monthly",
    initial_capital=100000,
    benchmark="^GSPC"
):
    """Compare tous les profils disponibles."""
    
    profiles = ['growth', 'value', 'quality', 'blend', 'turnaround', 'dividend']
    
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPARAISON DES PROFILS - BACKTEST")
    print(f"{'='*80}\n")
    
    for profile in profiles:
        print(f"\n{'─'*80}")
        print(f"🎯 Testing profile: {profile.upper()}")
        print(f"{'─'*80}")
        
        backtester = Backtester()
        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            profile=profile,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            initial_capital=initial_capital,
            benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()
benchmark
        )
        
        if results:
            all_results[profile] = results
            print(f"\n   ✅ {profile}: Alpha = {results['alpha']:.2f}%, Sharpe = {results['sharpe_ratio']:.2f}")
    
    # Créer un tableau comparatif
    print(f"\n\n{'='*80}")
    print(f"📊 TABLEAU COMPARATIF DES PROFILS")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for profile, results in all_results.items():
        comparison_data.append({
            'Profil': profile.upper(),
            'Return (%)': results['total_return'],
            'CAGR (%)': results['cagr'],
            'Alpha (%)': results['alpha'],
            'Volatilité (%)': results['volatility'],
            'Sharpe': results['sharpe_ratio'],
            'Max DD (%)': results['max_drawdown'],
            'Win Rate (%)': results['win_rate'],
            'Trades': results['num_trades']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Alpha (%)', ascending=False)
    
    print(df_comparison.to_string(index=False))
    
    # Identifier le meilleur profil
    best_alpha = df_comparison.iloc[0]
    best_sharpe = df_comparison.loc[df_comparison['Sharpe'].idxmax()]
    
    print(f"\n{'='*80}")
    print(f"🏆 MEILLEURS PROFILS")
    print(f"{'='*80}")
    print(f"\n🥇 Meilleur Alpha : {best_alpha['Profil']} ({best_alpha['Alpha (%)']:.2f}%)")
    print(f"🥇 Meilleur Sharpe : {best_sharpe['Profil']} ({best_sharpe['Sharpe']:.2f})")
    
    # Sauvegarder
    df_comparison.to_csv('backtest_comparison.csv', index=False)
    print(f"\n💾 Comparaison sauvegardée : backtest_comparison.csv")
    
    # Graphique comparatif
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for profile, results in all_results.items():
            portfolio_df = results['portfolio_history']
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['value'],
                name=profile.upper(),
                mode='lines'
            ))
        
        # Ajouter le benchmark
        if 'benchmark_history' in list(all_results.values())[0]:
            benchmark_df = list(all_results.values())[0]['benchmark_history']
            fig.add_trace(go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df['value'],
                name='BENCHMARK',
                line=dict(dash='dash', color='black', width=2)
            ))
        
        fig.update_layout(
            title='Comparaison des Profils - Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=600
        )
        
        fig.write_html('backtest_profiles_comparison.html')
        print(f"📊 Graphique sauvegardé : backtest_profiles_comparison.html")
        
    except ImportError:
        pass
    
    return df_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--rebalance", default="monthly")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--benchmark", default="^GSPC")
    parser.add_argument("--companies", default="companies.csv")
    
    args = parser.parse_args()
    
    # Charger les tickers
    companies_df = pd.read_csv(args.companies)
    tickers = companies_df['ticker'].dropna().tolist()
    
    # Comparer les profils
    comparison = compare_all_profiles(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top,
        rebalance_freq=args.rebalance,
        initial_capital=args.capital,
        benchmark=
    )