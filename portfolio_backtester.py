"""
딥러닝 기반 포트폴리오 백테스팅 시스템
Deep Learning Based Portfolio Backtesting System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 임포트
from font_config import apply_korean_style, safe_title, safe_xlabel, safe_ylabel

# 한글 폰트 스타일 적용
apply_korean_style()

class PortfolioBacktester:
    """딥러닝 기반 포트폴리오 백테스터"""
    
    def __init__(self, universe_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None):
        self.universe_data = universe_data
        self.volume_data = volume_data
        self.returns_data = universe_data.pct_change().fillna(0)
        
    def factor_to_weights(self, factor_scores: pd.DataFrame, method: str = 'rank', 
                         long_only: bool = True, max_position: float = 0.1) -> pd.DataFrame:
        """팩터 점수를 포트폴리오 가중치로 변환"""
        
        weights = pd.DataFrame(0, index=factor_scores.index, columns=factor_scores.columns)
        
        for date in factor_scores.index:
            scores = factor_scores.loc[date].dropna()
            
            if len(scores) == 0:
                continue
                
            if method == 'rank':
                # 순위 기반 가중치 (상위 종목에 높은 가중치)
                ranks = scores.rank(pct=True)
                if long_only:
                    # 상위 50% 종목만 매수
                    selected = ranks >= 0.5
                    if selected.sum() > 0:
                        raw_weights = ranks[selected]
                        # 정규화
                        normalized_weights = raw_weights / raw_weights.sum()
                        # 최대 포지션 제한
                        capped_weights = np.minimum(normalized_weights, max_position)
                        # 재정규화
                        final_weights = capped_weights / capped_weights.sum()
                        weights.loc[date, final_weights.index] = final_weights
                else:
                    # 롱숏 전략 (상위 30% 매수, 하위 30% 매도)
                    long_mask = ranks >= 0.7
                    short_mask = ranks <= 0.3
                    
                    if long_mask.sum() > 0:
                        long_weights = ranks[long_mask] / ranks[long_mask].sum() * 0.5
                        weights.loc[date, long_weights.index] = long_weights
                    
                    if short_mask.sum() > 0:
                        short_weights = -(1 - ranks[short_mask]) / (1 - ranks[short_mask]).sum() * 0.5
                        weights.loc[date, short_weights.index] = short_weights
                        
            elif method == 'linear':
                # 선형 가중치
                if long_only:
                    positive_scores = np.maximum(scores, 0)
                    if positive_scores.sum() > 0:
                        raw_weights = positive_scores / positive_scores.sum()
                        capped_weights = np.minimum(raw_weights, max_position)
                        final_weights = capped_weights / capped_weights.sum()
                        weights.loc[date, final_weights.index] = final_weights
                else:
                    # 점수를 직접 가중치로 사용 (정규화)
                    total_abs = np.abs(scores).sum()
                    if total_abs > 0:
                        raw_weights = scores / total_abs
                        weights.loc[date, raw_weights.index] = raw_weights
        
        return weights
    
    def calculate_transaction_costs(self, weights: pd.DataFrame, cost_bps: float = 10) -> pd.Series:
        """거래비용 계산 (단위: bps)"""
        weight_changes = weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * (cost_bps / 10000)
        return transaction_costs.fillna(0)
    
    def run_backtest(self, factor_scores: pd.DataFrame, 
                    method: str = 'rank',
                    long_only: bool = True,
                    rebalance_freq: str = 'daily',
                    transaction_cost_bps: float = 10,
                    max_position: float = 0.1) -> Dict:
        """포트폴리오 백테스팅 실행"""
        
        try:
            # 1. 팩터 점수를 가중치로 변환
            weights = self.factor_to_weights(
                factor_scores, method=method, 
                long_only=long_only, max_position=max_position
            )
            
            # 리밸런싱 빈도 조정
            if rebalance_freq == 'weekly':
                # 매주 금요일 또는 마지막 거래일에만 리밸런싱
                rebalance_dates = weights.index[weights.index.dayofweek == 4]  # 금요일
                weights = weights.reindex(weights.index).fillna(method='ffill')
                for date in weights.index:
                    if date not in rebalance_dates:
                        if date != weights.index[0]:  # 첫날 제외
                            weights.loc[date] = weights.shift(1).loc[date]
            elif rebalance_freq == 'monthly':
                # 매월 마지막 거래일에만 리밸런싱
                rebalance_dates = weights.resample('M').last().index
                weights = weights.reindex(weights.index).fillna(method='ffill')
                for date in weights.index:
                    if date not in rebalance_dates:
                        if date != weights.index[0]:
                            weights.loc[date] = weights.shift(1).loc[date]
            
            # 2. 포트폴리오 수익률 계산
            # 전일 가중치로 당일 수익률 계산
            portfolio_returns = (weights.shift(1) * self.returns_data).sum(axis=1).fillna(0)
            
            # 3. 거래비용 차감
            transaction_costs = self.calculate_transaction_costs(weights, transaction_cost_bps)
            net_returns = portfolio_returns - transaction_costs
            
            # 4. 누적 수익률 계산
            cumulative_returns = (1 + net_returns).cumprod()
            
            # 5. 벤치마크 (동일가중 포트폴리오)
            equal_weight_returns = self.returns_data.mean(axis=1)
            benchmark_cumulative = (1 + equal_weight_returns).cumprod()
            
            # 6. 성과 지표 계산
            performance_metrics = self._calculate_performance_metrics(
                net_returns, cumulative_returns, benchmark_cumulative.pct_change().fillna(0)
            )
            
            return {
                'portfolio_returns': net_returns,
                'cumulative_returns': cumulative_returns,
                'benchmark_returns': equal_weight_returns,
                'benchmark_cumulative': benchmark_cumulative,
                'weights': weights,
                'transaction_costs': transaction_costs,
                'performance_metrics': performance_metrics,
                'factor_scores': factor_scores
            }
            
        except Exception as e:
            st.error(f"백테스팅 실행 중 오류: {e}")
            return None
    
    def _calculate_performance_metrics(self, returns: pd.Series, 
                                     cumulative_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict:
        """성과 지표 계산"""
        
        # 기본 통계
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # 최대 손실폭 (Maximum Drawdown)
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 벤치마크 대비 성과
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        excess_return = annualized_return - benchmark_annualized
        
        # 승률 (일일 수익률 기준)
        win_rate = (returns > 0).mean()
        
        # 정보 비율 (Information Ratio)
        excess_daily_returns = returns - benchmark_returns
        tracking_error = excess_daily_returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # 칼마 비율 (Calmar Ratio)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'benchmark_return': benchmark_annualized
        }
    
    def plot_results(self, backtest_results: Dict, title: str = "포트폴리오 백테스팅 결과"):
        """백테스팅 결과 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 누적 수익률 비교
        ax1 = axes[0, 0]
        backtest_results['cumulative_returns'].plot(ax=ax1, label='포트폴리오', linewidth=2)
        backtest_results['benchmark_cumulative'].plot(ax=ax1, label='벤치마크 (동일가중)', linewidth=2, alpha=0.7)
        ax1.set_title('누적 수익률 비교', fontsize=14)
        ax1.set_ylabel('누적 수익률', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 일별 수익률 분포
        ax2 = axes[0, 1]
        returns = backtest_results['portfolio_returns']
        ax2.hist(returns, bins=50, alpha=0.7, density=True)
        ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'평균: {returns.mean():.4f}')
        ax2.set_title('일별 수익률 분포', fontsize=14)
        ax2.set_xlabel('일별 수익률', fontsize=12)
        ax2.set_ylabel('확률 밀도', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 드로우다운
        ax3 = axes[1, 0]
        cumulative = backtest_results['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=ax3, color='red', alpha=0.7)
        ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax3.set_title('드로우다운', fontsize=14)
        ax3.set_ylabel('드로우다운 (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. 월별 수익률 히트맵
        ax4 = axes[1, 1]
        monthly_returns = backtest_results['portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        if len(monthly_table) > 0:
            sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('월별 수익률', fontsize=14)
            ax4.set_xlabel('월', fontsize=12)
            ax4.set_ylabel('연도', fontsize=12)
        else:
            ax4.text(0.5, 0.5, '월별 데이터 부족', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('월별 수익률', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def create_performance_report(self, backtest_results: Dict) -> pd.DataFrame:
        """성과 리포트 생성"""
        
        metrics = backtest_results['performance_metrics']
        
        report_data = {
            '지표': [
                '총 수익률', '연간 수익률', '연간 변동성', '샤프 비율',
                '최대 손실폭', '승률', '벤치마크 대비 초과수익',
                '정보 비율', '칼마 비율'
            ],
            '값': [
                f"{metrics['total_return']:.2%}",
                f"{metrics['annualized_return']:.2%}",
                f"{metrics['annualized_volatility']:.2%}",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['win_rate']:.2%}",
                f"{metrics['excess_return']:.2%}",
                f"{metrics['information_ratio']:.3f}",
                f"{metrics['calmar_ratio']:.3f}"
            ],
            '해석': [
                '전체 기간 총 수익률',
                '연환산 수익률',
                '연환산 변동성 (위험도)',
                '위험 대비 수익률 (>1 우수)',
                '최대 누적 손실 (절댓값)',
                '수익을 낸 거래일 비율',
                '벤치마크 대비 추가 수익',
                '추적오차 대비 초과수익 (>0.5 우수)',
                '최대손실 대비 수익률 (>1 우수)'
            ]
        }
        
        return pd.DataFrame(report_data)

class FactorBacktester:
    """팩터별 백테스팅 시스템"""
    
    def __init__(self, universe_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None):
        self.backtester = PortfolioBacktester(universe_data, volume_data)
        
    def compare_factors(self, factors_dict: Dict[str, pd.DataFrame], 
                       factor_names_ko: Dict[str, str]) -> Dict[str, Dict]:
        """여러 팩터 성과 비교"""
        
        results = {}
        
        st.info(f"🔄 {len(factors_dict)}개 팩터 백테스팅 비교 시작...")
        
        progress_bar = st.progress(0)
        
        for i, (factor_name, factor_data) in enumerate(factors_dict.items()):
            progress_bar.progress((i + 1) / len(factors_dict))
            
            korean_name = factor_names_ko.get(factor_name, factor_name)
            
            try:
                # 개별 팩터 백테스팅
                result = self.backtester.run_backtest(
                    factor_data, 
                    method='rank',
                    long_only=True,
                    rebalance_freq='daily',
                    transaction_cost_bps=10
                )
                
                if result:
                    results[korean_name] = result
                    st.success(f"✅ {korean_name} 팩터 백테스팅 완료")
                    
            except Exception as e:
                st.warning(f"⚠️ {korean_name} 팩터 백테스팅 실패: {e}")
                continue
        
        progress_bar.empty()
        
        if results:
            st.success(f"✅ {len(results)}개 팩터 백테스팅 완료")
            return results
        else:
            st.error("❌ 백테스팅에 성공한 팩터가 없습니다.")
            return {}
    
    def _display_factor_comparison(self, results: Dict[str, Dict]):
        """팩터 비교 결과 표시"""
        
        st.subheader("📊 팩터별 성과 비교")
        
        # 1. 누적 수익률 비교 차트
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for factor_name, result in results.items():
            result['cumulative_returns'].plot(
                ax=ax, label=factor_name, linewidth=2, alpha=0.8
            )
        
        # 벤치마크 추가 (첫 번째 결과의 벤치마크 사용)
        first_result = list(results.values())[0]
        first_result['benchmark_cumulative'].plot(
            ax=ax, label='벤치마크 (동일가중)', 
            linewidth=2, linestyle='--', color='black', alpha=0.7
        )
        
        ax.set_title('팩터별 누적 수익률 비교', fontsize=14)
        ax.set_ylabel('누적 수익률', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # 2. 성과 지표 비교 테이블
        st.subheader("📈 주요 성과 지표 비교")
        
        comparison_data = []
        for factor_name, result in results.items():
            metrics = result['performance_metrics']
            comparison_data.append({
                '팩터': factor_name,
                '총 수익률': f"{metrics['total_return']:.2%}",
                '연간 수익률': f"{metrics['annualized_return']:.2%}",
                '연간 변동성': f"{metrics['annualized_volatility']:.2%}",
                '샤프 비율': f"{metrics['sharpe_ratio']:.3f}",
                '최대 손실폭': f"{metrics['max_drawdown']:.2%}",
                '승률': f"{metrics['win_rate']:.2%}",
                '정보 비율': f"{metrics['information_ratio']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # 3. 팩터별 위험-수익 산점도
        st.subheader("🎯 위험-수익 산점도")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        returns = []
        volatilities = []
        names = []
        
        for factor_name, result in results.items():
            metrics = result['performance_metrics']
            returns.append(metrics['annualized_return'])
            volatilities.append(metrics['annualized_volatility'])
            names.append(factor_name)
        
        scatter = ax.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(names)), cmap='tab10')
        
        for i, name in enumerate(names):
            ax.annotate(name, (volatilities[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('연간 변동성 (위험)', fontsize=12)
        ax.set_ylabel('연간 수익률', fontsize=12)
        ax.set_title('팩터별 위험-수익 프로파일', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
