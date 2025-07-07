"""
메가-알파 시뮬레이션 엔진 (Mega-Alpha Simulation Engine)
Dynamic Factor Combination and Backtesting System
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import FactorConfig
from alpha_factors import AlphaFactorEngine
from portfolio_backtester import PortfolioBacktester
from utils import logger, save_factor_to_zoo

@dataclass
class MegaAlphaConfig:
    """메가-알파 시뮬레이션 설정"""
    # 백테스팅 기간
    start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"
    
    # 팩터 풀 설정
    factor_pool_size: int = 10  # 사용할 팩터 개수
    min_factor_weight: float = 0.05  # 최소 팩터 가중치
    max_factor_weight: float = 0.4   # 최대 팩터 가중치
    
    # 동적 결합 설정
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    lookback_window: int = 60  # IC 계산 기간
    ic_threshold: float = 0.02  # 최소 IC 임계값
    
    # 포트폴리오 설정
    universe_size: int = 20  # 투자 유니버스 크기
    long_only: bool = True
    transaction_cost_bps: float = 10
    max_position_weight: float = 0.1

class MegaAlphaEngine:
    """메가-알파 시뮬레이션 엔진"""
    
    def __init__(self, config: MegaAlphaConfig):
        self.config = config
        self.factor_engine = AlphaFactorEngine(FactorConfig())
        self.factor_weights_history = {}  # 일별 팩터 가중치 기록
        self.performance_metrics = {}
        
    def run_mega_alpha_simulation(self, universe_data: pd.DataFrame, 
                                volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """메가-알파 시뮬레이션 실행"""
        
        st.info("🚀 메가-알파 시뮬레이션을 시작합니다...")
        
        # 1. 팩터 풀 생성
        factor_pool = self._create_factor_pool(universe_data, volume_data)
        if not factor_pool:
            st.error("팩터 풀 생성에 실패했습니다.")
            return {}
        
        st.success(f"✅ {len(factor_pool)}개 팩터로 구성된 팩터 풀 생성 완료")
        
        # 2. 동적 팩터 결합 및 백테스팅
        mega_alpha_results = self._dynamic_factor_combination(factor_pool, universe_data)
        
        if not mega_alpha_results:
            st.error("메가-알파 시뮬레이션에 실패했습니다.")
            return {}
        
        # 3. 성과 분석 및 시각화
        self._analyze_mega_alpha_performance(mega_alpha_results)
        
        return mega_alpha_results
    
    def _create_factor_pool(self, universe_data: pd.DataFrame, 
                          volume_data: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """팩터 풀 생성"""
        
        # 모든 가능한 팩터 타입
        all_factor_types = [
            'momentum', 'reversal', 'volatility', 'rsi', 'price_to_ma',
            'bollinger_band', 'macd', 'stochastic', 'williams_r', 'cci'
        ]
        
        if volume_data is not None:
            all_factor_types.extend([
                'volume', 'money_flow', 'obv', 'volume_price_trend',
                'chaikin_money_flow', 'force_index', 'ease_of_movement',
                'accumulation_distribution'
            ])
        
        with st.spinner("팩터 풀 생성 중..."):
            factor_pool = self.factor_engine.calculate_all_factors(
                universe_data, volume_data, all_factor_types
            )
        
        return factor_pool
    
    def _dynamic_factor_combination(self, factor_pool: Dict[str, pd.DataFrame],
                                  universe_data: pd.DataFrame) -> Dict[str, Any]:
        """동적 팩터 결합 실행"""
        
        # 미래 수익률 계산 (1일 후)
        future_returns = universe_data.pct_change().shift(-1)
        
        # 리밸런싱 날짜 결정
        rebalance_dates = self._get_rebalance_dates(universe_data.index)
        
        mega_alpha_factor = pd.DataFrame(index=universe_data.index, columns=universe_data.columns)
        daily_weights = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, date in enumerate(universe_data.index):
            
            # 리밸런싱 날짜인지 확인
            if date in rebalance_dates or i == 0:
                # 팩터 선택 및 가중치 계산
                selected_factors, weights = self._select_factors_and_weights(
                    factor_pool, future_returns, date
                )
                current_weights = weights.copy()
                daily_weights[date] = current_weights
            else:
                # 이전 가중치 유지
                daily_weights[date] = current_weights
            
            # 메가-알파 팩터 계산
            if current_weights:
                mega_alpha_value = self._calculate_mega_alpha(
                    factor_pool, current_weights, date
                )
                if mega_alpha_value is not None:
                    mega_alpha_factor.loc[date] = mega_alpha_value
            
            progress_bar.progress((i + 1) / len(universe_data.index))
            status_text.text(f"처리 중: {date.strftime('%Y-%m-%d')} ({i+1}/{len(universe_data.index)})")
        
        progress_bar.empty()
        status_text.empty()
        
        # 백테스팅 실행
        backtester = PortfolioBacktester(universe_data, None)
        backtest_results = backtester.run_backtest(
            mega_alpha_factor.fillna(0.5),  # NaN을 중립값으로 채움
            method='rank',
            long_only=self.config.long_only,
            rebalance_freq=self.config.rebalance_frequency,
            transaction_cost_bps=self.config.transaction_cost_bps,
            max_position=self.config.max_position_weight
        )
        
        self.factor_weights_history = daily_weights
        
        return {
            'mega_alpha_factor': mega_alpha_factor,
            'factor_weights_history': daily_weights,
            'backtest_results': backtest_results,
            'factor_pool': factor_pool
        }
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """리밸런싱 날짜 결정"""
        
        if self.config.rebalance_frequency == 'daily':
            return date_index.tolist()
        elif self.config.rebalance_frequency == 'weekly':
            return date_index[date_index.dayofweek == 4].tolist()  # 금요일
        elif self.config.rebalance_frequency == 'monthly':
            return date_index.to_series().resample('M').last().tolist()
        else:
            return date_index.tolist()
    
    def _select_factors_and_weights(self, factor_pool: Dict[str, pd.DataFrame],
                                  future_returns: pd.DataFrame, 
                                  current_date: pd.Timestamp) -> Tuple[List[str], Dict[str, float]]:
        """팩터 선택 및 가중치 계산"""
        
        factor_ics = {}
        
        # 각 팩터의 IC 계산
        for factor_name, factor_data in factor_pool.items():
            # 현재 날짜까지의 데이터로 IC 계산
            end_idx = factor_data.index.get_loc(current_date) if current_date in factor_data.index else -1
            start_idx = max(0, end_idx - self.config.lookback_window)
            
            if start_idx < end_idx:
                factor_window = factor_data.iloc[start_idx:end_idx]
                returns_window = future_returns.iloc[start_idx:end_idx]
                
                mean_ic, _ = self.factor_engine.calculate_ic(factor_window, returns_window)
                
                # IC 임계값 이상인 팩터만 선택
                if abs(mean_ic) >= self.config.ic_threshold:
                    factor_ics[factor_name] = mean_ic
        
        # 상위 팩터 선택
        if not factor_ics:
            return [], {}
        
        # IC 절댓값 기준으로 정렬하여 상위 팩터 선택
        sorted_factors = sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        selected_factors = sorted_factors[:self.config.factor_pool_size]
        
        # 가중치 계산 (IC 크기에 비례)
        total_abs_ic = sum(abs(ic) for _, ic in selected_factors)
        weights = {}
        
        for factor_name, ic in selected_factors:
            raw_weight = abs(ic) / total_abs_ic if total_abs_ic > 0 else 0
            # 가중치 제한 적용
            bounded_weight = np.clip(raw_weight, self.config.min_factor_weight, self.config.max_factor_weight)
            weights[factor_name] = bounded_weight * np.sign(ic)  # IC 부호 보존
        
        # 가중치 정규화
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return list(weights.keys()), weights
    
    def _calculate_mega_alpha(self, factor_pool: Dict[str, pd.DataFrame],
                            weights: Dict[str, float], date: pd.Timestamp) -> Optional[pd.Series]:
        """메가-알파 팩터 값 계산"""
        
        if not weights or date not in factor_pool[list(factor_pool.keys())[0]].index:
            return None
        
        mega_alpha = None
        
        for factor_name, weight in weights.items():
            if factor_name in factor_pool:
                factor_values = factor_pool[factor_name].loc[date]
                
                if mega_alpha is None:
                    mega_alpha = factor_values * weight
                else:
                    # 공통 종목에 대해서만 계산
                    common_stocks = mega_alpha.index.intersection(factor_values.index)
                    mega_alpha.loc[common_stocks] += factor_values.loc[common_stocks] * weight
        
        return mega_alpha
    
    def _analyze_mega_alpha_performance(self, results: Dict[str, Any]):
        """메가-알파 성과 분석 및 시각화"""
        
        st.subheader("📊 메가-알파 시뮬레이션 결과")
        
        backtest_results = results['backtest_results']
        
        if not backtest_results:
            st.error("백테스팅 결과가 없습니다.")
            return
        
        # 1. 누적 수익률 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 누적 수익률 비교**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            backtest_results['cumulative_returns'].plot(ax=ax, label='메가-알파', linewidth=2, color='red')
            backtest_results['benchmark_cumulative'].plot(ax=ax, label='벤치마크', linewidth=2, color='blue', alpha=0.7)
            
            ax.set_title('메가-알파 vs 벤치마크 누적 수익률')
            ax.set_ylabel('누적 수익률')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("**📊 주요 성과 지표**")
            metrics = backtest_results['performance_metrics']
            
            # 성과 지표 표시
            st.metric("총 수익률", f"{metrics['total_return']:.2%}")
            st.metric("연간 수익률", f"{metrics['annualized_return']:.2%}")
            st.metric("샤프 비율", f"{metrics['sharpe_ratio']:.3f}")
            st.metric("최대 손실폭", f"{metrics['max_drawdown']:.2%}")
            st.metric("정보 비율", f"{metrics['information_ratio']:.3f}")
        
        # 2. 팩터 가중치 시계열 분석
        self._plot_factor_weights_over_time(results['factor_weights_history'])
        
        # 3. 일별 팩터 분석 기능
        self._create_daily_factor_analysis(results)
    
    def _plot_factor_weights_over_time(self, weights_history: Dict):
        """팩터 가중치 시계열 시각화"""
        
        st.subheader("🎯 팩터 가중치 변화")
        
        if not weights_history:
            st.warning("팩터 가중치 기록이 없습니다.")
            return
        
        # 가중치 데이터 정리
        weights_df = pd.DataFrame(weights_history).T
        weights_df = weights_df.fillna(0)
        
        if weights_df.empty:
            st.warning("표시할 가중치 데이터가 없습니다.")
            return
        
        # 히트맵으로 시각화
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 날짜 샘플링 (너무 많으면 가독성 저하)
        if len(weights_df) > 50:
            sample_idx = np.linspace(0, len(weights_df)-1, 50, dtype=int)
            weights_sample = weights_df.iloc[sample_idx]
        else:
            weights_sample = weights_df
        
        sns.heatmap(weights_sample.T, cmap='RdBu_r', center=0, 
                   annot=False, fmt='.2f', ax=ax, cbar_kws={'label': '팩터 가중치'})
        
        ax.set_title('시간별 팩터 가중치 변화')
        ax.set_xlabel('날짜')
        ax.set_ylabel('팩터')
        
        # x축 레이블 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # 평균 팩터 가중치 표시
        st.markdown("**📊 평균 팩터 가중치**")
        avg_weights = weights_df.abs().mean().sort_values(ascending=False)
        avg_weights_df = pd.DataFrame({
            '팩터': avg_weights.index,
            '평균 가중치': avg_weights.values,
            '평균 가중치 (%)': avg_weights.values * 100
        })
        
        st.dataframe(avg_weights_df, use_container_width=True)
    
    def _create_daily_factor_analysis(self, results: Dict[str, Any]):
        """일별 팩터 분석 기능"""
        
        st.subheader("🔍 일별 팩터 분석")
        
        weights_history = results['factor_weights_history']
        available_dates = list(weights_history.keys())
        
        if not available_dates:
            st.warning("분석할 날짜 데이터가 없습니다.")
            return
        
        # 날짜 선택
        selected_date = st.selectbox(
            "분석할 날짜를 선택하세요:",
            available_dates,
            index=len(available_dates)//2,  # 중간 날짜 기본 선택
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
        
        if selected_date in weights_history:
            daily_weights = weights_history[selected_date]
            
            if daily_weights:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 해당 일자 팩터 구성**")
                    
                    # 가중치 데이터 정리
                    factor_analysis = []
                    for factor_name, weight in daily_weights.items():
                        factor_analysis.append({
                            '팩터명': factor_name,
                            '가중치': f"{weight:.4f}",
                            '가중치 (%)': f"{weight*100:.2f}%",
                            '방향': '매수' if weight > 0 else '매도'
                        })
                    
                    factor_df = pd.DataFrame(factor_analysis)
                    st.dataframe(factor_df, use_container_width=True)
                
                with col2:
                    st.markdown("**📈 팩터 가중치 차트**")
                    
                    # 파이 차트 생성
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    factors = list(daily_weights.keys())
                    weights = [abs(w) for w in daily_weights.values()]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(factors)))
                    
                    wedges, texts, autotexts = ax.pie(weights, labels=factors, autopct='%1.1f%%',
                                                    colors=colors, startangle=90)
                    
                    ax.set_title(f'{selected_date.strftime("%Y-%m-%d")} 팩터 구성')
                    
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("해당 날짜에는 활성 팩터가 없습니다.")
    
    def save_mega_alpha_results(self, results: Dict[str, Any], name_suffix: str = ""):
        """메가-알파 결과를 팩터 Zoo에 저장"""
        
        if not results or 'mega_alpha_factor' not in results:
            st.error("저장할 결과가 없습니다.")
            return
        
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        factor_name = f"{now_str}_mega_alpha{name_suffix}"
        
        # 메타데이터 생성
        meta_data = {
            'type': 'mega_alpha',
            'created_at': datetime.now().isoformat(),
            'config': {
                'factor_pool_size': self.config.factor_pool_size,
                'rebalance_frequency': self.config.rebalance_frequency,
                'lookback_window': self.config.lookback_window,
                'ic_threshold': self.config.ic_threshold
            },
            'performance': results['backtest_results']['performance_metrics'] if results['backtest_results'] else {},
            'factor_count': len(results['factor_pool'])
        }
        
        factor_data = {
            'meta': meta_data,
            'factor': results['mega_alpha_factor'],
            'weights_history': results['factor_weights_history'],
            'backtest_results': results['backtest_results']
        }
        
        save_factor_to_zoo(factor_name, factor_data)
        st.success(f"메가-알파 결과가 팩터 Zoo에 저장되었습니다: {factor_name}")