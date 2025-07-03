"""
올바른 알파 팩터 생성 클래스
Proper Alpha Factor Generation for Quantitative Investment
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import FactorConfig

class AlphaFactorEngine:
    """올바른 알파 팩터 생성 엔진"""
    
    def __init__(self, config: FactorConfig):
        self.config = config
        self.factors_cache = {}
        self.ic_cache = {}
    
    def calculate_momentum_factor(self, universe_data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """모멘텀 팩터: 과거 수익률 기반 횡단면 순위
        
        Args:
            universe_data: 종목별 가격 데이터 (날짜 x 종목)
            lookback: 과거 기간 (일)
            
        Returns:
            횡단면 순위 팩터 (0~1 백분위수)
        """
        returns = universe_data.pct_change(lookback)
        factor_scores = returns.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_reversal_factor(self, universe_data: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """단기 반전 팩터: 최근 하락 종목에 높은 점수"""
        short_returns = universe_data.pct_change(lookback)
        factor_scores = (-short_returns).rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_volatility_factor(self, universe_data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """변동성 팩터: 저변동성 종목에 높은 점수"""
        returns = universe_data.pct_change()
        volatility = returns.rolling(window=lookback).std()
        factor_scores = (-volatility).rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_volume_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, 
                              lookback: int = 20) -> pd.DataFrame:
        """거래량 팩터: 비정상적 거래량 증가 종목에 높은 점수"""
        avg_volume = volume_data.rolling(window=lookback).mean()
        volume_ratio = volume_data / avg_volume
        factor_scores = volume_ratio.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_rsi_factor(self, universe_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI 기반 기술적 팩터"""
        rsi_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > period:
                rsi_values = self._calculate_rsi(prices, period)
                rsi_data[ticker] = rsi_values
        
        if not rsi_data:
            return pd.DataFrame()
        
        rsi_df = pd.DataFrame(rsi_data)
        # RSI 30 이하는 과매도(높은 점수), 70 이상은 과매수(낮은 점수)
        adjusted_rsi = 100 - rsi_df
        factor_scores = adjusted_rsi.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_price_to_ma_factor(self, universe_data: pd.DataFrame, ma_period: int = 50) -> pd.DataFrame:
        """이동평균 대비 가격 팩터"""
        ma_data = universe_data.rolling(window=ma_period).mean()
        price_to_ma = universe_data / ma_data - 1
        factor_scores = price_to_ma.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 0으로 나누는 것 방지
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # NaN은 중립값 50으로 설정
    
    def calculate_all_factors(self, universe_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None,
                            factor_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """모든 팩터 계산"""
        
        if factor_types is None:
            factor_types = ['momentum', 'reversal', 'volatility', 'rsi', 'price_to_ma']
        
        factors_dict = {}
        
        try:
            if 'momentum' in factor_types:
                momentum = self.calculate_momentum_factor(universe_data, lookback=self.config.momentum_lookback)
                if not momentum.empty:
                    factors_dict['momentum'] = momentum
                    
            if 'reversal' in factor_types:
                reversal = self.calculate_reversal_factor(universe_data, lookback=self.config.reversal_lookback)
                if not reversal.empty:
                    factors_dict['reversal'] = reversal
                    
            if 'volatility' in factor_types:
                volatility = self.calculate_volatility_factor(universe_data, lookback=self.config.volatility_lookback)
                if not volatility.empty:
                    factors_dict['volatility'] = volatility
                    
            if 'volume' in factor_types and volume_data is not None:
                volume = self.calculate_volume_factor(universe_data, volume_data, lookback=self.config.volume_lookback)
                if not volume.empty:
                    factors_dict['volume'] = volume
                    
            if 'rsi' in factor_types:
                rsi = self.calculate_rsi_factor(universe_data, period=self.config.rsi_period)
                if not rsi.empty:
                    factors_dict['rsi'] = rsi
                    
            if 'price_to_ma' in factor_types:
                price_to_ma = self.calculate_price_to_ma_factor(universe_data, ma_period=self.config.ma_period)
                if not price_to_ma.empty:
                    factors_dict['price_to_ma'] = price_to_ma
                    
        except Exception as e:
            st.error(f"팩터 계산 중 오류: {e}")
            
        return factors_dict
    
    def calculate_ic(self, factor_data: pd.DataFrame, future_returns: pd.DataFrame, 
                    lookback: int = 60) -> Tuple[float, List[float]]:
        """Information Coefficient (IC) 계산"""
        
        ic_values = []
        
        for i in range(min(lookback, len(factor_data) - 1)):
            date_idx = -(i + 2)
            
            if abs(date_idx) > len(factor_data):
                continue
                
            try:
                factor_values = factor_data.iloc[date_idx]
                future_ret = future_returns.iloc[date_idx + 1]
                
                # 공통 종목에 대해서만 IC 계산
                common_tickers = factor_values.index.intersection(future_ret.index)
                if len(common_tickers) < 3:
                    continue
                
                factor_vals = factor_values[common_tickers].dropna()
                future_vals = future_ret[common_tickers].dropna()
                
                # 다시 교집합
                final_common = factor_vals.index.intersection(future_vals.index)
                if len(final_common) < 3:
                    continue
                
                factor_vals = factor_vals[final_common]
                future_vals = future_vals[final_common]
                
                # IC 계산 (스피어만 상관계수 사용)
                ic = factor_vals.corr(future_vals, method='spearman')
                if not np.isnan(ic):
                    ic_values.append(ic)
                    
            except Exception as e:
                continue
        
        if ic_values:
            mean_ic = np.mean(ic_values)
            return mean_ic, ic_values
        else:
            return 0.0, []
    
    def combine_factors_ic_weighted(self, factors_dict: Dict[str, pd.DataFrame], 
                                   future_returns: pd.DataFrame, 
                                   lookback: int = 60) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """IC 기반 팩터 가중 결합"""
        
        ic_weights = {}
        ic_details = {}
        
        # 각 팩터의 IC 계산
        for factor_name, factor_data in factors_dict.items():
            mean_ic, ic_values = self.calculate_ic(factor_data, future_returns, lookback)
            ic_weights[factor_name] = mean_ic
            ic_details[factor_name] = {
                'mean_ic': mean_ic,
                'ic_std': np.std(ic_values) if ic_values else 0,
                'ic_count': len(ic_values)
            }
        
        # 가중치 정규화
        total_abs_weight = sum(abs(w) for w in ic_weights.values())
        if total_abs_weight > 0:
            ic_weights = {k: v / total_abs_weight for k, v in ic_weights.items()}
        else:
            # 모든 IC가 0이면 균등 가중
            ic_weights = {k: 1/len(ic_weights) for k in ic_weights}
        
        # 가중 결합
        combined_factor = None
        
        for factor_name, weight in ic_weights.items():
            if factor_name in factors_dict and weight != 0:
                factor_data = factors_dict[factor_name]
                
                if combined_factor is None:
                    combined_factor = factor_data * weight
                else:
                    # 공통 인덱스와 컬럼으로 맞춤
                    common_index = combined_factor.index.intersection(factor_data.index)
                    common_columns = combined_factor.columns.intersection(factor_data.columns)
                    
                    if len(common_index) > 0 and len(common_columns) > 0:
                        combined_factor.loc[common_index, common_columns] += \
                            factor_data.loc[common_index, common_columns] * weight
        
        if combined_factor is None:
            # 빈 DataFrame 반환
            combined_factor = pd.DataFrame()
        
        return combined_factor, ic_weights
    
    def convert_to_qlib_format(self, factor_df: pd.DataFrame) -> pd.Series:
        """횡단면 팩터를 Qlib MultiIndex 형식으로 변환"""
        
        if factor_df.empty:
            return pd.Series(dtype=float, name='alpha_factor')
        
        data_list = []
        
        for date in factor_df.index:
            for ticker in factor_df.columns:
                value = factor_df.loc[date, ticker]
                if not np.isnan(value):
                    data_list.append((pd.to_datetime(date), ticker, float(value)))
        
        if not data_list:
            return pd.Series(dtype=float, name='alpha_factor')
        
        # MultiIndex 생성
        index_tuples = [(row[0], row[1]) for row in data_list]
        values = [row[2] for row in data_list]
        
        multi_index = pd.MultiIndex.from_tuples(
            index_tuples, names=['datetime', 'instrument']
        )
        
        return pd.Series(values, index=multi_index, name='alpha_factor')
    
    def analyze_factor_performance(self, factor_df: pd.DataFrame, 
                                 future_returns: pd.DataFrame) -> Dict[str, float]:
        """팩터 성능 분석"""
        
        if factor_df.empty:
            return {}
        
        # IC 계산
        mean_ic, ic_values = self.calculate_ic(factor_df, future_returns)
        
        # IC의 t-통계량 (ICIR)
        ic_std = np.std(ic_values) if ic_values else 0
        icir = mean_ic / ic_std if ic_std > 0 else 0
        
        # 팩터 값의 분산 (종목 간 차별화 정도)
        factor_spread = factor_df.std(axis=1).mean()
        
        # 팩터 턴오버 (안정성)
        factor_turnover = 0
        if len(factor_df) > 1:
            rank_corr = []
            for i in range(1, min(len(factor_df), 21)):  # 최대 20일
                try:
                    prev_rank = factor_df.iloc[-i-1].rank()
                    curr_rank = factor_df.iloc[-i].rank()
                    common_stocks = prev_rank.index.intersection(curr_rank.index)
                    if len(common_stocks) > 3:
                        corr = prev_rank[common_stocks].corr(curr_rank[common_stocks])
                        if not np.isnan(corr):
                            rank_corr.append(corr)
                except:
                    continue
            
            if rank_corr:
                factor_turnover = 1 - np.mean(rank_corr)
        
        return {
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'icir': icir,
            'ic_count': len(ic_values),
            'factor_spread': factor_spread,
            'factor_turnover': factor_turnover
        }