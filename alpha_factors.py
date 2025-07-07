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
    
    def calculate_bollinger_band_factor(self, universe_data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """볼린저 밴드 기반 팩터: 밴드 하단 근처 종목에 높은 점수"""
        bb_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > period:
                bb_values = self._calculate_bollinger_bands(prices, period, std_dev)
                bb_data[ticker] = bb_values
        
        if not bb_data:
            return pd.DataFrame()
        
        bb_df = pd.DataFrame(bb_data)
        # 밴드 하단 근처일수록 높은 점수
        factor_scores = bb_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_macd_factor(self, universe_data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD 기반 팩터: MACD 히스토그램이 양수인 종목에 높은 점수"""
        macd_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > slow:
                macd_values = self._calculate_macd(prices, fast, slow, signal)
                macd_data[ticker] = macd_values
        
        if not macd_data:
            return pd.DataFrame()
        
        macd_df = pd.DataFrame(macd_data)
        factor_scores = macd_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_stochastic_factor(self, universe_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """스토캐스틱 기반 팩터: 과매도 구간 종목에 높은 점수"""
        stoch_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > k_period:
                stoch_values = self._calculate_stochastic(prices, k_period, d_period)
                stoch_data[ticker] = stoch_values
        
        if not stoch_data:
            return pd.DataFrame()
        
        stoch_df = pd.DataFrame(stoch_data)
        # 과매도 구간일수록 높은 점수 (100에서 빼기)
        adjusted_stoch = 100 - stoch_df
        factor_scores = adjusted_stoch.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_williams_r_factor(self, universe_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R 기반 팩터: 과매도 구간 종목에 높은 점수"""
        williams_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > period:
                williams_values = self._calculate_williams_r(prices, period)
                williams_data[ticker] = williams_values
        
        if not williams_data:
            return pd.DataFrame()
        
        williams_df = pd.DataFrame(williams_data)
        # 과매도 구간일수록 높은 점수 (절댓값 사용)
        factor_scores = williams_df.abs().rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_cci_factor(self, universe_data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """CCI (Commodity Channel Index) 기반 팩터: 극단값 구간 종목에 높은 점수"""
        cci_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > period:
                cci_values = self._calculate_cci(prices, period)
                cci_data[ticker] = cci_values
        
        if not cci_data:
            return pd.DataFrame()
        
        cci_df = pd.DataFrame(cci_data)
        # 극단값일수록 높은 점수 (절댓값 사용)
        factor_scores = cci_df.abs().rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_money_flow_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index 기반 팩터: 과매도 구간 종목에 높은 점수"""
        mfi_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    mfi_values = self._calculate_money_flow_index(prices, volumes, period)
                    mfi_data[ticker] = mfi_values
        
        if not mfi_data:
            return pd.DataFrame()
        
        mfi_df = pd.DataFrame(mfi_data)
        # 과매도 구간일수록 높은 점수 (100에서 빼기)
        adjusted_mfi = 100 - mfi_df
        factor_scores = adjusted_mfi.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_aroon_factor(self, universe_data: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """Aroon 지표 기반 팩터: 추세 전환 구간 종목에 높은 점수"""
        aroon_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            if len(prices) > period:
                aroon_values = self._calculate_aroon(prices, period)
                aroon_data[ticker] = aroon_values
        
        if not aroon_data:
            return pd.DataFrame()
        
        aroon_df = pd.DataFrame(aroon_data)
        # Aroon Down이 높을수록 (추세 전환 가능성) 높은 점수
        factor_scores = aroon_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_obv_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """OBV (On-Balance Volume) 기반 팩터: 거래량 추세가 강한 종목에 높은 점수"""
        obv_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    obv_values = self._calculate_obv(prices, volumes, period)
                    obv_data[ticker] = obv_values
        
        if not obv_data:
            return pd.DataFrame()
        
        obv_df = pd.DataFrame(obv_data)
        factor_scores = obv_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_volume_price_trend_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """VPT (Volume Price Trend) 기반 팩터: 가격-거래량 추세가 강한 종목에 높은 점수"""
        vpt_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    vpt_values = self._calculate_vpt(prices, volumes, period)
                    vpt_data[ticker] = vpt_values
        
        if not vpt_data:
            return pd.DataFrame()
        
        vpt_df = pd.DataFrame(vpt_data)
        factor_scores = vpt_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_chaikin_money_flow_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Chaikin Money Flow 기반 팩터: 자금 유입이 강한 종목에 높은 점수"""
        cmf_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    cmf_values = self._calculate_chaikin_money_flow(prices, volumes, period)
                    cmf_data[ticker] = cmf_values
        
        if not cmf_data:
            return pd.DataFrame()
        
        cmf_df = pd.DataFrame(cmf_data)
        factor_scores = cmf_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_force_index_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Force Index 기반 팩터: 가격 변동과 거래량의 곱으로 강도 측정"""
        force_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    force_values = self._calculate_force_index(prices, volumes, period)
                    force_data[ticker] = force_values
        
        if not force_data:
            return pd.DataFrame()
        
        force_df = pd.DataFrame(force_data)
        factor_scores = force_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_ease_of_movement_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Ease of Movement 기반 팩터: 가격 변동의 용이성 측정"""
        eom_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    eom_values = self._calculate_ease_of_movement(prices, volumes, period)
                    eom_data[ticker] = eom_values
        
        if not eom_data:
            return pd.DataFrame()
        
        eom_df = pd.DataFrame(eom_data)
        factor_scores = eom_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_accumulation_distribution_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Accumulation/Distribution 기반 팩터: 자금 유입/유출 패턴 측정"""
        ad_data = {}
        
        for ticker in universe_data.columns:
            if ticker in volume_data.columns:
                prices = universe_data[ticker].dropna()
                volumes = volume_data[ticker].dropna()
                if len(prices) > period and len(volumes) > period:
                    ad_values = self._calculate_accumulation_distribution(prices, volumes, period)
                    ad_data[ticker] = ad_values
        
        if not ad_data:
            return pd.DataFrame()
        
        ad_df = pd.DataFrame(ad_data)
        factor_scores = ad_df.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_all_factors(self, universe_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None,
                            factor_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """모든 팩터 계산"""
        
        if factor_types is None:
            factor_types = ['momentum', 'reversal', 'volatility', 'rsi', 'price_to_ma', 'bollinger_band', 'macd', 'stochastic', 'williams_r', 'cci', 'money_flow', 'aroon', 'obv', 'volume_price_trend', 'chaikin_money_flow', 'force_index', 'ease_of_movement', 'accumulation_distribution']
        
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
                    
            if 'bollinger_band' in factor_types:
                bollinger_band = self.calculate_bollinger_band_factor(universe_data, period=self.config.bollinger_band_period, std_dev=self.config.bollinger_band_std_dev)
                if not bollinger_band.empty:
                    factors_dict['bollinger_band'] = bollinger_band
                    
            if 'macd' in factor_types:
                macd = self.calculate_macd_factor(universe_data, fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal)
                if not macd.empty:
                    factors_dict['macd'] = macd
                    
            if 'stochastic' in factor_types:
                stochastic = self.calculate_stochastic_factor(universe_data, k_period=self.config.stochastic_k_period, d_period=self.config.stochastic_d_period)
                if not stochastic.empty:
                    factors_dict['stochastic'] = stochastic
                    
            if 'williams_r' in factor_types:
                williams_r = self.calculate_williams_r_factor(universe_data, period=self.config.williams_r_period)
                if not williams_r.empty:
                    factors_dict['williams_r'] = williams_r
                    
            if 'cci' in factor_types:
                cci = self.calculate_cci_factor(universe_data, period=self.config.cci_period)
                if not cci.empty:
                    factors_dict['cci'] = cci
                    
            if 'money_flow' in factor_types:
                money_flow = self.calculate_money_flow_factor(universe_data, volume_data, period=self.config.money_flow_period)
                if not money_flow.empty:
                    factors_dict['money_flow'] = money_flow
                    
            if 'aroon' in factor_types:
                aroon = self.calculate_aroon_factor(universe_data, period=self.config.aroon_period)
                if not aroon.empty:
                    factors_dict['aroon'] = aroon
                    
            if 'obv' in factor_types:
                obv = self.calculate_obv_factor(universe_data, volume_data, period=self.config.obv_period)
                if not obv.empty:
                    factors_dict['obv'] = obv
                    
            if 'volume_price_trend' in factor_types:
                volume_price_trend = self.calculate_volume_price_trend_factor(universe_data, volume_data, period=self.config.volume_price_trend_period)
                if not volume_price_trend.empty:
                    factors_dict['volume_price_trend'] = volume_price_trend
                    
            if 'chaikin_money_flow' in factor_types:
                chaikin_money_flow = self.calculate_chaikin_money_flow_factor(universe_data, volume_data, period=self.config.chaikin_money_flow_period)
                if not chaikin_money_flow.empty:
                    factors_dict['chaikin_money_flow'] = chaikin_money_flow
                    
            if 'force_index' in factor_types:
                force_index = self.calculate_force_index_factor(universe_data, volume_data, period=self.config.force_index_period)
                if not force_index.empty:
                    factors_dict['force_index'] = force_index
                    
            if 'ease_of_movement' in factor_types:
                ease_of_movement = self.calculate_ease_of_movement_factor(universe_data, volume_data, period=self.config.ease_of_movement_period)
                if not ease_of_movement.empty:
                    factors_dict['ease_of_movement'] = ease_of_movement
                    
            if 'accumulation_distribution' in factor_types:
                accumulation_distribution = self.calculate_accumulation_distribution_factor(universe_data, volume_data, period=self.config.accumulation_distribution_period)
                if not accumulation_distribution.empty:
                    factors_dict['accumulation_distribution'] = accumulation_distribution
                    
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
    
    def combine_factors_fixed_weights(self, factors_dict: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        팩터별 고정 가중치로 결합하는 함수입니다.
        - factors_dict: {팩터명: 팩터 DataFrame}
        - weights: {팩터명: 가중치(float)}
        반환: (결합된 팩터 DataFrame, 실제 적용된 가중치 dict)
        """
        # 가중치 합이 0이 아니면 정규화, 0이면 균등 가중치
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            norm_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            n = len(weights)
            norm_weights = {k: 1/n for k in weights}

        combined_factor = None
        for factor_name, weight in norm_weights.items():
            if factor_name in factors_dict and weight != 0:
                factor_data = factors_dict[factor_name]
                if combined_factor is None:
                    combined_factor = factor_data * weight
                else:
                    # 인덱스/컬럼 교집합만 결합
                    common_index = combined_factor.index.intersection(factor_data.index)
                    common_columns = combined_factor.columns.intersection(factor_data.columns)
                    if len(common_index) > 0 and len(common_columns) > 0:
                        combined_factor.loc[common_index, common_columns] += \
                            factor_data.loc[common_index, common_columns] * weight
        if combined_factor is None:
            combined_factor = pd.DataFrame()
        return combined_factor, norm_weights
    
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
    
    def calculate_rolling_ic(self, factor_df: pd.DataFrame, future_returns: pd.DataFrame, window: int = 20) -> dict:
        """
        팩터별 rolling IC 및 ICIR(rolling window 기반)을 계산합니다.
        - factor_df: 팩터 DataFrame (index: 날짜, columns: 종목)
        - future_returns: 미래 수익률 DataFrame (index: 날짜, columns: 종목)
        - window: rolling window 크기(일)
        반환: {'ic': IC 시계열, 'icir': ICIR 시계열}
        """
        ic_series = []
        icir_series = []
        dates = []
        ic_buffer = []
        for i in range(window, len(factor_df) - 1):
            # 윈도우 내 팩터/수익률 추출
            factor_window = factor_df.iloc[i-window:i]
            returns_window = future_returns.iloc[i-window+1:i+1]
            # 각 날짜별 IC 계산
            daily_ics = []
            for j in range(window):
                try:
                    f = factor_window.iloc[j]
                    r = returns_window.iloc[j]
                    common = f.index.intersection(r.index)
                    if len(common) < 3:
                        continue
                    fvals = f[common].dropna()
                    rvals = r[common].dropna()
                    final_common = fvals.index.intersection(rvals.index)
                    if len(final_common) < 3:
                        continue
                    ic = fvals[final_common].corr(rvals[final_common], method='spearman')
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                except Exception:
                    continue
            # 윈도우 마지막 날짜 기준
            dates.append(factor_df.index[i])
            if daily_ics:
                mean_ic = np.mean(daily_ics)
                ic_series.append(mean_ic)
                ic_buffer.append(mean_ic)
                # rolling ICIR: 윈도우 내 IC 평균 / 표준편차
                if len(ic_buffer) >= window:
                    icir = np.mean(ic_buffer[-window:]) / (np.std(ic_buffer[-window:]) + 1e-8)
                else:
                    icir = np.mean(ic_buffer) / (np.std(ic_buffer) + 1e-8)
                icir_series.append(icir)
            else:
                ic_series.append(np.nan)
                icir_series.append(np.nan)
        return {'dates': dates, 'ic': ic_series, 'icir': icir_series}

    # 기술적 지표 계산 헬퍼 함수들
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """볼린저 밴드 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        lower_band = sma - (std * std_dev)
        # 밴드 하단 대비 현재 가격의 위치 (0~1)
        bb_position = (prices - lower_band) / (sma + (std * std_dev) - lower_band)
        return bb_position.fillna(0.5)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return histogram.fillna(0)
    
    def _calculate_stochastic(self, prices: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.Series:
        """스토캐스틱 계산"""
        low_min = prices.rolling(window=k_period).min()
        high_max = prices.rolling(window=k_period).max()
        k_percent = 100 * ((prices - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return d_percent.fillna(50)
    
    def _calculate_williams_r(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R 계산"""
        low_min = prices.rolling(window=period).min()
        high_max = prices.rolling(window=period).max()
        williams_r = -100 * ((high_max - prices) / (high_max - low_min))
        return williams_r.fillna(-50)
    
    def _calculate_cci(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """CCI 계산"""
        sma = prices.rolling(window=period).mean()
        mean_deviation = prices.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (prices - sma) / (0.015 * mean_deviation)
        return cci.fillna(0)
    
    def _calculate_money_flow_index(self, prices: pd.Series, volumes: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index 계산"""
        typical_price = prices  # OHLC가 없으므로 Close 가격 사용
        money_flow = typical_price * volumes
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi.fillna(50)
    
    def _calculate_aroon(self, prices: pd.Series, period: int = 25) -> pd.Series:
        """Aroon 지표 계산"""
        high_period = prices.rolling(window=period).apply(lambda x: x.argmax())
        low_period = prices.rolling(window=period).apply(lambda x: x.argmin())
        
        aroon_up = 100 * (period - high_period) / period
        aroon_down = 100 * (period - low_period) / period
        
        # Aroon Down 반환 (추세 전환 신호)
        return aroon_down.fillna(50)
    
    def _calculate_obv(self, prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
        """OBV 계산"""
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volumes.iloc[0]
        
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # OBV 변화율
        obv_change = obv.pct_change(period)
        return obv_change.fillna(0)
    
    def _calculate_vpt(self, prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
        """VPT 계산"""
        price_change = prices.pct_change()
        vpt = (price_change * volumes).cumsum()
        vpt_change = vpt.pct_change(period)
        return vpt_change.fillna(0)
    
    def _calculate_chaikin_money_flow(self, prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow 계산"""
        # Close 가격만 있으므로 단순화된 계산
        money_flow_multiplier = ((prices - prices.rolling(window=period).min()) / 
                                (prices.rolling(window=period).max() - prices.rolling(window=period).min()))
        money_flow_volume = money_flow_multiplier * volumes
        cmf = money_flow_volume.rolling(window=period).sum() / volumes.rolling(window=period).sum()
        return cmf.fillna(0)
    
    def _calculate_force_index(self, prices: pd.Series, volumes: pd.Series, period: int = 13) -> pd.Series:
        """Force Index 계산"""
        price_change = prices.diff()
        force_index = price_change * volumes
        smoothed_force = force_index.rolling(window=period).mean()
        return smoothed_force.fillna(0)
    
    def _calculate_ease_of_movement(self, prices: pd.Series, volumes: pd.Series, period: int = 14) -> pd.Series:
        """Ease of Movement 계산"""
        price_change = prices.diff()
        box_ratio = price_change / volumes
        eom = box_ratio.rolling(window=period).mean()
        return eom.fillna(0)
    
    def _calculate_accumulation_distribution(self, prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
        """Accumulation/Distribution 계산"""
        # Close 가격만 있으므로 단순화된 계산
        ad_line = (prices * volumes).cumsum()
        ad_change = ad_line.pct_change(period)
        return ad_change.fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 0으로 나누는 것 방지
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # NaN은 중립값 50으로 설정
