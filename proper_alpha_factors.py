"""
올바른 알파 팩터 생성 방법론 예시
Real alpha factor generation methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ProperAlphaFactors:
    """올바른 알파 팩터 생성 클래스"""
    
    def __init__(self):
        self.factors = {}
    
    def calculate_momentum_factor(self, universe_data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """모멘텀 팩터: 과거 수익률 기반 횡단면 순위"""
        returns = universe_data.pct_change(lookback)
        # 각 날짜별로 종목들을 순위화 (0~1 사이 백분위수)
        factor_scores = returns.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)  # 중립값으로 채움
    
    def calculate_reversal_factor(self, universe_data: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """단기 반전 팩터: 최근 하락 종목에 높은 점수"""
        short_returns = universe_data.pct_change(lookback)
        # 반전 효과: 음의 수익률을 양의 점수로 변환
        factor_scores = (-short_returns).rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_volatility_factor(self, universe_data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """변동성 팩터: 저변동성 종목에 높은 점수"""
        returns = universe_data.pct_change()
        volatility = returns.rolling(window=lookback).std()
        # 낮은 변동성이 높은 점수
        factor_scores = (-volatility).rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_volume_factor(self, universe_data: pd.DataFrame, volume_data: pd.DataFrame, 
                              lookback: int = 20) -> pd.DataFrame:
        """거래량 팩터: 비정상적 거래량 증가"""
        avg_volume = volume_data.rolling(window=lookback).mean()
        volume_ratio = volume_data / avg_volume
        factor_scores = volume_ratio.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def calculate_technical_factor(self, universe_data: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
        """기술적 팩터: RSI 기반"""
        rsi_data = {}
        
        for ticker in universe_data.columns:
            prices = universe_data[ticker].dropna()
            rsi_values = self._calculate_rsi(prices, rsi_period)
            rsi_data[ticker] = rsi_values
        
        rsi_df = pd.DataFrame(rsi_data)
        # RSI 30 이하는 과매도(높은 점수), 70 이상은 과매수(낮은 점수)
        adjusted_rsi = 100 - rsi_df  # RSI 반전
        factor_scores = adjusted_rsi.rank(axis=1, pct=True, method='dense')
        return factor_scores.fillna(0.5)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def combine_factors_ic_weighted(self, factors_dict: Dict[str, pd.DataFrame], 
                                   future_returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """IC(Information Coefficient) 기반 팩터 가중 결합"""
        
        ic_weights = {}
        
        for factor_name, factor_data in factors_dict.items():
            ic_values = []
            
            # 최근 lookback 기간의 IC 계산
            for i in range(min(lookback, len(factor_data) - 1)):
                date_idx = -(i + 2)  # 미래 수익률과 매칭
                
                if date_idx < -len(factor_data):
                    continue
                    
                factor_values = factor_data.iloc[date_idx]
                future_ret = future_returns.iloc[date_idx + 1]
                
                # 공통 종목에 대해서만 IC 계산
                common_tickers = factor_values.index.intersection(future_ret.index)
                if len(common_tickers) < 5:  # 최소 5개 종목 필요
                    continue
                
                factor_vals = factor_values[common_tickers]
                future_vals = future_ret[common_tickers]
                
                # IC 계산 (상관계수)
                ic = factor_vals.corr(future_vals)
                if not np.isnan(ic):
                    ic_values.append(ic)
            
            # IC의 평균을 가중치로 사용
            if ic_values:
                ic_weights[factor_name] = np.mean(ic_values)
            else:
                ic_weights[factor_name] = 0
        
        # 가중치 정규화
        total_abs_weight = sum(abs(w) for w in ic_weights.values())
        if total_abs_weight > 0:
            ic_weights = {k: v / total_abs_weight for k, v in ic_weights.items()}
        else:
            # 모든 IC가 0이면 균등 가중
            ic_weights = {k: 1/len(ic_weights) for k in ic_weights}
        
        print(f"IC 기반 가중치: {ic_weights}")
        
        # 가중 결합
        combined_factor = pd.DataFrame(0, index=factor_data.index, columns=factor_data.columns)
        
        for factor_name, weight in ic_weights.items():
            if factor_name in factors_dict and weight != 0:
                combined_factor += factors_dict[factor_name] * weight
        
        return combined_factor
    
    def convert_to_qlib_format(self, factor_df: pd.DataFrame) -> pd.Series:
        """횡단면 팩터를 Qlib MultiIndex 형식으로 변환"""
        data_list = []
        
        for date in factor_df.index:
            for ticker in factor_df.columns:
                value = factor_df.loc[date, ticker]
                if not np.isnan(value):
                    data_list.append((pd.to_datetime(date), ticker, value))
        
        if not data_list:
            return pd.Series(dtype=float)
        
        # MultiIndex 생성
        index_tuples = [(row[0], row[1]) for row in data_list]
        values = [row[2] for row in data_list]
        
        multi_index = pd.MultiIndex.from_tuples(
            index_tuples, names=['datetime', 'instrument']
        )
        
        return pd.Series(values, index=multi_index, name='alpha_factor')

# 사용 예시
def demonstrate_proper_alpha_generation():
    """올바른 알파 팩터 생성 시연"""
    
    # 가상의 유니버스 데이터 생성 (실제로는 여러 종목 데이터)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    
    np.random.seed(42)
    universe_data = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # 거래량 데이터 (가상)
    volume_data = pd.DataFrame(
        np.random.exponential(1000000, (len(dates), len(tickers))),
        index=dates,
        columns=tickers
    )
    
    # 미래 수익률 (1일 후)
    future_returns = universe_data.pct_change().shift(-1)
    
    # 알파 팩터 생성기 초기화
    alpha_gen = ProperAlphaFactors()
    
    # 개별 팩터들 계산
    momentum_factor = alpha_gen.calculate_momentum_factor(universe_data, lookback=20)
    reversal_factor = alpha_gen.calculate_reversal_factor(universe_data, lookback=5)
    volatility_factor = alpha_gen.calculate_volatility_factor(universe_data, lookback=20)
    volume_factor = alpha_gen.calculate_volume_factor(universe_data, volume_data, lookback=20)
    technical_factor = alpha_gen.calculate_technical_factor(universe_data, rsi_period=14)
    
    # 팩터 딕셔너리
    factors_dict = {
        'momentum': momentum_factor,
        'reversal': reversal_factor,
        'volatility': volatility_factor,
        'volume': volume_factor,
        'technical': technical_factor
    }
    
    # IC 기반 가중 결합
    combined_factor = alpha_gen.combine_factors_ic_weighted(
        factors_dict, future_returns, lookback=60
    )
    
    # Qlib 형식으로 변환
    qlib_factor = alpha_gen.convert_to_qlib_format(combined_factor)
    
    print("✅ 올바른 알파 팩터 생성 완료!")
    print(f"팩터 데이터 포인트 수: {len(qlib_factor)}")
    print(f"팩터 값 범위: {qlib_factor.min():.4f} ~ {qlib_factor.max():.4f}")
    print(f"팩터 평균: {qlib_factor.mean():.4f}")
    
    return qlib_factor, combined_factor, factors_dict

if __name__ == "__main__":
    qlib_factor, combined_factor, individual_factors = demonstrate_proper_alpha_generation()