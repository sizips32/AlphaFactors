"""
공식 기반 팩터 생성 파이프라인
Formula-based Factor Generation Pipeline
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import re
import ast
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FormulaParser:
    """수학 공식 파서"""
    
    def __init__(self):
        self.supported_functions = {
            'abs': abs,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'max': np.maximum,
            'min': np.minimum,
            'mean': np.mean,
            'std': np.std,
            'sum': np.sum,
            'rank': lambda x: pd.Series(x).rank(pct=True).values,
            'rolling_mean': lambda x, window: pd.Series(x).rolling(window=window).mean().values,
            'rolling_std': lambda x, window: pd.Series(x).rolling(window=window).std().values,
            'rolling_max': lambda x, window: pd.Series(x).rolling(window=window).max().values,
            'rolling_min': lambda x, window: pd.Series(x).rolling(window=window).min().values,
            'pct_change': lambda x, periods=1: pd.Series(x).pct_change(periods=periods).values,
            'diff': lambda x, periods=1: pd.Series(x).diff(periods=periods).values,
            'shift': lambda x, periods=1: pd.Series(x).shift(periods=periods).values,
            'ewm_mean': lambda x, span: pd.Series(x).ewm(span=span).mean().values,
            'ewm_std': lambda x, span: pd.Series(x).ewm(span=span).std().values,
            'zscore': lambda x: (pd.Series(x) - pd.Series(x).mean()) / pd.Series(x).std(),
            'normalize': lambda x: (pd.Series(x) - pd.Series(x).min()) / (pd.Series(x).max() - pd.Series(x).min()),
            'winsorize': lambda x, limits=(0.05, 0.05): self._winsorize(x, limits),
            'momentum': lambda x, period: pd.Series(x).pct_change(period).values,
            'volatility': lambda x, period: pd.Series(x).pct_change().rolling(period).std().values,
            'rsi': lambda x, period: self._calculate_rsi(x, period),
            'macd': lambda x, fast=12, slow=26, signal=9: self._calculate_macd(x, fast, slow, signal),
            'bollinger_bands': lambda x, period=20, std_dev=2: self._calculate_bollinger_bands(x, period, std_dev),
            'stochastic': lambda x, k_period=14, d_period=3: self._calculate_stochastic(x, k_period, d_period),
            'williams_r': lambda x, period=14: self._calculate_williams_r(x, period),
            'cci': lambda x, period=20: self._calculate_cci(x, period)
        }
        
        self.supported_operators = ['+', '-', '*', '/', '**', '>', '<', '>=', '<=', '==', '!=', '&', '|']
        
    def _winsorize(self, x: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """윈저라이제이션"""
        lower_limit = np.percentile(x, limits[0] * 100)
        upper_limit = np.percentile(x, (1 - limits[1]) * 100)
        return np.clip(x, lower_limit, upper_limit)
    
    def _calculate_rsi(self, x: np.ndarray, period: int) -> np.ndarray:
        """RSI 계산"""
        series = pd.Series(x)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_macd(self, x: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
        """MACD 계산"""
        series = pd.Series(x)
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return histogram.fillna(0).values
    
    def _calculate_bollinger_bands(self, x: np.ndarray, period: int, std_dev: float) -> np.ndarray:
        """볼린저 밴드 계산"""
        series = pd.Series(x)
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        lower_band = sma - (std * std_dev)
        bb_position = (series - lower_band) / (sma + (std * std_dev) - lower_band)
        return bb_position.fillna(0.5).values
    
    def _calculate_stochastic(self, x: np.ndarray, k_period: int, d_period: int) -> np.ndarray:
        """스토캐스틱 계산"""
        series = pd.Series(x)
        low_min = series.rolling(window=k_period).min()
        high_max = series.rolling(window=k_period).max()
        k_percent = 100 * ((series - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return d_percent.fillna(50).values
    
    def _calculate_williams_r(self, x: np.ndarray, period: int) -> np.ndarray:
        """Williams %R 계산"""
        series = pd.Series(x)
        low_min = series.rolling(window=period).min()
        high_max = series.rolling(window=period).max()
        williams_r = -100 * ((high_max - series) / (high_max - low_min))
        return williams_r.fillna(-50).values
    
    def _calculate_cci(self, x: np.ndarray, period: int) -> np.ndarray:
        """CCI 계산"""
        series = pd.Series(x)
        sma = series.rolling(window=period).mean()
        mean_deviation = series.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (series - sma) / (0.015 * mean_deviation)
        return cci.fillna(0).values
    
    def validate_formula(self, formula: str) -> Tuple[bool, str]:
        """공식 유효성 검증"""
        try:
            # 기본 문법 검사
            ast.parse(formula)
            
            # 지원되지 않는 함수나 연산자 검사
            for func in re.findall(r'\b\w+\s*\(', formula):
                func_name = func.split('(')[0].strip()
                if func_name not in self.supported_functions and not func_name.isdigit():
                    return False, f"지원되지 않는 함수: {func_name}"
            
            # 위험한 함수 검사
            dangerous_functions = ['eval', 'exec', 'open', 'file', 'input', 'raw_input']
            for dangerous in dangerous_functions:
                if dangerous in formula.lower():
                    return False, f"보안상 위험한 함수 사용 금지: {dangerous}"
            
            return True, "공식이 유효합니다."
            
        except SyntaxError as e:
            return False, f"문법 오류: {str(e)}"
        except Exception as e:
            return False, f"검증 오류: {str(e)}"
    
    def get_available_functions(self) -> Dict[str, str]:
        """사용 가능한 함수 목록 반환"""
        function_descriptions = {
            'abs(x)': '절댓값',
            'sqrt(x)': '제곱근',
            'log(x)': '자연로그',
            'log10(x)': '상용로그',
            'exp(x)': '지수함수',
            'sin(x), cos(x), tan(x)': '삼각함수',
            'max(x, y), min(x, y)': '최대값, 최소값',
            'mean(x)': '평균',
            'std(x)': '표준편차',
            'sum(x)': '합계',
            'rank(x)': '순위 (백분위)',
            'rolling_mean(x, window)': '이동평균',
            'rolling_std(x, window)': '이동표준편차',
            'rolling_max(x, window)': '이동최대값',
            'rolling_min(x, window)': '이동최소값',
            'pct_change(x, periods)': '변화율',
            'diff(x, periods)': '차분',
            'shift(x, periods)': '시프트',
            'ewm_mean(x, span)': '지수가중이동평균',
            'ewm_std(x, span)': '지수가중이동표준편차',
            'zscore(x)': 'Z-점수 정규화',
            'normalize(x)': '0-1 정규화',
            'winsorize(x, limits)': '윈저라이제이션',
            'momentum(x, period)': '모멘텀',
            'volatility(x, period)': '변동성',
            'rsi(x, period)': 'RSI',
            'macd(x, fast, slow, signal)': 'MACD',
            'bollinger_bands(x, period, std_dev)': '볼린저 밴드',
            'stochastic(x, k_period, d_period)': '스토캐스틱',
            'williams_r(x, period)': 'Williams %R',
            'cci(x, period)': 'CCI'
        }
        return function_descriptions

class FormulaFactorGenerator:
    """공식 기반 팩터 생성기"""
    
    def __init__(self):
        self.parser = FormulaParser()
        self.factor_cache = {}
        
    def create_factor_from_formula(self, 
                                 formula: str, 
                                 universe_data: pd.DataFrame,
                                 volume_data: Optional[pd.DataFrame] = None,
                                 params: Dict[str, Any] = None) -> pd.DataFrame:
        """공식으로부터 팩터 생성"""
        
        if params is None:
            params = {}
        
        # 공식 유효성 검증
        is_valid, message = self.parser.validate_formula(formula)
        if not is_valid:
            raise ValueError(f"공식 검증 실패: {message}")
        
        # 팩터 계산
        factor_data = {}
        
        for ticker in universe_data.columns:
            try:
                # 기본 데이터 준비
                price_data = universe_data[ticker].dropna()
                
                if len(price_data) == 0:
                    continue
                
                # 공식 실행을 위한 환경 설정
                local_env = {
                    'price': price_data.values,
                    'close': price_data.values,
                    'returns': price_data.pct_change().values,
                    'log_returns': np.log(price_data / price_data.shift(1)).values,
                    'high': price_data.values,  # OHLC가 없으므로 close 사용
                    'low': price_data.values,
                    'open': price_data.values,
                    'volume': volume_data[ticker].values if volume_data is not None and ticker in volume_data.columns else np.ones_like(price_data.values),
                    'vol': volume_data[ticker].values if volume_data is not None and ticker in volume_data.columns else np.ones_like(price_data.values),
                    't': np.arange(len(price_data)),
                    'n': len(price_data)
                }
                
                # 사용자 정의 파라미터 추가
                local_env.update(params)
                
                # 지원되는 함수들 추가
                local_env.update(self.parser.supported_functions)
                
                # 공식 실행
                result = eval(formula, {"__builtins__": {}}, local_env)
                
                # 결과 검증 및 처리
                if isinstance(result, (int, float)):
                    result = np.full(len(price_data), result)
                elif isinstance(result, (list, tuple)):
                    result = np.array(result)
                elif isinstance(result, pd.Series):
                    result = result.values
                
                # NaN 처리
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 길이 맞추기
                if len(result) != len(price_data):
                    if len(result) < len(price_data):
                        # 패딩
                        padding = np.full(len(price_data) - len(result), result[0] if len(result) > 0 else 0)
                        result = np.concatenate([padding, result])
                    else:
                        # 자르기
                        result = result[-len(price_data):]
                
                factor_data[ticker] = result
                
            except Exception as e:
                logger.warning(f"팩터 계산 실패 ({ticker}): {e}")
                continue
        
        if not factor_data:
            raise ValueError("모든 종목에서 팩터 계산에 실패했습니다.")
        
        # DataFrame으로 변환
        factor_df = pd.DataFrame(factor_data, index=universe_data.index)
        
        # 횡단면 순위화
        ranked_factor = factor_df.rank(axis=1, pct=True, method='dense')
        
        return ranked_factor.fillna(0.5)
    
    def create_multiple_factors(self, 
                              formulas: Dict[str, str],
                              universe_data: pd.DataFrame,
                              volume_data: Optional[pd.DataFrame] = None,
                              params: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """여러 공식으로부터 팩터들 생성"""
        
        factors = {}
        
        for factor_name, formula in formulas.items():
            try:
                factor = self.create_factor_from_formula(formula, universe_data, volume_data, params)
                factors[factor_name] = factor
                st.success(f"✅ {factor_name} 팩터 생성 완료")
            except Exception as e:
                st.error(f"❌ {factor_name} 팩터 생성 실패: {e}")
        
        return factors
    
    def get_formula_templates(self) -> Dict[str, Dict[str, str]]:
        """공식 템플릿 제공"""
        templates = {
            "기본 팩터": {
                "모멘텀": "momentum(price, 20)",
                "반전": "-momentum(price, 5)",
                "변동성": "-volatility(price, 20)",
                "거래량 가중 모멘텀": "momentum(price, 20) * normalize(volume)",
                "로그 수익률": "log_returns",
                "Z-점수 정규화": "zscore(price)"
            },
            "기술적 지표": {
                "RSI": "rsi(price, 14)",
                "MACD": "macd(price, 12, 26, 9)",
                "볼린저 밴드": "bollinger_bands(price, 20, 2)",
                "스토캐스틱": "stochastic(price, 14, 3)",
                "Williams %R": "williams_r(price, 14)",
                "CCI": "cci(price, 20)"
            },
            "고급 팩터": {
                "이동평균 크로스오버": "rolling_mean(price, 5) - rolling_mean(price, 20)",
                "변동성 브레이크아웃": "(price - rolling_mean(price, 20)) / rolling_std(price, 20)",
                "거래량 가중 가격": "price * normalize(volume)",
                "모멘텀 가속도": "momentum(price, 5) - momentum(price, 20)",
                "변동성 레지임": "rolling_std(returns, 20) / rolling_std(returns, 60)",
                "가격 모멘텀 + 거래량": "momentum(price, 10) * (1 + normalize(volume))"
            },
            "복합 팩터": {
                "멀티 타임프레임 모멘텀": "(momentum(price, 5) + momentum(price, 10) + momentum(price, 20)) / 3",
                "변동성 조정 모멘텀": "momentum(price, 20) / (rolling_std(returns, 20) + 1e-8)",
                "거래량 가중 RSI": "rsi(price, 14) * normalize(volume)",
                "볼린저 밴드 + RSI": "(bollinger_bands(price, 20, 2) + normalize(rsi(price, 14))) / 2",
                "MACD + 스토캐스틱": "(normalize(macd(price, 12, 26, 9)) + normalize(stochastic(price, 14, 3))) / 2",
                "통합 기술적 지표": "(rsi(price, 14) + normalize(macd(price, 12, 26, 9)) + bollinger_bands(price, 20, 2)) / 3"
            }
        }
        return templates

class FormulaPipeline:
    """공식 기반 팩터 파이프라인"""
    
    def __init__(self):
        self.generator = FormulaFactorGenerator()
        self.parser = FormulaParser()
        
    def run_pipeline(self, 
                    formulas: Dict[str, str],
                    universe_data: pd.DataFrame,
                    volume_data: Optional[pd.DataFrame] = None,
                    params: Dict[str, Any] = None,
                    combine_method: str = "ic_weighted") -> Dict[str, Any]:
        """공식 파이프라인 실행"""
        
        # 1. 팩터 생성
        factors = self.generator.create_multiple_factors(
            formulas, universe_data, volume_data, params
        )
        
        if not factors:
            raise ValueError("생성된 팩터가 없습니다.")
        
        # 2. 팩터 결합
        if len(factors) > 1 and combine_method != "none":
            combined_factor = self._combine_factors(factors, universe_data, combine_method)
        else:
            combined_factor = list(factors.values())[0]
        
        # 3. 성과 분석
        performance = self._analyze_performance(combined_factor, universe_data)
        
        return {
            'individual_factors': factors,
            'combined_factor': combined_factor,
            'performance': performance,
            'formulas': formulas,
            'params': params
        }
    
    def _combine_factors(self, 
                        factors: Dict[str, pd.DataFrame],
                        universe_data: pd.DataFrame,
                        method: str) -> pd.DataFrame:
        """팩터 결합"""
        
        if method == "equal_weight":
            # 동일 가중치
            weights = {name: 1.0/len(factors) for name in factors.keys()}
            combined = sum(factor * weight for factor, weight in zip(factors.values(), weights.values()))
            
        elif method == "ic_weighted":
            # IC 기반 가중치 (간단한 버전)
            future_returns = universe_data.pct_change().shift(-1)
            ic_scores = {}
            
            for name, factor in factors.items():
                ic = self._calculate_ic(factor, future_returns)
                ic_scores[name] = abs(ic)
            
            total_ic = sum(ic_scores.values())
            if total_ic > 0:
                weights = {name: ic/total_ic for name, ic in ic_scores.items()}
            else:
                weights = {name: 1.0/len(factors) for name in factors.keys()}
            
            combined = sum(factors[name] * weight for name, weight in weights.items())
            
        else:
            # 기본적으로 첫 번째 팩터 사용
            combined = list(factors.values())[0]
        
        return combined
    
    def _calculate_ic(self, factor: pd.DataFrame, future_returns: pd.DataFrame) -> float:
        """Information Coefficient 계산"""
        try:
            # 공통 기간에 대해서만 계산
            common_dates = factor.index.intersection(future_returns.index)
            if len(common_dates) < 10:
                return 0.0
            
            factor_vals = factor.loc[common_dates]
            future_vals = future_returns.loc[common_dates]
            
            ic_values = []
            for date in common_dates[-20:]:  # 최근 20일
                factor_series = factor_vals.loc[date].dropna()
                future_series = future_vals.loc[date].dropna()
                
                common_tickers = factor_series.index.intersection(future_series.index)
                if len(common_tickers) >= 5:
                    ic = factor_series[common_tickers].corr(future_series[common_tickers], method='spearman')
                    if not np.isnan(ic):
                        ic_values.append(ic)
            
            return np.mean(ic_values) if ic_values else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_performance(self, factor: pd.DataFrame, universe_data: pd.DataFrame) -> Dict[str, float]:
        """성과 분석"""
        try:
            future_returns = universe_data.pct_change().shift(-1)
            ic = self._calculate_ic(factor, future_returns)
            
            return {
                'ic': ic,
                'ic_abs': abs(ic),
                'factor_std': factor.std().mean(),
                'factor_mean': factor.mean().mean(),
                'data_points': len(factor) * len(factor.columns)
            }
        except Exception:
            return {
                'ic': 0.0,
                'ic_abs': 0.0,
                'factor_std': 0.0,
                'factor_mean': 0.0,
                'data_points': 0
            } 
