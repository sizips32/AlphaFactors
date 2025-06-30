import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Optional, Union
import logging
from datetime import datetime

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def validate_dataframe(df: pd.DataFrame, ticker: str, required_columns: List[str]) -> bool:
    """데이터프레임 유효성 검사"""
    if df is None:
        st.error(f"'{ticker}' 데이터가 None입니다.")
        return False
        
    if df.empty:
        st.error(f"'{ticker}' 데이터가 비어있습니다.")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        return False
    
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error(f"'{ticker}' 데이터의 인덱스가 날짜 형식이 아닙니다. (현재 타입: {type(df.index)})")
        return False
    
    # 데이터 품질 검사
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        st.warning(f"'{ticker}' 데이터에 결측값이 있습니다: {null_counts[null_counts > 0].to_dict()}")
    
    # 최소 데이터 개수 확인
    if len(df) < 50:
        st.error(f"'{ticker}' 데이터가 너무 적습니다. (현재: {len(df)}개, 최소 필요: 50개)")
        return False
    
    return True

def normalize_timezone(df: pd.DataFrame, target_tz: str = 'Asia/Seoul') -> pd.DataFrame:
    """타임존 정규화"""
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(target_tz)
        else:
            df.index = df.index.tz_convert(target_tz)
        return df
    except Exception as e:
        logger.warning(f"타임존 정규화 실패: {e}")
        return df

def clean_data(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """데이터 정리 및 전처리"""
    df_clean = df.copy()
    
    # 결측값 처리 (forward fill 후 backward fill)
    df_clean[required_columns] = df_clean[required_columns].fillna(method='ffill').fillna(method='bfill')
    
    # 이상값 처리 (각 컬럼별로 99.5% 분위수로 캡핑)
    for col in required_columns:
        if col != 'Volume':  # Volume은 0이 될 수 있으므로 제외
            upper_bound = df_clean[col].quantile(0.995)
            lower_bound = df_clean[col].quantile(0.005)
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Volume이 0인 경우 처리
    if 'Volume' in required_columns:
        df_clean['Volume'] = df_clean['Volume'].replace(0, df_clean['Volume'].median())
    
    return df_clean

def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.Series:
    """수익률 계산"""
    return df[price_col].pct_change().fillna(0)

def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """변동성 계산"""
    return returns.rolling(window=window).std()

def format_performance_metrics(metrics: dict) -> str:
    """성과 지표 포맷팅"""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in key.lower() or 'sharpe' in key.lower():
                formatted.append(f"**{key}**: {value:.4f}")
            elif 'return' in key.lower():
                formatted.append(f"**{key}**: {value:.2%}")
            else:
                formatted.append(f"**{key}**: {value:.4f}")
        else:
            formatted.append(f"**{key}**: {value}")
    return "\n".join(formatted)

def safe_division(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], default: float = 0.0) -> Union[float, np.ndarray]:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        return np.where(denominator != 0, numerator / denominator, default)

def create_date_range(start_date: str, end_date: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """날짜 범위 생성 및 검증"""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start >= end:
            raise ValueError("시작일이 종료일보다 늦습니다.")
        
        if end > pd.Timestamp.now():
            st.warning("종료일이 현재 날짜보다 미래입니다. 현재 날짜로 조정합니다.")
            end = pd.Timestamp.now()
        
        return start, end
    except Exception as e:
        st.error(f"날짜 형식이 올바르지 않습니다: {e}")
        raise

def show_dataframe_info(df: pd.DataFrame, title: str = "데이터 정보"):
    """데이터프레임 정보 표시"""
    with st.expander(f"📊 {title}", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("데이터 수", len(df))
        
        with col2:
            st.metric("시작일", df.index.min().strftime('%Y-%m-%d'))
        
        with col3:
            st.metric("종료일", df.index.max().strftime('%Y-%m-%d'))
        
        st.dataframe(df.describe(), use_container_width=True)

def display_error_with_suggestions(error_msg: str, suggestions: List[str] = None):
    """에러 메시지와 해결 방안 표시"""
    st.error(error_msg)
    
    if suggestions:
        with st.expander("💡 해결 방안", expanded=True):
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")

def cache_key_generator(*args) -> str:
    """캐시 키 생성"""
    return "_".join(str(arg) for arg in args)

@st.cache_data(ttl=3600)  # 1시간 캐시
def cached_calculation(func, *args, **kwargs):
    """계산 결과 캐싱"""
    return func(*args, **kwargs)