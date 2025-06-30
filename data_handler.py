import os
import pandas as pd
import numpy as np
import streamlit as st
import FinanceDataReader as fdr
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import pickle
import hashlib

from config import DataConfig, ModelConfig
from utils import (
    validate_dataframe, 
    normalize_timezone, 
    clean_data, 
    create_date_range,
    show_dataframe_info,
    display_error_with_suggestions,
    logger
)

class DataHandler:
    """데이터 다운로드, 캐싱, 전처리를 담당하는 클래스"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.local_path = config.local_data_path
        self._ensure_directories()
        
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.local_path, exist_ok=True)
        os.makedirs(os.path.join(self.local_path, 'cache'), exist_ok=True)
        os.makedirs(os.path.join(self.local_path, 'processed'), exist_ok=True)
    
    def _generate_cache_key(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """캐시 키 생성"""
        key_string = f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """캐시 파일 경로 반환"""
        return os.path.join(self.local_path, 'cache', f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """캐시 유효성 확인"""
        if not os.path.exists(cache_path):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < timedelta(hours=max_age_hours)
    
    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드"""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: str) -> bool:
        """데이터를 캐시에 저장"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            return True
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False
    
    def download_data(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, 
                     use_cache: bool = True, clean: bool = True) -> Optional[pd.DataFrame]:
        """데이터 다운로드 및 캐싱"""
        
        # 입력 검증
        if not ticker or not ticker.strip():
            st.error("티커를 입력해주세요.")
            return None
        
        ticker = ticker.upper().strip()
        
        try:
            start, end = create_date_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        except Exception:
            return None
        
        # 캐시 확인
        cache_key = self._generate_cache_key(ticker, start, end)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path):
            df = self._load_from_cache(cache_path)
            if df is not None and validate_dataframe(df, ticker, self.config.required_columns):
                st.success(f"캐시에서 {ticker} 데이터를 로드했습니다.")
                return normalize_timezone(df)
        
        # 새로운 데이터 다운로드
        with st.spinner(f"{ticker} 데이터를 다운로드 중..."):
            try:
                df = self._download_from_source(ticker, start, end)
                if df is None:
                    return None
                
                # 데이터 검증
                if not validate_dataframe(df, ticker, self.config.required_columns):
                    return None
                
                # 타임존 정규화
                df = normalize_timezone(df)
                
                # 데이터 정리
                if clean:
                    df = clean_data(df, self.config.required_columns)
                
                # 캐시 저장
                if use_cache:
                    self._save_to_cache(df, cache_path)
                
                st.success(f"{ticker} 데이터 다운로드 완료: {len(df)}개 데이터")
                return df
                
            except Exception as e:
                error_msg = f"{ticker} 데이터 다운로드 실패: {str(e)}"
                suggestions = [
                    "티커 심볼이 올바른지 확인하세요 (예: AAPL, GOOGL)",
                    "네트워크 연결을 확인하세요",
                    "날짜 범위를 조정해보세요",
                    "상장폐지된 종목이 아닌지 확인하세요"
                ]
                display_error_with_suggestions(error_msg, suggestions)
                return None
    
    def _download_from_source(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
        """실제 데이터 소스에서 다운로드"""
        try:
            # FinanceDataReader 사용
            df = fdr.DataReader(ticker, start, end)
            
            if df.empty:
                st.error(f"'{ticker}'에 대한 데이터가 없습니다.")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"FDR 다운로드 실패: {e}")
            # 백업으로 yfinance 시도
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(start=start, end=end)
                
                if df.empty:
                    return None
                
                # 컬럼명 표준화
                df = df.rename(columns={
                    'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                    'Close': 'Close', 'Volume': 'Volume'
                })
                
                return df
                
            except Exception as e2:
                logger.error(f"yfinance 백업 다운로드 실패: {e2}")
                raise e
    
    def get_multiple_tickers(self, tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """여러 티커의 데이터를 동시에 다운로드"""
        results = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            
            df = self.download_data(ticker, start, end)
            if df is not None:
                results[ticker] = df
            
        progress_bar.empty()
        
        if results:
            st.success(f"{len(results)}개 종목 데이터 다운로드 완료")
        else:
            st.error("다운로드된 데이터가 없습니다.")
        
        return results
    
    def download_universe_data(self, tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """유니버스 전체 종목 데이터 다운로드 및 정렬"""
        
        price_data = {}
        volume_data = {}
        
        with st.spinner(f"{len(tickers)}개 종목 데이터 다운로드 중..."):
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                
                try:
                    df = self.download_data(ticker, start, end, use_cache=True, clean=True)
                    
                    if df is not None and len(df) > 50:  # 최소 50일 데이터 필요
                        price_data[ticker] = df['Close']
                        volume_data[ticker] = df['Volume']
                        
                except Exception as e:
                    st.warning(f"{ticker} 다운로드 실패: {e}")
                    continue
            
            progress_bar.empty()
        
        if not price_data:
            st.error("다운로드된 데이터가 없습니다.")
            return None, None
        
        # DataFrame으로 변환 및 정렬
        try:
            universe_df = pd.DataFrame(price_data)
            volume_df = pd.DataFrame(volume_data)
            
            # 결측값 처리
            universe_df = universe_df.fillna(method='ffill').fillna(method='bfill')
            volume_df = volume_df.fillna(method='ffill').fillna(method='bfill')
            
            # 모든 종목이 데이터를 가진 기간만 선택
            min_length = min(len(universe_df), 252)  # 최대 1년
            universe_df = universe_df.tail(min_length)
            volume_df = volume_df.tail(min_length)
            
            # 최종 결측값 체크
            if universe_df.isnull().any().any():
                st.warning("일부 결측값이 있어 보간 처리했습니다.")
                universe_df = universe_df.interpolate(method='linear')
                volume_df = volume_df.interpolate(method='linear')
            
            st.success(f"✅ {len(universe_df.columns)}개 종목, {len(universe_df)}일 데이터 준비 완료")
            
            return universe_df, volume_df
            
        except Exception as e:
            st.error(f"유니버스 데이터 정렬 실패: {e}")
            return None, None
    
    def create_features_targets(self, df: pd.DataFrame, window: int = None, 
                              horizon: int = None) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """시계열 특징/타겟 생성"""
        if window is None:
            window = 10
        if horizon is None:
            horizon = 5
        
        feature_columns = self.config.required_columns
        X, y, dates = [], [], []
        
        for i in range(len(df) - window - horizon):
            # 특징 추출
            window_data = df[feature_columns].iloc[i:i+window]
            
            # 정규화된 특징 생성
            features = []
            for col in feature_columns:
                values = window_data[col].values
                if col == 'Volume':
                    # Volume은 로그 변환
                    features.extend(np.log1p(values))
                else:
                    # 가격 데이터는 수익률로 변환
                    returns = np.diff(values) / values[:-1]
                    features.extend(returns)
                    features.append(values[-1])  # 최근 가격
            
            X.append(features)
            
            # 타겟 (미래 수익률)
            current_price = df['Close'].iloc[i + window]
            future_price = df['Close'].iloc[i + window + horizon]
            future_return = (future_price / current_price) - 1
            y.append(future_return)
            
            dates.append(df.index[i + window])
        
        return np.array(X), np.array(y), dates
    
    def save_processed_data(self, data: Dict, filename: str):
        """처리된 데이터 저장"""
        filepath = os.path.join(self.local_path, 'processed', filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            st.success(f"데이터가 저장되었습니다: {filepath}")
        except Exception as e:
            st.error(f"데이터 저장 실패: {e}")
    
    def load_processed_data(self, filename: str) -> Optional[Dict]:
        """처리된 데이터 로드"""
        filepath = os.path.join(self.local_path, 'processed', filename)
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            return None
    
    def clear_cache(self):
        """캐시 정리"""
        cache_dir = os.path.join(self.local_path, 'cache')
        try:
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            st.success("캐시가 정리되었습니다.")
        except Exception as e:
            st.error(f"캐시 정리 실패: {e}")
    
    def get_cache_info(self) -> Dict:
        """캐시 정보 반환"""
        cache_dir = os.path.join(self.local_path, 'cache')
        info = {
            'cache_files': 0,
            'total_size': 0,
            'oldest_file': None,
            'newest_file': None
        }
        
        try:
            files = os.listdir(cache_dir)
            info['cache_files'] = len(files)
            
            if files:
                file_stats = []
                for filename in files:
                    file_path = os.path.join(cache_dir, filename)
                    stat = os.stat(file_path)
                    file_stats.append((filename, stat.st_size, stat.st_mtime))
                    info['total_size'] += stat.st_size
                
                file_stats.sort(key=lambda x: x[2])
                info['oldest_file'] = datetime.fromtimestamp(file_stats[0][2])
                info['newest_file'] = datetime.fromtimestamp(file_stats[-1][2])
                
        except Exception as e:
            logger.error(f"캐시 정보 조회 실패: {e}")
        
        return info