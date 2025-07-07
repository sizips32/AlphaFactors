"""
데이터 다운로드, 캐싱, 전처리 담당 클래스 (개선된 버전)
Handles data downloading, caching, and preprocessing (Improved Version)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import pickle
import hashlib

from config import DataConfig
from utils import (
    validate_dataframe,
    normalize_timezone,
    clean_data,
    create_date_range,
    display_error_with_suggestions,
    logger
)

class DataHandler:
    """데이터 다운로드, 캐싱, 전처리를 담당하는 클래스 (개선된 버전)"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.local_path = config.local_data_path
        self._ensure_directories()

    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(os.path.join(self.local_path, 'cache'), exist_ok=True)
        os.makedirs(os.path.join(self.local_path, 'processed'), exist_ok=True)

    def _generate_cache_key(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """캐시 키 생성"""
        key_string = f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """캐시 파일 경로 반환"""
        return os.path.join(self.local_path, 'cache', f"{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """캐시 유효성 확인 (파일 존재 및 유효 기간)"""
        if not os.path.exists(cache_path):
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        return cache_age < timedelta(seconds=self.config.cache_ttl)

    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드"""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logger.warning(f"캐시 로드 실패 ({cache_path}): {e}. 캐시 파일을 삭제합니다.")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: str):
        """데이터를 캐시에 저장"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.error(f"캐시 저장 실패 ({cache_path}): {e}")

    def download_data(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
        """단일 종목 데이터 다운로드 및 캐싱 (로직 개선)"""
        if not ticker or not isinstance(ticker, str):
            st.error("유효한 티커 문자열을 입력해야 합니다.")
            return None
        ticker = ticker.upper().strip()

        cache_key = self._generate_cache_key(ticker, start, end)
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            df = self._load_from_cache(cache_path)
            if df is not None:
                st.success(f"캐시에서 {ticker} 데이터를 로드했습니다.")
                return normalize_timezone(df)

        with st.spinner(f"{ticker} 데이터를 다운로드 중..."):
            try:
                df = self._download_from_source(ticker, start, end)
                if df is None or not validate_dataframe(df, ticker, self.config.required_columns):
                    return None
                
                df = normalize_timezone(df)
                df = clean_data(df, self.config.required_columns)
                
                self._save_to_cache(df, cache_path)
                st.success(f"{ticker} 데이터 다운로드 완료: {len(df)} 행")
                return df
            except Exception as e:
                error_msg = f"{ticker} 데이터 다운로드 실패: {str(e)}"
                suggestions = [
                    "티커 심볼이 올바른지 확인하세요 (예: AAPL, GOOGL).",
                    "네트워크 연결을 확인하세요.",
                    "날짜 범위를 조정해보세요."
                ]
                display_error_with_suggestions(error_msg, suggestions)
                return None

    def _download_from_source(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
        """데이터 소스에서 다운로드 (fdr 우선, yfinance 백업)"""
        try:
            df = fdr.DataReader(ticker, start, end)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"FinanceDataReader 다운로드 실패 ({ticker}): {e}. yfinance로 재시도합니다.")

        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                st.error(f"'{ticker}'에 대한 데이터를 찾을 수 없습니다.")
                return None
            return df
        except Exception as e:
            raise IOError(f"yfinance 다운로드도 실패했습니다 ({ticker}): {e}")

    def download_universe_data(self, tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """유니버스 전체 종목 데이터 다운로드 (메모리 최적화)"""
        price_series_list, volume_series_list = [], []
        
        # 메모리 사용량 제한 (최대 종목 수)
        max_tickers = min(len(tickers), 50)  # 메모리 절약을 위해 최대 50개로 제한
        if len(tickers) > max_tickers:
            st.warning(f"메모리 절약을 위해 {max_tickers}개 종목으로 제한합니다.")
            tickers = tickers[:max_tickers]
        
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            df = self.download_data(ticker, start, end)
            if df is not None and len(df) > 50:
                # 메모리 절약을 위해 필요한 컬럼만 저장
                price_series_list.append(df['Close'].rename(ticker))
                volume_series_list.append(df['Volume'].rename(ticker))
                
                # 원본 데이터 즉시 삭제로 메모리 절약
                del df
            progress_bar.progress((i + 1) / len(tickers))
        progress_bar.empty()

        if not price_series_list:
            st.error("유효한 데이터를 가진 종목이 없습니다.")
            return None, None

        # pd.concat으로 한 번에 DataFrame 생성
        universe_df = pd.concat(price_series_list, axis=1)
        volume_df = pd.concat(volume_series_list, axis=1)

        # 공통 날짜 인덱스 생성 및 데이터 정렬
        common_index = universe_df.dropna().index
        universe_df = universe_df.loc[common_index].ffill()
        volume_df = volume_df.loc[common_index].ffill()

        st.success(f"✅ {len(universe_df.columns)}개 종목, {len(universe_df)}일 데이터 준비 완료")
        return universe_df, volume_df

    def create_features_targets(self, df: pd.DataFrame, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """시계열 특징/타겟 생성 (가독성 및 효율성 개선)"""
        
        # 1. 특징(Feature) 계산
        features = pd.DataFrame(index=df.index)
        features['returns'] = df['Close'].pct_change()
        features['log_volume'] = np.log1p(df['Volume'])
        # ... 추가적인 특징들 ...
        features = features.fillna(0).replace([np.inf, -np.inf], 0)

        # 2. 타겟(Target) 계산
        target = df['Close'].pct_change(periods=horizon).shift(-horizon).rename('target')

        # 3. 데이터셋 생성 (windowing)
        X, y, dates = [], [], []
        # 공통 인덱스에서만 루프 실행
        valid_indices = features.dropna().index.intersection(target.dropna().index)
        
        for i in range(len(valid_indices) - window):
            window_end_idx = valid_indices[i + window -1]
            target_idx = valid_indices[i + window -1] # 현재 윈도우의 마지막 날을 기준으로 타겟 예측
            
            if target_idx not in target.index: continue

            window_slice = features.loc[valid_indices[i]:window_end_idx]
            
            X.append(window_slice.values.flatten()) # 윈도우를 1차원 배열로 펼침
            y.append(target.loc[target_idx])
            dates.append(target_idx)

        return np.array(X), np.array(y), dates

    def clear_cache(self, max_age_days: int = 7, max_size_mb: int = 500):
        """캐시 정리 (오래된 파일 및 크기 제한)"""
        cache_dir = os.path.join(self.local_path, 'cache')
        removed_count = 0
        total_size = 0
        
        try:
            # 파일 크기와 수정 시간 정보 수집
            files_info = []
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    file_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
                    files_info.append((file_path, stat.st_size, file_age))
                    total_size += stat.st_size
            
            # 크기 제한 확인 (MB 단위)
            total_size_mb = total_size / (1024 * 1024)
            
            # 오래된 파일 삭제
            for file_path, file_size, file_age in files_info:
                if file_age > timedelta(days=max_age_days):
                    os.remove(file_path)
                    removed_count += 1
                    total_size_mb -= file_size / (1024 * 1024)
            
            # 크기 제한 초과시 오래된 순으로 추가 삭제
            if total_size_mb > max_size_mb:
                files_info = [(p, s, a) for p, s, a in files_info if os.path.exists(p)]
                files_info.sort(key=lambda x: x[2], reverse=True)  # 오래된 순
                
                for file_path, file_size, _ in files_info:
                    if total_size_mb <= max_size_mb:
                        break
                    os.remove(file_path)
                    removed_count += 1
                    total_size_mb -= file_size / (1024 * 1024)
            
            st.success(f"캐시 정리 완료: {removed_count}개 파일 삭제, 현재 크기: {total_size_mb:.1f}MB")
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
