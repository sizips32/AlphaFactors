import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt

from config import QlibConfig
from utils import logger, display_error_with_suggestions

# Qlib 관련 임포트
try:
    import qlib
    from qlib.constant import REG_US
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.data.handler import Alpha158
    QLIB_AVAILABLE = True
except ImportError as e:
    QLIB_AVAILABLE = False
    logger.error(f"Qlib 임포트 실패: {e}")

class QlibHandler:
    """Qlib 백테스팅을 담당하는 클래스"""
    
    def __init__(self, config: QlibConfig, data_path: str):
        self.config = config
        self.data_path = data_path
        self.is_initialized = False
        
        if QLIB_AVAILABLE:
            self._initialize_qlib()
        else:
            st.error("Qlib가 설치되어 있지 않습니다. 'pip install pyqlib'로 설치하세요.")
    
    def _initialize_qlib(self):
        """Qlib 초기화"""
        if not os.path.exists(self.data_path):
            error_msg = f"Qlib 미국 데이터셋이 없습니다: {self.data_path}"
            suggestions = [
                "다음 명령어로 데이터셋을 다운로드하세요:",
                "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us",
                "또는 Qlib 공식 문서를 참고하세요: https://qlib.readthedocs.io/"
            ]
            display_error_with_suggestions(error_msg, suggestions)
            return
        
        try:
            qlib.init(provider_uri=self.data_path, region=REG_US)
            self.is_initialized = True
            logger.info("Qlib 초기화 성공")
        except Exception as e:
            error_msg = f"Qlib 초기화 실패: {e}"
            suggestions = [
                "데이터 경로가 올바른지 확인하세요",
                "Qlib 데이터가 손상되지 않았는지 확인하세요",
                "다시 데이터를 다운로드해보세요"
            ]
            display_error_with_suggestions(error_msg, suggestions)
    
    def check_availability(self) -> bool:
        """Qlib 사용 가능 여부 확인"""
        return QLIB_AVAILABLE and self.is_initialized
    
    def get_available_instruments(self) -> List[str]:
        """사용 가능한 instrument 목록 반환"""
        return self.config.available_instruments
    
    def load_alpha158_data(self, instrument: str = None) -> Optional[pd.DataFrame]:
        """Alpha158 데이터 로드"""
        if not self.check_availability():
            return None
        
        if instrument is None:
            instrument = self.config.default_instrument
        
        try:
            with st.spinner(f"{instrument} Alpha158 데이터 로딩 중..."):
                handler = Alpha158(instruments=instrument)
                df = handler.fetch(col_set="feature")
                
                if df.empty:
                    st.error(f"{instrument}에 대한 데이터가 없습니다.")
                    return None
                
                st.success(f"{instrument} 데이터 로드 완료: {df.shape}")
                return df
                
        except Exception as e:
            error_msg = f"Alpha158 데이터 로드 실패: {e}"
            suggestions = [
                "instrument 이름이 올바른지 확인하세요",
                "Qlib 데이터가 최신인지 확인하세요",
                "네트워크 연결을 확인하세요"
            ]
            display_error_with_suggestions(error_msg, suggestions)
            return None
    
    def prepare_custom_factor(self, factor_series: pd.Series, 
                            qlib_data: pd.DataFrame) -> Optional[pd.Series]:
        """올바른 횡단면 팩터를 Qlib 형식에 맞게 준비"""
        
        # 이미 MultiIndex 형태인지 확인 (올바른 알파 팩터)
        if isinstance(factor_series.index, pd.MultiIndex):
            # 이미 올바른 형식
            st.info(f"✅ 올바른 횡단면 팩터 확인: {len(factor_series)}개 데이터 포인트")
            return factor_series
        
        # 구 버전 호환성: 단일 시리즈를 MultiIndex로 변환 (권장하지 않음)
        if isinstance(factor_series.index, pd.DatetimeIndex):
            st.warning("⚠️ 단일 시계열 팩터가 감지되었습니다. 이는 올바른 알파 팩터가 아닙니다.")
            
            if not isinstance(qlib_data.index, pd.MultiIndex):
                st.error("Qlib 데이터가 MultiIndex 형식이 아닙니다.")
                return None
            
            try:
                # Qlib 데이터의 날짜 및 종목 정보 추출
                qlib_dates = qlib_data.index.get_level_values('datetime').unique()
                qlib_instruments = qlib_data.index.get_level_values('instrument').unique()
                
                # 팩터 시리즈와 Qlib 데이터의 날짜 교집합 구하기
                common_dates = factor_series.index.intersection(qlib_dates)
                
                if len(common_dates) == 0:
                    st.error("팩터 데이터와 Qlib 데이터 간에 공통 날짜가 없습니다.")
                    return None
                
                # 단일 종목 팩터를 전체 포트폴리오로 확장 (문제가 있는 방식)
                expanded_factor = []
                
                for date in common_dates:
                    if date in factor_series.index:
                        factor_value = factor_series.loc[date]
                        for instrument in qlib_instruments:
                            expanded_factor.append((date, instrument, factor_value))
                
                if not expanded_factor:
                    st.error("확장된 팩터 데이터가 없습니다.")
                    return None
                
                # MultiIndex DataFrame 생성
                index_tuples = [(row[0], row[1]) for row in expanded_factor]
                values = [row[2] for row in expanded_factor]
                
                multi_index = pd.MultiIndex.from_tuples(
                    index_tuples, names=['datetime', 'instrument']
                )
                
                custom_factor = pd.Series(values, index=multi_index, name='custom_factor')
                
                st.warning(f"⚠️ 단일 값을 모든 종목에 적용한 팩터: {len(custom_factor)}개 데이터 포인트")
                st.warning("이는 의미 있는 포트폴리오를 구성할 수 없습니다.")
                return custom_factor
                
            except Exception as e:
                st.error(f"팩터 준비 실패: {e}")
                return None
        
        else:
            st.error("지원하지 않는 팩터 형식입니다.")
            return None
    
    def run_backtest(self, instrument: str = None, custom_factor: pd.Series = None,
                    show_details: bool = True) -> Tuple[Optional[pd.Series], Optional[Dict]]:
        """개선된 백테스팅 실행"""
        
        if not self.check_availability():
            return None, None
        
        if instrument is None:
            instrument = self.config.default_instrument
        
        try:
            # Alpha158 데이터 로드
            qlib_data = self.load_alpha158_data(instrument)
            if qlib_data is None:
                return None, None
            
            # 팩터 준비
            if custom_factor is not None:
                prepared_factor = self.prepare_custom_factor(custom_factor, qlib_data)
                if prepared_factor is None:
                    st.warning("❌ 커스텀 팩터를 사용할 수 없어 기본 팩터(RESI5)를 사용합니다.")
                    factor_to_use = qlib_data['RESI5']
                else:
                    # 올바른 형식인지 체크
                    if isinstance(prepared_factor.index, pd.MultiIndex):
                        st.success("✅ 올바른 횡단면 팩터로 백테스팅을 진행합니다.")
                    else:
                        st.warning("⚠️ 단일 시계열 팩터로 백테스팅을 진행합니다. (정확한 결과가 아닐 수 있음)")
                    
                    # 커스텀 팩터를 Qlib 데이터에 추가
                    try:
                        qlib_data['custom_factor'] = prepared_factor
                        factor_to_use = qlib_data['custom_factor']
                        
                        # 팩터 통계 표시
                        factor_stats = {
                            '데이터 포인트': len(prepared_factor),
                            '팩터 값 범위': f"{prepared_factor.min():.4f} ~ {prepared_factor.max():.4f}",
                            '평균': f"{prepared_factor.mean():.4f}",
                            '표준편차': f"{prepared_factor.std():.4f}"
                        }
                        
                        st.info("**팩터 통계:**")
                        for key, value in factor_stats.items():
                            st.write(f"- {key}: {value}")
                            
                    except Exception as factor_error:
                        st.error(f"팩터 적용 실패: {factor_error}")
                        st.info("기본 팩터(RESI5)를 사용합니다.")
                        factor_to_use = qlib_data['RESI5']
            else:
                factor_to_use = qlib_data['RESI5']
                st.info("기본 팩터(RESI5)를 사용하여 백테스팅을 진행합니다.")
            
            # 전략 실행
            strategy = TopkDropoutStrategy(
                topk=self.config.topk, 
                n_drop=self.config.n_drop
            )
            
            with st.spinner("백테스팅 실행 중..."):
                positions = strategy.generate_position(factor_to_use)
                
                # 수익률 계산
                daily_returns = (positions.shift(1) * qlib_data['LABEL0']).sum(axis=1)
                cum_returns = (1 + daily_returns).cumprod()
                
                # 리스크 분석
                risk_metrics = risk_analysis(daily_returns)
            
            if show_details:
                self._display_backtest_results(cum_returns, risk_metrics, daily_returns)
            
            return cum_returns, risk_metrics
            
        except Exception as e:
            error_msg = f"백테스팅 실패: {e}"
            suggestions = [
                "데이터 기간을 확인하세요",
                "커스텀 팩터 형식을 확인하세요",
                "Qlib 설정을 다시 확인하세요"
            ]
            display_error_with_suggestions(error_msg, suggestions)
            return None, None
    
    def _display_backtest_results(self, cum_returns: pd.Series, 
                                 risk_metrics: Dict, 
                                 daily_returns: pd.Series):
        """백테스팅 결과 표시"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 누적 수익률")
            fig, ax = plt.subplots(figsize=(12, 6))
            cum_returns.plot(ax=ax, linewidth=2)
            ax.set_title("포트폴리오 누적 수익률")
            ax.set_xlabel("날짜")
            ax.set_ylabel("누적 수익률")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            # 주요 성과 지표
            total_return = cum_returns.iloc[-1] - 1
            volatility = daily_returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(cum_returns)
            
            st.metric("총 수익률", f"{total_return:.2%}")
            st.metric("연간 변동성", f"{volatility:.2%}")
            st.metric("최대 손실폭", f"{max_drawdown:.2%}")
        
        with col2:
            st.subheader("📊 리스크 분석")
            
            # 리스크 메트릭을 DataFrame으로 변환
            risk_df = pd.DataFrame(risk_metrics, index=[0]).T
            risk_df.columns = ['값']
            
            # 주요 지표만 표시
            important_metrics = ['IC', 'ICIR', 'Rank IC', 'Rank ICIR']
            filtered_risk_df = risk_df.loc[risk_df.index.intersection(important_metrics)]
            
            if isinstance(filtered_risk_df, pd.DataFrame):
                for col in filtered_risk_df.columns:
                    if filtered_risk_df[col].dtype == 'object':
                        filtered_risk_df[col] = filtered_risk_df[col].astype(str)
            st.dataframe(filtered_risk_df, use_container_width=True)
            
            # 일별 수익률 분포
            st.subheader("📊 일별 수익률 분포")
            fig, ax = plt.subplots(figsize=(10, 6))
            daily_returns.hist(bins=50, ax=ax, alpha=0.7)
            ax.axvline(daily_returns.mean(), color='red', linestyle='--', 
                      label=f'평균: {daily_returns.mean():.4f}')
            ax.set_xlabel("일별 수익률")
            ax.set_ylabel("빈도")
            ax.set_title("일별 수익률 분포")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
    
    def _calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """최대 손실폭 계산"""
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def compare_strategies(self, factors: Dict[str, pd.Series], 
                          instrument: str = None) -> Dict[str, Tuple[pd.Series, Dict]]:
        """여러 전략 비교"""
        
        if not self.check_availability():
            return {}
        
        results = {}
        
        for factor_name, factor_series in factors.items():
            st.info(f"{factor_name} 전략 백테스팅 중...")
            cum_returns, risk_metrics = self.run_backtest(
                instrument=instrument, 
                custom_factor=factor_series,
                show_details=False
            )
            
            if cum_returns is not None and risk_metrics is not None:
                results[factor_name] = (cum_returns, risk_metrics)
        
        if results:
            self._display_strategy_comparison(results)
        
        return results
    
    def _display_strategy_comparison(self, results: Dict[str, Tuple[pd.Series, Dict]]):
        """전략 비교 결과 표시"""
        
        st.subheader("🔄 전략 비교")
        
        # 누적 수익률 비교
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for strategy_name, (cum_returns, _) in results.items():
            cum_returns.plot(ax=ax, label=strategy_name, linewidth=2)
        
        ax.set_title("전략별 누적 수익률 비교")
        ax.set_xlabel("날짜")
        ax.set_ylabel("누적 수익률")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        # 성과 지표 비교 테이블
        comparison_data = []
        for strategy_name, (cum_returns, risk_metrics) in results.items():
            total_return = cum_returns.iloc[-1] - 1
            max_drawdown = self._calculate_max_drawdown(cum_returns)
            
            comparison_data.append({
                '전략': strategy_name,
                '총 수익률': f"{total_return:.2%}",
                '최대 손실폭': f"{max_drawdown:.2%}",
                'IC': f"{risk_metrics.get('IC', 0):.4f}",
                'Sharpe': f"{risk_metrics.get('SR', 0):.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if isinstance(comparison_df, pd.DataFrame):
            for col in comparison_df.columns:
                if comparison_df[col].dtype == 'object':
                    comparison_df[col] = comparison_df[col].astype(str)
        st.dataframe(comparison_df, use_container_width=True)
    
    def export_results(self, cum_returns: pd.Series, risk_metrics: Dict, 
                      filename: str = "backtest_results.xlsx"):
        """결과 내보내기"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 누적 수익률
                cum_returns.to_frame('cum_returns').to_excel(
                    writer, sheet_name='CumulativeReturns'
                )
                
                # 리스크 메트릭
                risk_df = pd.DataFrame(risk_metrics, index=[0]).T
                risk_df.to_excel(writer, sheet_name='RiskMetrics')
            
            st.success(f"결과가 저장되었습니다: {filename}")
            
        except Exception as e:
            st.error(f"결과 저장 실패: {e}")
