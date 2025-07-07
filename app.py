import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, List, Any

# 새로운 모듈들 임포트
from config import AppConfig
from data_handler import DataHandler
from models import ModelTrainer
from qlib_handler import QlibHandler
from alpha_factors import AlphaFactorEngine
from portfolio_backtester import PortfolioBacktester, FactorBacktester
from font_config import apply_korean_style
from utils import (
    show_dataframe_info, 
    display_error_with_suggestions,
    logger,
    load_factors_from_zoo,
    delete_factor_from_zoo,
    save_factor_to_zoo,
    analyze_factor_performance_text,
    analyze_backtest_performance_text
)

st.set_page_config(
  page_title="AlphaFactors",
  page_icon=":chart_with_upwards_trend:",
  layout="wide"
)

class AlphaForgeApp:
    """AlphaForge 메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.config = AppConfig()
        self.data_handler = DataHandler(self.config.data)
        self.model_trainer = ModelTrainer(self.config.model)
        self.qlib_handler = QlibHandler(self.config.qlib, self.config.data.qlib_data_path)
        self.alpha_engine = AlphaFactorEngine(self.config.factor)
        
        # 한글 폰트 설정 적용
        apply_korean_style()
        
        # 세션 상태 초기화
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'universe_loaded' not in st.session_state:
            st.session_state.universe_loaded = False
        if 'factor_generated' not in st.session_state:
            st.session_state.factor_generated = False
        
        # 탭별 독립적인 상태 관리
        if 'tab_states' not in st.session_state:
            st.session_state.tab_states = {
                'statistical': {
                    'selected_factors': ['momentum', 'reversal', 'volatility'],
                    'combine_method': 'IC 기반 동적 가중치',
                    'ic_lookback': 60,
                    'factor_generated': False,
                    'results': None
                },
                'deep_learning': {
                    'model_type': 'mlp',
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'window_size': 20,
                    'hidden_dim_1': 128,
                    'hidden_dim_2': 64,
                    'dropout_rate': 0.2,
                    'factor_generated': False,
                    'results': None
                },
                'formula': {
                    'formula_type': 'template',
                    'selected_template': 'momentum',
                    'custom_formula': '',
                    'factor_generated': False,
                    'results': None
                },
                'zoo': {
                    'selected_factor': None,
                    'analysis_completed': False,
                    'results': None
                },
                'comparison': {
                    'linear_factor': None,
                    'nonlinear_factor': None,
                    'comparison_completed': False,
                    'results': None
                }
            }
    
    def run(self):
        """메인 애플리케이션 실행"""
        self._setup_page()
        self._render_sidebar()
        self._render_main_content()
    
    def _setup_page(self):
        """페이지 설정"""
        st.title("🚀 AlphaFactors: 미국주식 알파 팩터 + Qlib 백테스팅 플랫폼")
        st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 20px;'>
        <p><strong>퀀트 투자 연구를 위한 전문적인 알파 팩터 생성 및 백테스팅 플랫폼</strong></p>
        <p>횡단면 순위 기반 팩터 • 딥러닝 통합 • Qlib 백테스팅 • 팩터 Zoo • Mega-Alpha 신호</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("📋 프로세스 단계")
            
            # 진행 상황 표시
            st.subheader("🎯 현재 진행 상황")
            
            # 단계별 상태 표시
            step1_status = "✅" if st.session_state.get('data_loaded', False) else "⏳"
            step2_status = "✅" if st.session_state.get('factor_generated', False) else "⏳"
            step3_status = "✅" if st.session_state.get('backtest_completed', False) else "⏳"
            
            st.markdown(f"""
            {step1_status} **1단계: 데이터 준비**
            - 투자 유니버스 구성
            - OHLCV 데이터 다운로드
            
            {step2_status} **2단계: 팩터 생성**
            - 알파 팩터 계산
            - IC 기반 성능 분석
            
            {step3_status} **3단계: 백테스팅**
            - Qlib 포트폴리오 분석
            - 리스크 지표 계산
            """)
            
            # 전체 진행률 표시
            completed_steps = sum([
                st.session_state.get('data_loaded', False),
                st.session_state.get('factor_generated', False),
                st.session_state.get('backtest_completed', False)
            ])
            progress = completed_steps / 3
            st.progress(progress)
            st.caption(f"진행률: {progress:.1%}")
            
            st.markdown("---")
            
            # 팩터 설정
            st.header("📊 팩터 설정")
            
            default_factors = st.multiselect(
                "기본 팩터 선택",
                ['momentum', 'reversal', 'volatility', 'volume', 'rsi', 'price_to_ma'],
                default=['momentum', 'reversal', 'volatility'],
                help="생성할 알파 팩터 종류 선택"
            )
            
            ic_lookback = st.slider(
                "IC 계산 기간",
                min_value=20, max_value=120,
                value=60,
                help="Information Coefficient 계산을 위한 과거 기간"
            )
            
            if st.button("팩터 설정 저장", type="secondary"):
                st.session_state.selected_factors = default_factors
                st.session_state.ic_lookback = ic_lookback
                st.success("팩터 설정이 저장되었습니다!")
            
            st.markdown("---")
            
            # 캐시 관리
            st.header("🗂️ 캐시 관리")
            cache_info = self.data_handler.get_cache_info()
            st.metric("캐시 파일 수", cache_info['cache_files'])
            st.metric("캐시 크기", f"{cache_info['total_size']/1024/1024:.1f} MB")
            
            if st.button("캐시 정리"):
                self.data_handler.clear_cache()
                st.rerun()
            
            # 팩터 Zoo 상태
            st.markdown("---")
            st.header("🦁 팩터 Zoo 상태")
            factors = load_factors_from_zoo()
            st.metric("저장된 팩터", len(factors))
            
            if factors:
                factor_types = {}
                for factor_name, factor_data in factors.items():
                    factor_type = factor_data.get('meta', {}).get('weight_mode', 'Unknown')
                    factor_types[factor_type] = factor_types.get(factor_type, 0) + 1
                
                for factor_type, count in factor_types.items():
                    st.caption(f"• {factor_type}: {count}개")
    
    def _render_main_content(self):
        """메인 컨텐츠 렌더링"""
        
        # 사용자 가이드 및 도움말 시스템
        self._render_user_guide()
        
        # 1. 데이터 준비 섹션
        self._render_data_section()
        
        st.header("2. 🎯 알파 팩터 생성")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 통계/기술적 팩터", 
            "🧠 딥러닝 팩터", 
            "📝 공식 기반 팩터", 
            "🦁 팩터 Zoo", 
            "⚡ 선형/비선형 비교"
        ])
        with tab1:
            self._render_statistical_factor_section()
        with tab2:
            self._render_dl_factor_section()
        with tab3:
            self._render_formula_factor_section()
        with tab4:
            self._render_factor_zoo_section()
        with tab5:
            self._render_linear_vs_nonlinear_section()
        
        # 3. Qlib 백테스팅 섹션
        self._render_backtest_section()
        
        # 4. 설명 섹션
        self._render_explanation_section()
    
    def _render_user_guide(self):
        """사용자 가이드 및 도움말 시스템"""
        
        # 진행 상황에 따른 동적 가이드
        current_step = 0
        if st.session_state.get('data_loaded', False):
            current_step = 1
        if st.session_state.get('factor_generated', False):
            current_step = 2
        if st.session_state.get('backtest_completed', False):
            current_step = 3
        
        with st.expander("🚀 AlphaFactors 사용법 가이드", expanded=current_step == 0):
            
            # 프로젝트 소개
            st.markdown("""
            ## 🎯 AlphaFactors란?
            
            **AlphaFactors**는 미국 주식 데이터를 기반으로 다양한 알파 팩터를 생성하고, 
            Qlib과 연동하여 전문적인 포트폴리오 백테스팅을 제공하는 **퀀트 투자 연구 플랫폼**입니다.
            
            ### ✨ 핵심 특징
            - 🎯 **진정한 알파 팩터**: 횡단면 순위 기반 팩터 생성 (실제 퀀트 투자 방식)
            - 🧠 **딥러닝 통합**: MLP, LSTM, Transformer 등 다양한 모델 지원
            - 📊 **전문 백테스팅**: Qlib 기반 리스크 분석 및 성과 평가
            - 🦁 **팩터 Zoo**: 실험 결과 저장/관리/재사용 시스템
            - ⚡ **Mega-Alpha 신호**: 선형/비선형 팩터 동적 조합
            """)
            
            if current_step == 0:
                st.info("🚀 **시작하기**: 아래 단계를 따라 첫 번째 알파 팩터를 만들어보세요!")
                
                # 빠른 시작 가이드
                st.markdown("""
                ### ⚡ 빠른 시작 (5분 튜토리얼)
                
                **1️⃣ 데이터 준비 (1분)**
                - "미국 대형주 (추천)" 선택
                - 시작일: 2022-01-01, 종료일: 2023-12-31
                - "🚀 유니버스 데이터 다운로드" 클릭
                
                **2️⃣ 팩터 생성 (2분)**
                - "📊 통계/기술적 팩터" 탭 선택
                - 기본 팩터: 모멘텀, 반전, 저변동성 선택
                - 결합 방식: "IC 기반 동적 가중치" 선택
                - "🎯 알파 팩터 생성" 클릭
                
                **3️⃣ 백테스팅 (2분)**
                - "Qlib 포트폴리오 백테스팅" 섹션으로 이동
                - "📊 Qlib 백테스팅 실행" 클릭
                - 결과 확인 및 분석
                """)
                
                st.markdown("""
                ### 📋 전체 워크플로우
                
                **1단계: 데이터 준비** ⏳
                - 투자 유니버스 구성 (10개 내외 종목 권장)
                - OHLCV 데이터 다운로드 및 캐싱
                - 데이터 품질 검증 및 시각화
                
                **2단계: 팩터 생성** ⏳
                - 통계/기술적 팩터 또는 딥러닝 팩터 선택
                - IC 기반 동적 가중치 또는 고정 가중치 선택
                - 팩터 성능 분석 및 검증 (IC, ICIR)
                
                **3단계: 백테스팅** ⏳
                - Qlib 기반 포트폴리오 백테스팅
                - 리스크 지표 및 수익률 분석
                - 결과 시각화 및 리포트 생성
                """)
                
                # 기술 스택 설명
                st.markdown("""
                ### 🛠️ 기술 스택 및 아키텍처
                
                **데이터 처리**
                - `pandas`, `numpy`: 데이터 조작 및 수치 계산
                - `yfinance`, `finance-datareader`: 실시간 주식 데이터
                
                **팩터 생성**
                - `scikit-learn`: 기계학습 기반 팩터
                - `torch`: 딥러닝 모델 (MLP, LSTM, Transformer)
                - `scipy`: 통계적 계산 및 최적화
                
                **백테스팅**
                - `pyqlib`: Microsoft Qlib 기반 전문 백테스팅
                - `matplotlib`, `seaborn`: 시각화
                
                **웹 인터페이스**
                - `streamlit`: 대화형 웹 애플리케이션
                - 반응형 UI 및 실시간 데이터 처리
                """)
                
            elif current_step == 1:
                st.success("✅ **1단계 완료**: 데이터 준비가 완료되었습니다!")
                st.info("🎯 **다음 단계**: 알파 팩터를 생성하여 투자 신호를 만들어보세요!")
                
                st.markdown("""
                ### 💡 팩터 선택 가이드
                
                **초보자 추천 조합:**
                - 🟢 **안정형**: 모멘텀 + 반전 + 저변동성 (낮은 리스크)
                - 🟡 **균형형**: 모멘텀 + 거래량 + RSI (중간 리스크)
                - 🔴 **적극형**: 모멘텀 + 반전 + 거래량 (높은 리스크)
                
                **고급 사용자:**
                - 🧠 **딥러닝 팩터**: 비선형 패턴 포착 (LSTM, Transformer)
                - 📝 **공식 기반**: 수학적 공식으로 완전 커스텀 팩터
                - ⚡ **Mega-Alpha**: 선형/비선형 팩터 동적 조합
                
                ### 📊 팩터별 특징
                
                | 팩터 | 예측 기간 | 리스크 | 설명 |
                |------|-----------|--------|------|
                | 모멘텀 | 중기 (1-3개월) | 중간 | 과거 수익률 기반 추세 추종 |
                | 반전 | 단기 (1-2주) | 높음 | 단기 반전 효과 포착 |
                | 저변동성 | 장기 (3-6개월) | 낮음 | 안정적인 종목 선호 |
                | 거래량 | 단기 (1주) | 높음 | 비정상적 거래량 증가 감지 |
                | RSI | 단기 (1주) | 중간 | 기술적 과매도/과매수 |
                | 이동평균 | 중기 (1-2개월) | 중간 | 추세선 대비 상대적 위치 |
                """)
                
            elif current_step == 2:
                st.success("✅ **2단계 완료**: 팩터 생성이 완료되었습니다!")
                st.info("🎯 **다음 단계**: 백테스팅을 통해 실제 투자 성과를 확인해보세요!")
                
                st.markdown("""
                ### 💡 백테스팅 가이드
                
                **권장 설정 (초보자):**
                - 🔄 **리밸런싱 주기**: 월간 (안정적, 거래 비용 절약)
                - 💰 **거래 비용**: 0.1% (현실적인 수준)
                - 📈 **최대 포지션**: 10% (분산 투자 효과)
                - 📊 **전략**: Long-Only (롱 포지션만)
                
                **고급 설정:**
                - 🔄 **리밸런싱**: 주간 (더 적극적, 높은 거래 비용)
                - 💰 **거래 비용**: 0.05% (낮은 비용, 높은 빈도)
                - 📈 **최대 포지션**: 5% (더 극단적 분산)
                - 📊 **전략**: Long-Short (롱/숏 포지션)
                
                ### 📈 성과 지표 해석
                
                | 지표 | 양호 | 우수 | 설명 |
                |------|------|------|------|
                | Sharpe Ratio | > 1.0 | > 1.5 | 위험 대비 수익률 |
                | IC | > 0.02 | > 0.05 | 팩터 예측력 |
                | ICIR | > 0.5 | > 1.0 | 예측력 안정성 |
                | 최대 낙폭 | < 20% | < 10% | 최대 손실 폭 |
                | Calmar Ratio | > 0.5 | > 1.0 | 수익률/최대낙폭 |
                """)
                
            elif current_step == 3:
                st.success("✅ **모든 단계 완료**: 전체 워크플로우가 완료되었습니다!")
                st.info("🎯 **다음 단계**: 팩터 Zoo에서 다양한 실험을 해보거나, 선형/비선형 비교를 통해 더 나은 조합을 찾아보세요!")
                
                st.markdown("""
                ### 💡 고급 분석 가이드
                
                **🦁 팩터 Zoo 활용법:**
                - 📁 **저장**: 생성된 팩터는 자동으로 팩터 Zoo에 저장
                - 🔍 **분석**: 저장된 팩터의 상세 성능 분석
                - 🔄 **재사용**: 이전 실험 결과를 새로운 분석에 활용
                - 🗑️ **정리**: 불필요한 팩터 삭제로 저장공간 관리
                
                **성과 비교:**
                - 📊 **Rolling IC/ICIR**: 시계열별 예측력 변화 분석
                - 📈 **백테스트 결과**: 누적 수익률, Sharpe, 최대낙폭 비교
                - 🎯 **파라미터 최적화**: 다양한 설정으로 성과 개선
                
                **⚡ 선형/비선형 비교 및 Mega-Alpha 신호:**
                - 📊 **선형 팩터**: 통계적/기술적 지표 (해석 가능)
                - 🧠 **비선형 팩터**: 딥러닝 모델 (복잡한 패턴 포착)
                - ⚖️ **성과 비교**: IC, ICIR, 백테스트 결과 병렬 비교
                - 🔄 **동적 조합**: 두 팩터를 IC 기반으로 동적 가중치 조합
                - 📈 **성능 향상**: 개별 팩터 대비 우수한 성과 기대
                - 🎯 **실시간 분석**: 즉시 성과 분석 및 시각화
                """)
            
            # 공통 도움말
            st.markdown("---")
            st.markdown("""
            ### 🔧 문제 해결 가이드
            
            **데이터 관련 문제:**
            - ❌ **데이터 다운로드 실패**: 네트워크 연결 확인, 티커명 재확인
            - ❌ **캐시 오류**: 사이드바에서 "캐시 정리" 버튼 클릭
            - ❌ **날짜 범위 오류**: 시작일 < 종료일 확인
            
            **팩터 생성 문제:**
            - ❌ **팩터 생성 오류**: 데이터 품질 확인, 파라미터 조정
            - ❌ **IC 계산 실패**: 충분한 데이터 확보 (최소 60일)
            - ❌ **메모리 부족**: 종목 수 줄이기 (10개 이하 권장)
            
            **백테스팅 문제:**
            - ❌ **Qlib 오류**: Qlib 데이터셋 설치 확인
            - ❌ **팩터 형식 오류**: 팩터가 올바른 형식인지 확인
            - ❌ **성과 지표 NaN**: 데이터 품질 및 기간 확인
            
            ### 📞 추가 지원
            - 💡 **도움말**: 각 섹션의 도움말(?) 아이콘 클릭
            - 📊 **상세 정보**: expander를 열어 추가 정보 확인
            - 🔄 **새로고침**: 문제 발생 시 페이지 새로고침
            - 📧 **문의**: 에러 메시지와 함께 상세 내용 공유
            """)
    
    def _render_data_section(self):
        """유니버스 구성 섹션 렌더링"""
        st.header("1. 🌐 투자 유니버스 구성")
        
        # 유니버스 선택
        col1, col2 = st.columns(2)
        
        with col1:
            universe_type = st.selectbox(
                "유니버스 타입",
                ["미국 대형주 (추천)", "기술주", "커스텀"],
                help="알파 팩터는 여러 종목에서 계산되어야 의미가 있습니다."
            )
        
        with col2:
            if universe_type == "커스텀":
                custom_tickers = st.text_input(
                    "종목 리스트 (쉼표 구분)",
                    value="AAPL,GOOGL,MSFT,TSLA,AMZN,META,NVDA,NFLX",
                    help="예: AAPL,GOOGL,MSFT,TSLA,AMZN"
                )
            else:
                # 미리 정의된 유니버스
                universe_tickers = {
                    "미국 대형주 (추천)": "AAPL,GOOGL,MSFT,TSLA,AMZN,META,NVDA,NFLX,CRM,ADBE",
                    "기술주": "AAPL,GOOGL,MSFT,NVDA,AMD,INTC,CRM,ORCL,ADBE,NFLX"
                }
                custom_tickers = universe_tickers[universe_type]
                st.text_input("선택된 종목들", value=custom_tickers, disabled=True)
        
        # 날짜 설정
        col3, col4 = st.columns(2)
        with col3:
            start_date = st.date_input(
                "시작일", 
                value=pd.to_datetime("2022-01-01")
            )
        with col4:
            end_date = st.date_input(
                "종료일", 
                value=pd.to_datetime("2023-12-31")
            )
        
        tickers_list = [t.strip().upper() for t in custom_tickers.split(",")]
        st.info(f"선택된 유니버스: {', '.join(tickers_list)} ({len(tickers_list)}개 종목)")
        
        if st.button("🚀 유니버스 데이터 다운로드", type="primary"):
            try:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                
                # 유니버스 데이터 다운로드
                universe_data, volume_data = self.data_handler.download_universe_data(
                    tickers_list, start_ts, end_ts
                )
                
                if universe_data is not None and not universe_data.empty:
                    st.session_state.universe_data = universe_data
                    st.session_state.volume_data = volume_data
                    st.session_state.tickers_list = tickers_list
                    st.session_state.start_date = start_date
                    st.session_state.end_date = end_date
                    st.session_state.universe_loaded = True
                    st.session_state.data_loaded = True  # 진행 상황 업데이트
                    
                    # 유니버스 정보 표시
                    st.success(f"✅ {len(universe_data.columns)}개 종목 데이터 다운로드 완료")
                    
                    # 데이터 요약
                    show_dataframe_info(universe_data, "유니버스 데이터 정보")
                    
                    # 최근 가격 차트
                    st.subheader("📈 유니버스 주가 차트 (정규화)")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 정규화된 가격 (시작점을 100으로)
                    normalized_data = universe_data / universe_data.iloc[0] * 100
                    
                    for ticker in normalized_data.columns:
                        normalized_data[ticker].plot(ax=ax, label=ticker, alpha=0.8)
                    
                    ax.set_title("유니버스 종목별 정규화 주가 (시작점=100)", fontsize=14)
                    ax.set_ylabel("정규화 가격", fontsize=12)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                else:
                    st.error("유니버스 데이터 다운로드에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"유니버스 구성 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    def _render_statistical_factor_section(self):
        """
        올바른 알파 팩터 생성 섹션 렌더링
        - 팩터 선택, 파라미터 입력, 가중치 방식, 팩터 생성, 성능 분석, 팩터 Zoo 저장, 결과 시각화
        - UI/UX(위젯, 출력 포맷 등)는 절대 변경하지 않음
        """
        if not st.session_state.get('universe_loaded', False):
            st.warning("먼저 투자 유니버스를 구성하세요.")
            return

        tab_state = st.session_state.tab_states['statistical']

        # --- 내부 유틸 함수 분리 ---
        def get_factor_names_ko() -> dict:
            """팩터명 한글 매핑 반환"""
            return {
                'momentum': '모멘텀',
                'reversal': '반전',
                'volatility': '저변동성',
                'volume': '거래량',
                'rsi': 'RSI',
                'price_to_ma': '이동평균 대비 가격',
                'bollinger_band': '볼린저 밴드',
                'macd': 'MACD',
                'stochastic': '스토캐스틱',
                'williams_r': 'Williams %R',
                'cci': 'CCI',
                'money_flow': 'Money Flow Index',
                'obv': 'OBV',
                'volume_price_trend': 'VPT',
                'chaikin_money_flow': 'Chaikin Money Flow',
                'force_index': 'Force Index',
                'ease_of_movement': 'Ease of Movement',
                'accumulation_distribution': 'Accumulation/Distribution'
            }

        def make_weights_df(weights: dict, names_ko: dict) -> pd.DataFrame:
            """가중치 dict를 한글 컬럼명으로 DataFrame 변환"""
            return pd.DataFrame.from_dict(
                {names_ko.get(k, k): [v] for k, v in weights.items()},
                orient='index', columns=['가중치']
            )

        def render_factor_param_sliders():
            """고급 팩터 파라미터 슬라이더 UI 렌더링"""
            with st.expander("고급 팩터 파라미터", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    self.config.factor.bollinger_band_period = st.slider(
                        "볼린저 밴드 기간", 10, 50, self.config.factor.bollinger_band_period,
                        key="statistical_bollinger_period"
                    )
                    self.config.factor.macd_fast = st.slider(
                        "MACD 빠른선", 5, 20, self.config.factor.macd_fast,
                        key="statistical_macd_fast"
                    )
                    self.config.factor.stochastic_k_period = st.slider(
                        "스토캐스틱 K기간", 5, 30, self.config.factor.stochastic_k_period,
                        key="statistical_stochastic_k"
                    )
                with col_b:
                    self.config.factor.williams_r_period = st.slider(
                        "Williams %R 기간", 5, 30, self.config.factor.williams_r_period,
                        key="statistical_williams_r"
                    )
                    self.config.factor.cci_period = st.slider(
                        "CCI 기간", 10, 50, self.config.factor.cci_period,
                        key="statistical_cci"
                    )
                    self.config.factor.money_flow_period = st.slider(
                        "Money Flow 기간", 5, 30, self.config.factor.money_flow_period,
                        key="statistical_money_flow"
                    )

        def render_fixed_weight_inputs(factor_types: list, names_ko: dict) -> dict:
            """고정 가중치 입력 UI 렌더링 및 값 반환"""
            fixed_weights = {}
            st.markdown("#### 팩터별 가중치 입력 (합계 0 또는 1이어도 자동 정규화)")
            cols = st.columns(len(factor_types))
            for i, factor in enumerate(factor_types):
                with cols[i]:
                    fixed_weights[factor] = st.number_input(
                        f"{names_ko.get(factor, factor)} 가중치",
                        value=1.0, step=0.1, format="%.2f",
                        key=f"statistical_weight_{factor}"
                    )
            return fixed_weights

        # --- 팩터 선택 UI ---
        st.subheader("📊 팩터 타입 선택")
        col1, col2 = st.columns(2)
        with col1:
            basic_factors = st.multiselect(
                "기본 팩터",
                ['momentum', 'reversal', 'volatility', 'volume', 'rsi', 'price_to_ma'],
                default=tab_state.get('selected_factors', ['momentum', 'reversal', 'volatility']),
                key="statistical_basic_factors",
                help="전통적인 기술적 지표 기반 팩터"
            )
            advanced_factors = st.multiselect(
                "고급 기술적 지표",
                ['bollinger_band', 'macd', 'stochastic', 'williams_r', 'cci'],
                key="statistical_advanced_factors",
                help="고급 기술적 분석 지표 기반 팩터"
            )
            volume_factors = st.multiselect(
                "거래량 기반 지표",
                ['money_flow', 'obv', 'volume_price_trend', 'chaikin_money_flow', 'force_index', 'ease_of_movement', 'accumulation_distribution'],
                key="statistical_volume_factors",
                help="거래량과 가격의 관계를 분석하는 팩터"
            )
        with col2:
            ic_lookback = st.slider(
                "IC 계산 기간 (일)",
                min_value=20, max_value=120,
                value=tab_state.get('ic_lookback', 60),
                key="statistical_ic_lookback",
                help="Information Coefficient 계산을 위한 과거 기간"
            )
            st.markdown("**⚙️ 팩터 파라미터 설정**")
            render_factor_param_sliders()

        factor_types = basic_factors + advanced_factors + volume_factors
        if not factor_types:
            st.error("최소 하나의 팩터를 선택해주세요.")
            return

        factor_names_ko = get_factor_names_ko()
        selected_names = [factor_names_ko.get(f, f) for f in factor_types]
        st.info(f"선택된 팩터: {', '.join(selected_names)} (총 {len(factor_types)}개)")

        # 카테고리별 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("기본 팩터", len(basic_factors))
        with col2:
            st.metric("고급 기술적 지표", len(advanced_factors))
        with col3:
            st.metric("거래량 기반 지표", len(volume_factors))

        # --- 가중치 방식 선택 및 입력 ---
        st.subheader("⚖️ 팩터 가중치 방식 선택")
        weight_mode = st.radio(
            "팩터 결합 가중치 방식",
            ["IC 기반 동적 가중치", "고정 가중치 직접 입력"],
            index=0 if tab_state.get('combine_method') == "IC 기반 동적 가중치" else 1,
            key="statistical_weight_mode",
            help="IC 기반: 각 팩터의 예측력(IC)에 따라 자동 가중치 부여 / 고정: 사용자가 직접 가중치 입력"
        )
        fixed_weights = {}
        if weight_mode == "고정 가중치 직접 입력" and len(factor_types) > 1:
            fixed_weights = render_fixed_weight_inputs(factor_types, factor_names_ko)

        # --- 팩터 생성 버튼 및 로직 ---
        if st.button("🚀 알파 팩터 생성", type="primary", key="statistical_generate"):
            try:
                tab_state['selected_factors'] = factor_types
                tab_state['combine_method'] = weight_mode
                tab_state['ic_lookback'] = ic_lookback
                universe_data = st.session_state.universe_data
                volume_data = st.session_state.get('volume_data')
                with st.spinner("알파 팩터 계산 중..."):
                    factors_dict = self.alpha_engine.calculate_all_factors(
                        universe_data, volume_data, factor_types
                    )
                    if not factors_dict:
                        st.error("팩터 생성에 실패했습니다.")
                        return
                    st.success(f"✅ {len(factors_dict)}개 팩터 생성 완료")
                    future_returns = universe_data.pct_change().shift(-1)
                    # 팩터 결합 방식 분기
                    if len(factors_dict) > 1:
                        if weight_mode == "고정 가중치 직접 입력":
                            combined_factor, used_weights = self.alpha_engine.combine_factors_fixed_weights(
                                factors_dict, fixed_weights
                            )
                            st.info("사용자 입력 고정 가중치로 팩터를 결합했습니다.")
                            st.subheader("⚖️ 적용된 팩터별 가중치")
                            st.dataframe(make_weights_df(used_weights, factor_names_ko), use_container_width=True)
                        else:
                            combined_factor, ic_weights = self.alpha_engine.combine_factors_ic_weighted(
                                factors_dict, future_returns, ic_lookback
                            )
                            st.info("여러 팩터를 IC 가중 방식으로 결합했습니다.")
                            st.subheader("⚖️ IC 기반 팩터 가중치")
                            st.dataframe(make_weights_df(ic_weights, factor_names_ko), use_container_width=True)
                    else:
                        combined_factor = list(factors_dict.values())[0]
                        ic_weights = {list(factors_dict.keys())[0]: 1.0}
                        st.info("단일 팩터를 사용합니다.")
                    if combined_factor.empty:
                        st.error("결합된 팩터가 비어있습니다.")
                        return
                    performance = self.alpha_engine.analyze_factor_performance(
                        combined_factor, future_returns
                    )
                    qlib_factor = self.alpha_engine.convert_to_qlib_format(combined_factor)
                    st.session_state.custom_factor = qlib_factor
                    st.session_state.combined_factor_df = combined_factor
                    st.session_state.individual_factors = factors_dict
                    st.session_state.factor_performance = performance
                    st.session_state.factor_generated = True
                    st.session_state.ic_weights = ic_weights if weight_mode == "IC 기반 동적 가중치" else used_weights
                    st.session_state.factor_generated = True
                    st.success("✅ 올바른 알파 팩터 생성 완료!")
                    # --- 팩터 Zoo 자동 저장 ---
                    rolling_ic = self.alpha_engine.calculate_rolling_ic(combined_factor, future_returns, window=20)
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    factor_label = f"{now_str}_" + "_".join(factor_types) + ("_동적" if weight_mode=="IC 기반 동적 가중치" else "_고정")
                    meta = {
                        'factor_name': factor_label,
                        'created_at': now_str,
                        'factor_types': factor_types,
                        'weight_mode': weight_mode,
                        'weights': ic_weights if weight_mode=="IC 기반 동적 가중치" else used_weights,
                        'performance': performance,
                        'params': {k: getattr(self.config.factor, k) for k in dir(self.config.factor) if not k.startswith('__') and not callable(getattr(self.config.factor, k))},
                        'rolling_ic': rolling_ic
                    }
                    save_factor_to_zoo(factor_label, {'meta': meta, 'factor': combined_factor})
                    st.info(f"[팩터 Zoo]에 자동 저장됨: {factor_label}")
                self._display_factor_analysis(
                    factors_dict, combined_factor, performance, factor_names_ko
                )
            except Exception as e:
                st.error(f"알파 팩터 생성 중 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    def _display_factor_analysis(self, factors_dict: Dict[str, pd.DataFrame], 
                               combined_factor: pd.DataFrame, 
                               performance: Dict[str, float],
                               factor_names_ko: Dict[str, str]):
        """팩터 분석 결과 표시"""
        
        st.subheader("📈 팩터 분석 결과")
        
        # 성능 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("평균 IC", f"{performance.get('mean_ic', 0):.4f}")
        with col2:
            st.metric("ICIR", f"{performance.get('icir', 0):.4f}")
        with col3:
            st.metric("팩터 분산", f"{performance.get('factor_spread', 0):.4f}")
        with col4:
            st.metric("데이터 포인트", f"{len(st.session_state.custom_factor):,}")
        
        # 개별 팩터들 시각화 (최대 6개)
        if len(factors_dict) > 1:
            st.subheader("📊 개별 팩터 시계열")
            
            n_factors = min(len(factors_dict), 6)
            cols = 2
            rows = (n_factors + 1) // 2
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (factor_name, factor_data) in enumerate(list(factors_dict.items())[:n_factors]):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                # 첫 번째 종목의 팩터 값 시계열
                first_ticker = factor_data.columns[0]
                factor_data[first_ticker].dropna().plot(
                    ax=ax, 
                    title=f"{factor_names_ko.get(factor_name, factor_name)} ({first_ticker})",
                    color='steelblue',
                    alpha=0.8
                )
                ax.set_title(f"{factor_names_ko.get(factor_name, factor_name)} ({first_ticker})", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylabel("팩터 점수 (0-1)", fontsize=10)
            
            # 빈 subplot 숨기기
            for i in range(n_factors, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # 결합된 팩터 히트맵
        st.subheader("🔥 최종 결합 팩터 히트맵")
        
        if not combined_factor.empty and len(combined_factor.columns) <= 15:
            # 최근 30일 데이터만 표시
            recent_data = combined_factor.tail(min(30, len(combined_factor)))
            
            if not recent_data.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                import seaborn as sns
                sns.heatmap(
                    recent_data.T, 
                    ax=ax, 
                    cmap='RdYlBu_r', 
                    center=0.5,
                    cbar_kws={'label': '팩터 점수 (0-1)'},
                    xticklabels=False  # x축 라벨 숨기기 (너무 많아서)
                )
                ax.set_title(f"최근 {len(recent_data)}일 종목별 알파 팩터 점수", fontsize=14)
                ax.set_xlabel("날짜", fontsize=12)
                ax.set_ylabel("종목", fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        
        # 팩터 설명
        with st.expander("💡 생성된 팩터의 의미", expanded=False):
            st.markdown(f"""
            ### 📈 팩터별 의미
            
            **모멘텀 팩터**: 과거 {self.config.factor.momentum_lookback}일 수익률이 높은 종목에 높은 점수  
            **반전 팩터**: 최근 {self.config.factor.reversal_lookback}일 하락한 종목에 높은 점수 (단기 반등 기대)  
            **저변동성 팩터**: 변동성이 낮은 안정적인 종목에 높은 점수  
            **거래량 팩터**: 평균 대비 거래량이 급증한 종목에 높은 점수  
            **RSI 팩터**: RSI 과매도 구간(30 이하) 종목에 높은 점수  
            **이동평균 대비 가격**: {self.config.factor.ma_period}일 이동평균선 위에 있는 종목에 높은 점수
            
            ### 🎯 올바른 알파 팩터의 특징
            
            1. **횡단면 순위**: 각 시점에서 종목들을 0~1 백분위수로 순위화
            2. **의미 있는 분산**: 종목별로 서로 다른 값 → 포트폴리오 구성 가능
            3. **예측력**: IC(Information Coefficient) > 0 이면 미래 수익률과 양의 상관관계
            4. **안정성**: ICIR(IC/IC의 표준편차) > 1 이면 안정적인 예측력
            
            ### 📊 성능 지표 해석
            
            - **평균 IC**: {performance.get('mean_ic', 0):.4f} ({'양호' if performance.get('mean_ic', 0) > 0.05 else '보통' if performance.get('mean_ic', 0) > 0.02 else '개선 필요'})
            - **ICIR**: {performance.get('icir', 0):.4f} ({'우수' if performance.get('icir', 0) > 1 else '양호' if performance.get('icir', 0) > 0.5 else '개선 필요'})
            - **팩터 분산**: {performance.get('factor_spread', 0):.4f} (종목 간 차별화 정도)
            """)
        
        # rolling IC/ICIR 시계열 시각화 (조합 팩터)
        st.subheader("📉 결합 팩터의 Rolling IC/ICIR 시계열")
        rolling_window = 20  # 기본값, 필요시 UI에서 조정 가능
        universe_data = st.session_state.universe_data
        future_returns = universe_data.pct_change().shift(-1)
        rolling_result = self.alpha_engine.calculate_rolling_ic(combined_factor, future_returns, window=rolling_window)
        if rolling_result['ic']:
            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax1.plot(rolling_result['dates'], rolling_result['ic'], label='Rolling IC', color='tab:blue')
            ax1.set_ylabel('IC', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()
            ax2.plot(rolling_result['dates'], rolling_result['icir'], label='Rolling ICIR', color='tab:red', alpha=0.6)
            ax2.set_ylabel('ICIR', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax1.set_title(f'결합 팩터의 {rolling_window}일 Rolling IC/ICIR', fontsize=13)
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # 개별 팩터 rolling IC/ICIR (최대 3개만)
        if len(factors_dict) > 1:
            st.subheader("📉 개별 팩터의 Rolling IC/ICIR (최대 3개)")
            shown = 0
            for fname, fdata in list(factors_dict.items())[:3]:
                rolling = self.alpha_engine.calculate_rolling_ic(fdata, future_returns, window=rolling_window)
                if rolling['ic']:
                    fig, ax1 = plt.subplots(figsize=(12, 3))
                    ax1.plot(rolling['dates'], rolling['ic'], label='Rolling IC', color='tab:blue')
                    ax1.set_ylabel('IC', color='tab:blue')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')
                    ax2 = ax1.twinx()
                    ax2.plot(rolling['dates'], rolling['icir'], label='Rolling ICIR', color='tab:red', alpha=0.6)
                    ax2.set_ylabel('ICIR', color='tab:red')
                    ax2.tick_params(axis='y', labelcolor='tab:red')
                    ax1.set_title(f'{factor_names_ko.get(fname, fname)}의 {rolling_window}일 Rolling IC/ICIR', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    shown += 1
            if shown == 0:
                st.info("개별 팩터의 rolling IC/ICIR 시계열을 계산할 수 없습니다.")
        
        # 고도화된 AI 해석 적용 버튼
        if 'use_llm_analysis' not in st.session_state:
            st.session_state['use_llm_analysis'] = False
        if st.button('고도화된 AI 해석 적용', key='llm_factor_analysis'):
            st.session_state['use_llm_analysis'] = True
        
        # AI 해석 결과를 expander로 표시
        with st.expander("🤖 AI 해석 결과", expanded=True):
            st.info(analyze_factor_performance_text(performance, llm_api_key=None if not st.session_state['use_llm_analysis'] else 'env'))
    
    def _render_backtest_section(self):
        """백테스팅 섹션 렌더링"""
        st.header("3. 📊 Qlib 포트폴리오 백테스팅")
        
        if not st.session_state.get('factor_generated', False):
            st.warning("먼저 알파 팩터를 생성하세요.")
            return
        
        # 백테스팅 방법 선택
        st.subheader("🔧 백테스팅 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_method = st.selectbox(
                "백테스팅 방법",
                ["상세 분석 백테스터 (추천)", "Qlib 백테스팅"],
                help="'상세 분석 백테스터'는 직접 구현한 백테스터로, 상세한 성과 분석과 시각화를 제공합니다. 'Qlib 백테스팅'은 Qlib의 표준 리스크 분석에 유용합니다."
            )
        
        with col2:
            strategy_type = st.selectbox(
                "전략 유형",
                ["Long Only (매수 전용)", "Long-Short (롱숏)"],
                help="Long Only는 상위 종목만 매수, Long-Short는 상위 매수/하위 매도"
            )
        
        # 추가 설정
        col3, col4, col5 = st.columns(3)
        
        with col3:
            rebalance_freq = st.selectbox(
                "리밸런싱 주기",
                ["daily", "weekly", "monthly"],
                index=0,
                help="포트폴리오 재조정 빈도"
            )
        
        with col4:
            transaction_cost = st.slider(
                "거래비용 (bps)",
                min_value=0, max_value=50, value=10,
                help="거래 시 발생하는 비용 (1bps = 0.01%)"
            )
        
        with col5:
            max_position = st.slider(
                "최대 종목 비중 (%)",
                min_value=5, max_value=50, value=10,
                help="단일 종목의 최대 포트폴리오 비중"
            ) / 100
        
        # 백테스팅 실행
        if st.button("🚀 백테스팅 실행", type="primary"):
            
            if backtest_method == "상세 분석 백테스터 (추천)":
                self._run_custom_backtest(
                    strategy_type == "Long Only (매수 전용)",
                    rebalance_freq, transaction_cost, max_position
                )
            else:
                self._run_qlib_backtest()
    
    def _run_custom_backtest(self, long_only: bool, rebalance_freq: str, 
                                  transaction_cost: float, max_position: float):
        """사용자 정의 상세 분석 백테스팅 실행"""
        
        try:
            universe_data = st.session_state.universe_data
            volume_data = st.session_state.get('volume_data')
            combined_factor_df = st.session_state.combined_factor_df
            individual_factors = st.session_state.individual_factors
            
            # 포트폴리오 백테스터 초기화
            backtester = PortfolioBacktester(universe_data, volume_data)
            
            # 단일 통합 팩터 백테스팅
            st.subheader("🎯 통합 알파 팩터 백테스팅")
            
            with st.spinner("통합 팩터 백테스팅 실행 중..."):
                result = backtester.run_backtest(
                    combined_factor_df,
                    method='rank',
                    long_only=long_only,
                    rebalance_freq=rebalance_freq,
                    transaction_cost_bps=transaction_cost,
                    max_position=max_position
                )
            
            if result:
                # 결과 시각화
                fig = backtester.plot_results(result, "통합 알파 팩터 백테스팅 결과")
                st.pyplot(fig)
                plt.close(fig)
                
                # 성과 리포트
                st.subheader("📈 성과 리포트")
                report_df = backtester.create_performance_report(result)
                st.dataframe(report_df, use_container_width=True)
                
                # AI 해석 결과를 expander로 표시
                with st.expander("🤖 AI 해석 결과", expanded=True):
                    st.info(analyze_backtest_performance_text(result.get('performance_metrics', {}), llm_api_key=None if not st.session_state['use_llm_analysis'] else 'env'))
                
                # 세션에 결과 저장
                st.session_state.dl_backtest_results = result
                
                # 개별 팩터 비교 옵션
                if len(individual_factors) > 1:
                    st.subheader("🔄 개별 팩터 성과 비교")
                    
                    if st.button("개별 팩터들과 성과 비교", type="secondary"):
                        self._compare_individual_factors(
                            individual_factors, long_only, rebalance_freq, 
                            transaction_cost, max_position
                        )
                
                # 결과 다운로드 옵션
                if st.button("📥 백테스팅 결과 다운로드"):
                    self._export_backtest_results(result)
            
        except Exception as e:
            st.error(f"상세 분석 백테스팅 실행 중 오류: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _compare_individual_factors(self, individual_factors: Dict, long_only: bool,
                                  rebalance_freq: str, transaction_cost: float, max_position: float):
        """개별 팩터들 성과 비교"""
        
        universe_data = st.session_state.universe_data
        volume_data = st.session_state.get('volume_data')
        
        # 확장된 팩터명 매핑
        factor_names_ko = {
            'momentum': '모멘텀',
            'reversal': '반전',
            'volatility': '저변동성',
            'volume': '거래량',
            'rsi': 'RSI',
            'price_to_ma': '이동평균 대비 가격',
            'bollinger_band': '볼린저 밴드',
            'macd': 'MACD',
            'stochastic': '스토캐스틱',
            'williams_r': 'Williams %R',
            'cci': 'CCI',
            'money_flow': 'Money Flow Index',
            'aroon': 'Aroon',
            'obv': 'OBV',
            'volume_price_trend': 'VPT',
            'chaikin_money_flow': 'Chaikin Money Flow',
            'force_index': 'Force Index',
            'ease_of_movement': 'Ease of Movement',
            'accumulation_distribution': 'Accumulation/Distribution',
            'dl_factor': '딥러닝 팩터'
        }
        
        factor_backtester = FactorBacktester(universe_data, volume_data)
        
        with st.spinner("개별 팩터들 백테스팅 비교 중..."):
            comparison_results = factor_backtester.compare_factors(
                individual_factors, factor_names_ko
            )
        
        if comparison_results:
            st.session_state.factor_comparison_results = comparison_results
            st.success("✅ 개별 팩터 성과 비교 완료!")
            
            # 추가 분석 및 시각화
            self._display_detailed_factor_comparison(comparison_results)
        else:
            st.error("❌ 팩터 비교에 실패했습니다.")
    
    def _display_detailed_factor_comparison(self, comparison_results: Dict[str, Dict]):
        """상세한 팩터 비교 결과 표시"""
        
        st.subheader("🔍 상세 팩터 분석")
        
        # 1. 팩터별 성과 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            best_factor = max(comparison_results.items(), 
                            key=lambda x: x[1]['performance_metrics']['sharpe_ratio'])
            st.metric("최고 샤프 비율", f"{best_factor[1]['performance_metrics']['sharpe_ratio']:.3f}", 
                     f"({best_factor[0]})")
        
        with col2:
            best_return = max(comparison_results.items(), 
                            key=lambda x: x[1]['performance_metrics']['annualized_return'])
            st.metric("최고 연간 수익률", f"{best_return[1]['performance_metrics']['annualized_return']:.2%}", 
                     f"({best_return[0]})")
        
        with col3:
            best_win_rate = max(comparison_results.items(), 
                              key=lambda x: x[1]['performance_metrics']['win_rate'])
            st.metric("최고 승률", f"{best_win_rate[1]['performance_metrics']['win_rate']:.2%}", 
                     f"({best_win_rate[0]})")
        
        # 2. 팩터별 상세 성과 테이블
        st.subheader("📊 팩터별 상세 성과 지표")
        
        detailed_data = []
        for factor_name, result in comparison_results.items():
            metrics = result['performance_metrics']
            detailed_data.append({
                '팩터명': factor_name,
                '총 수익률': f"{metrics['total_return']:.2%}",
                '연간 수익률': f"{metrics['annualized_return']:.2%}",
                '연간 변동성': f"{metrics['annualized_volatility']:.2%}",
                '샤프 비율': f"{metrics['sharpe_ratio']:.3f}",
                '최대 손실폭': f"{metrics['max_drawdown']:.2%}",
                '승률': f"{metrics['win_rate']:.2%}",
                '정보 비율': f"{metrics['information_ratio']:.3f}",
                '칼마 비율': f"{metrics['calmar_ratio']:.3f}",
                '벤치마크 대비 초과수익': f"{metrics['excess_return']:.2%}"
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # 샤프 비율 기준으로 정렬
        detailed_df['샤프 비율_정렬용'] = detailed_df['샤프 비율'].str.rstrip('%').astype(float)
        detailed_df = detailed_df.sort_values('샤프 비율_정렬용', ascending=False)
        detailed_df = detailed_df.drop('샤프 비율_정렬용', axis=1)
        
        st.dataframe(detailed_df, use_container_width=True)
        
        # 3. 팩터별 월별 수익률 히트맵
        st.subheader("📅 팩터별 월별 수익률")
        
        # 월별 수익률 계산
        monthly_returns_data = {}
        for factor_name, result in comparison_results.items():
            monthly_returns = result['portfolio_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_data[factor_name] = monthly_returns
        
        # 히트맵 생성
        if monthly_returns_data:
            monthly_df = pd.DataFrame(monthly_returns_data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(monthly_df.T, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('팩터별 월별 수익률 히트맵', fontsize=14)
            ax.set_xlabel('월', fontsize=12)
            ax.set_ylabel('팩터', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # 4. 팩터별 상관관계 분석
        st.subheader("🔗 팩터별 상관관계 분석")
        
        # 수익률 상관관계 계산
        returns_data = {}
        for factor_name, result in comparison_results.items():
            returns_data[factor_name] = result['portfolio_returns']
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('팩터별 수익률 상관관계', fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # 5. 팩터별 드로우다운 비교
        st.subheader("📉 팩터별 드로우다운 비교")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for factor_name, result in comparison_results.items():
            cumulative = result['cumulative_returns']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            drawdown.plot(ax=ax, label=factor_name, alpha=0.7)
        
        ax.set_title('팩터별 드로우다운 비교', fontsize=14)
        ax.set_ylabel('드로우다운 (%)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # 6. 팩터별 수익률 분포 비교
        st.subheader("📊 팩터별 수익률 분포 비교")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (factor_name, result) in enumerate(comparison_results.items()):
            if i < 4:  # 최대 4개 팩터만 표시
                returns = result['portfolio_returns']
                axes[i].hist(returns, bins=30, alpha=0.7, density=True)
                axes[i].axvline(returns.mean(), color='red', linestyle='--', 
                               label=f'평균: {returns.mean():.4f}')
                axes[i].set_title(f'{factor_name} 수익률 분포', fontsize=12)
                axes[i].set_xlabel('일별 수익률', fontsize=10)
                axes[i].set_ylabel('확률 밀도', fontsize=10)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 빈 서브플롯 숨기기
        for i in range(len(comparison_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # 7. 팩터별 성과 요약 및 권장사항
        st.subheader("💡 팩터별 성과 요약 및 권장사항")
        
        # 최고 성과 팩터들
        best_sharpe = max(comparison_results.items(), 
                         key=lambda x: x[1]['performance_metrics']['sharpe_ratio'])
        best_return = max(comparison_results.items(), 
                         key=lambda x: x[1]['performance_metrics']['annualized_return'])
        best_win_rate = max(comparison_results.items(), 
                           key=lambda x: x[1]['performance_metrics']['win_rate'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 최고 성과 팩터**")
            st.write(f"• **최고 샤프 비율**: {best_sharpe[0]} ({best_sharpe[1]['performance_metrics']['sharpe_ratio']:.3f})")
            st.write(f"• **최고 수익률**: {best_return[0]} ({best_return[1]['performance_metrics']['annualized_return']:.2%})")
            st.write(f"• **최고 승률**: {best_win_rate[0]} ({best_win_rate[1]['performance_metrics']['win_rate']:.2%})")
        
        with col2:
            st.markdown("**💡 투자 권장사항**")
            st.write("• **보수적 투자**: 샤프 비율이 높은 팩터 선택")
            st.write("• **공격적 투자**: 수익률이 높은 팩터 선택")
            st.write("• **안정적 투자**: 승률이 높은 팩터 선택")
            st.write("• **다각화**: 상관관계가 낮은 팩터들 조합")
        
        # 8. 결과 다운로드 옵션
        st.subheader("📥 결과 다운로드")
        
        if st.button("📊 팩터 비교 결과 다운로드"):
            self._export_factor_comparison_results(comparison_results)
    
    def _export_factor_comparison_results(self, comparison_results: Dict[str, Dict]):
        """팩터 비교 결과 내보내기"""
        
        try:
            import io
            
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # 1. 성과 지표 요약
                summary_data = []
                for factor_name, result in comparison_results.items():
                    metrics = result['performance_metrics']
                    summary_data.append({
                        '팩터명': factor_name,
                        '총 수익률': metrics['total_return'],
                        '연간 수익률': metrics['annualized_return'],
                        '연간 변동성': metrics['annualized_volatility'],
                        '샤프 비율': metrics['sharpe_ratio'],
                        '최대 손실폭': metrics['max_drawdown'],
                        '승률': metrics['win_rate'],
                        '정보 비율': metrics['information_ratio'],
                        '칼마 비율': metrics['calmar_ratio'],
                        '벤치마크 대비 초과수익': metrics['excess_return']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='성과 지표 요약', index=False)
                
                # 2. 팩터별 수익률
                returns_data = {}
                for factor_name, result in comparison_results.items():
                    returns_data[factor_name] = result['portfolio_returns']
                
                returns_df = pd.DataFrame(returns_data)
                returns_df.to_excel(writer, sheet_name='팩터별 수익률')
                
                # 3. 팩터별 누적 수익률
                cumulative_data = {}
                for factor_name, result in comparison_results.items():
                    cumulative_data[factor_name] = result['cumulative_returns']
                
                cumulative_df = pd.DataFrame(cumulative_data)
                cumulative_df.to_excel(writer, sheet_name='팩터별 누적 수익률')
                
                # 4. 상관관계 매트릭스
                correlation_matrix = returns_df.corr()
                correlation_matrix.to_excel(writer, sheet_name='상관관계 매트릭스')
            
            buffer.seek(0)
            
            filename = f"factor_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label="📥 Excel 파일 다운로드",
                data=buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"✅ 팩터 비교 결과를 다운로드할 수 있습니다: {filename}")
            
        except Exception as e:
            st.error(f"결과 내보내기 실패: {e}")
    
    def _run_qlib_backtest(self):
        """Qlib 백테스팅 실행 (백업 옵션)"""
        
        if not self.qlib_handler.check_availability():
            st.error("❌ Qlib이 초기화되지 않았습니다.")
            st.info("📝 Qlib 설치 및 데이터 설정이 필요합니다:")
            st.code("""
            # Qlib 설치
            pip install pyqlib
            
            # 미국 데이터셋 다운로드
            python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
            """)
            return
        
        try:
            custom_factor = st.session_state.get('custom_factor')
            instrument = "sp500"  # 기본값
            
            with st.spinner("Qlib 백테스팅 실행 중..."):
                cum_returns, risk_metrics = self.qlib_handler.run_backtest(
                    instrument=instrument,
                    custom_factor=custom_factor,
                    show_details=True
                )
            
            if cum_returns is not None and risk_metrics is not None:
                st.session_state.qlib_backtest_results = {
                    'cum_returns': cum_returns,
                    'risk_metrics': risk_metrics
                }
                st.session_state.backtest_completed = True  # 진행 상황 업데이트
                st.success("✅ Qlib 백테스팅 완료!")
                
        except Exception as e:
            st.error(f"❌ Qlib 백테스팅 실패: {e}")
            st.info("💡 Qlib 백테스팅 대신 상세 분석 백테스터를 사용해보세요. 더 안정적이고 유연합니다.")
    
    def _export_backtest_results(self, result: Dict):
        """백테스팅 결과 내보내기"""
        
        try:
            import io
            
            # Excel 파일로 내보내기
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # 포트폴리오 수익률
                result['portfolio_returns'].to_frame('Portfolio Returns').to_excel(
                    writer, sheet_name='Returns'
                )
                
                # 누적 수익률
                pd.DataFrame({
                    'Portfolio': result['cumulative_returns'],
                    'Benchmark': result['benchmark_cumulative']
                }).to_excel(writer, sheet_name='Cumulative Returns')
                
                # 성과 지표
                metrics_df = pd.DataFrame.from_dict(
                    result['performance_metrics'], orient='index', columns=['Value']
                )
                metrics_df.to_excel(writer, sheet_name='Performance Metrics')
                
                # 포트폴리오 가중치 (최근 30일)
                recent_weights = result['weights'].tail(30)
                recent_weights.to_excel(writer, sheet_name='Recent Weights')
            
            buffer.seek(0)
            
            filename = f"portfolio_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label="📥 Excel 파일 다운로드",
                data=buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"✅ 백테스팅 결과를 다운로드할 수 있습니다: {filename}")
            
        except Exception as e:
            st.error(f"결과 내보내기 실패: {e}")
    
    def _render_explanation_section(self):
        """설명 섹션 렌더링"""
        with st.expander("💡 AlphaFactors 시스템 설명 및 고급 사용법", expanded=False):
            st.markdown(f"""
            ## 🎯 AlphaFactors 시스템 아키텍처
            
            ### ✅ 핵심 특징
            1. **진정한 알파 팩터**: 횡단면 순위 기반 팩터 생성 (실제 퀀트 투자 방식)
            2. **IC 기반 결합**: Information Coefficient로 팩터 가중치 최적화
            3. **멀티 종목 분석**: 여러 종목을 동시에 분석하여 의미 있는 알파 창출
            4. **성능 검증**: IC, ICIR 등 퀀트 업계 표준 지표로 팩터 품질 검증
            5. **딥러닝 통합**: MLP, LSTM, Transformer 등 다양한 모델 지원
            6. **팩터 Zoo**: 실험 결과 저장/관리/재사용 시스템
            7. **Mega-Alpha 신호**: 선형/비선형 팩터 동적 조합
            
            ### 🔍 지원하는 알파 팩터
            
            **📊 통계/기술적 팩터:**
            - **모멘텀**: 과거 수익률 기반 추세 추종 (중기 예측)
            - **반전**: 단기 반전 효과 포착 (단기 예측)
            - **저변동성**: 안정적인 종목 선호 (장기 예측)
            - **거래량**: 비정상적 거래량 증가 감지 (단기 예측)
            - **RSI**: 기술적 과매도/과매수 구간 활용 (단기 예측)
            - **이동평균**: 추세선 대비 상대적 위치 (중기 예측)
            
            **🧠 딥러닝 팩터:**
            - **MLP**: 다층 퍼셉트론 (단기 패턴)
            - **LSTM**: 장단기 메모리 (시계열 패턴)
            - **GRU**: 게이트 순환 유닛 (시계열 패턴)
            - **Transformer**: 어텐션 메커니즘 (복잡한 패턴)
            - **CNN1D**: 1차원 합성곱 (지역적 패턴)
            - **Hybrid**: CNN + LSTM 결합 (복합 패턴)
            
            **📝 공식 기반 팩터:**
            - **템플릿 기반**: 미리 정의된 공식 템플릿
            - **직접 입력**: Python 문법으로 수학적 공식 작성
            - **고급 편집기**: 여러 공식 동시 관리 및 결합
            
            ### 📊 전체 워크플로우
            
            **1단계: 데이터 준비**
            - 투자 유니버스 구성 (10개 내외 종목 권장)
            - OHLCV 데이터 다운로드 및 캐싱
            - 데이터 품질 검증 및 시각화
            
            **2단계: 팩터 생성**
            - 통계/기술적 팩터 또는 딥러닝 팩터 선택
            - IC 기반 동적 가중치 또는 고정 가중치 선택
            - 팩터 성능 분석 및 검증 (IC, ICIR)
            - 팩터 Zoo에 자동 저장
            
            **3단계: 백테스팅**
            - Qlib 기반 포트폴리오 백테스팅
            - 리스크 지표 및 수익률 분석
            - 결과 시각화 및 리포트 생성
            
            **4단계: 고급 분석**
            - 팩터 Zoo에서 다양한 실험
            - 선형/비선형 팩터 비교
            - Mega-Alpha 신호 생성 및 분석
            
            ### 🎯 성과 지표 해석
            
            | 지표 | 양호 | 우수 | 설명 |
            |------|------|------|------|
            | Sharpe Ratio | > 1.0 | > 1.5 | 위험 대비 수익률 |
            | IC | > 0.02 | > 0.05 | 팩터 예측력 |
            | ICIR | > 0.5 | > 1.0 | 예측력 안정성 |
            | 최대 낙폭 | < 20% | < 10% | 최대 손실 폭 |
            | Calmar Ratio | > 0.5 | > 1.0 | 수익률/최대낙폭 |
            
            ### ⚠️ 일반적인 팩터의 문제점
            - ❌ 단일 종목 팩터를 모든 종목에 동일 적용
            - ❌ 횡단면 정보 부재로 순위 차이 없음
            - ❌ 실제 퀀트 투자에서 사용 불가능한 방식
            - ❌ 성과 검증 없이 팩터 사용
            
            ### ✅ AlphaFactors의 장점
            - ✅ 종목별 상대적 순위로 포트폴리오 구성 가능
            - ✅ IC 기반 과학적 팩터 결합
            - ✅ 업계 표준 성능 지표로 검증
            - ✅ 실제 헤지펀드에서 사용하는 정통 방법론
            - ✅ 딥러닝 기반 백테스팅으로 더 유연하고 안정적인 성과 분석
            - ✅ 팩터 Zoo를 통한 실험 결과 관리 및 재사용
            - ✅ Mega-Alpha 신호로 선형/비선형 팩터 동적 조합
            """)

    def _render_dl_factor_section(self):
        """
        딥러닝 팩터 생성 섹션 렌더링
        - 모델 선택, 파라미터 입력, 학습/예측, 팩터 Zoo 저장, 결과 시각화
        - UI/UX(위젯, 출력 포맷 등)는 절대 변경하지 않음
        """
        tab_state = st.session_state.tab_states['deep_learning']

        # --- 내부 유틸 함수 분리 ---
        def get_model_options() -> list:
            """딥러닝 모델 옵션 리스트 반환"""
            return [
                ("mlp", "MLP (다층 퍼셉트론)"),
                ("lstm", "LSTM (장단기 메모리)"),
                ("gru", "GRU (게이트 순환 유닛)"),
                ("transformer", "Transformer"),
                ("cnn1d", "1D CNN"),
                ("hybrid", "하이브리드 (CNN + LSTM)")
            ]

        def get_model_descriptions() -> dict:
            """모델별 설명 반환"""
            return {
                "mlp": "전통적인 다층 퍼셉트론. 간단하고 빠르지만 시계열 특성을 잘 포착하지 못할 수 있습니다.",
                "lstm": "장단기 메모리 네트워크. 시계열 데이터의 장기 의존성을 잘 포착합니다.",
                "gru": "게이트 순환 유닛. LSTM보다 간단하면서도 시계열 특성을 잘 포착합니다.",
                "transformer": "어텐션 메커니즘 기반 모델. 복잡한 패턴을 잘 포착하지만 학습 시간이 오래 걸립니다.",
                "cnn1d": "1차원 합성곱 신경망. 지역적 패턴을 잘 포착합니다.",
                "hybrid": "CNN과 LSTM을 결합한 모델. 지역적 패턴과 시계열 특성을 모두 포착합니다."
            }

        def render_model_recommendations(selected_model: str):
            """모델별 권장 설정 UI 렌더링"""
            if selected_model == "mlp":
                st.write("• 적합한 데이터: 단기 패턴")
                st.write("• 학습 시간: 빠름")
                st.write("• 메모리 사용량: 낮음")
            elif selected_model in ["lstm", "gru"]:
                st.write("• 적합한 데이터: 시계열 패턴")
                st.write("• 학습 시간: 보통")
                st.write("• 메모리 사용량: 보통")
            elif selected_model == "transformer":
                st.write("• 적합한 데이터: 복잡한 패턴")
                st.write("• 학습 시간: 느림")
                st.write("• 메모리 사용량: 높음")
            elif selected_model == "cnn1d":
                st.write("• 적합한 데이터: 지역적 패턴")
                st.write("• 학습 시간: 보통")
                st.write("• 메모리 사용량: 보통")
            elif selected_model == "hybrid":
                st.write("• 적합한 데이터: 복합 패턴")
                st.write("• 학습 시간: 느림")
                st.write("• 메모리 사용량: 높음")

        def render_model_param_sliders(tab_state: dict):
            """모델 파라미터 슬라이더 UI 렌더링"""
            col1, col2, col3 = st.columns(3)
            with col1:
                self.config.model.epochs = st.slider(
                    "Epochs", 10, 100, 
                    tab_state.get('epochs', self.config.model.epochs), 
                    key='dl_epochs'
                )
                self.config.model.learning_rate = st.select_slider(
                    "Learning Rate", 
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=tab_state.get('learning_rate', self.config.model.learning_rate),
                    key='dl_lr'
                )
            with col2:
                self.config.model.window_size = st.slider(
                    "Window Size", 5, 30, 
                    tab_state.get('window_size', self.config.model.window_size), 
                    key='dl_window_size'
                )
                self.config.model.prediction_horizon = st.slider(
                    "Prediction Horizon", 1, 10, 
                    self.config.model.prediction_horizon, 
                    key='dl_prediction_horizon'
                )
                self.config.model.dropout_rate = st.slider(
                    "Dropout Rate", 0.0, 0.5, 
                    tab_state.get('dropout_rate', self.config.model.dropout_rate), 
                    step=0.1, key='dl_dropout'
                )
            with col3:
                self.config.model.hidden_dim_1 = st.slider(
                    "Hidden Dim 1", 32, 256, 
                    tab_state.get('hidden_dim_1', self.config.model.hidden_dim_1), 
                    step=32, key='dl_hidden1'
                )
                self.config.model.hidden_dim_2 = st.slider(
                    "Hidden Dim 2", 16, 128, 
                    tab_state.get('hidden_dim_2', self.config.model.hidden_dim_2), 
                    step=16, key='dl_hidden2'
                )
                self.config.model.batch_size = st.select_slider(
                    "Batch Size",
                    options=[16, 32, 64, 128, 256],
                    value=self.config.model.batch_size,
                    key='dl_batch'
                )

        def render_advanced_model_settings(selected_model: str):
            """고급 모델 설정 UI 렌더링"""
            with st.expander("🔧 고급 모델 설정", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if selected_model in ["lstm", "gru"]:
                        self.config.model.lstm_layers = st.slider("LSTM/GRU Layers", 1, 4, self.config.model.lstm_layers, key="dl_lstm_layers")
                        self.config.model.lstm_bidirectional = st.checkbox("Bidirectional", self.config.model.lstm_bidirectional, key="dl_bidirectional")
                    if selected_model == "transformer":
                        self.config.model.transformer_layers = st.slider("Transformer Layers", 1, 6, self.config.model.transformer_layers, key="dl_transformer_layers")
                        self.config.model.transformer_heads = st.slider("Attention Heads", 4, 16, self.config.model.transformer_heads, step=2, key="dl_attention_heads")
                with col2:
                    self.config.model.early_stopping_patience = st.slider("Early Stopping Patience", 5, 20, self.config.model.early_stopping_patience, key="dl_early_stopping")
                    self.config.model.lr_scheduler_patience = st.slider("LR Scheduler Patience", 3, 10, self.config.model.lr_scheduler_patience, key="dl_lr_scheduler")
                    self.config.model.validation_split = st.slider("Validation Split", 0.1, 0.3, self.config.model.validation_split, step=0.05, key="dl_validation_split")

        # --- 모델 선택 UI ---
        st.subheader("🧠 딥러닝 팩터 생성")
        col1, col2 = st.columns(2)
        model_options = get_model_options()
        model_descriptions = get_model_descriptions()
        current_model_type = tab_state.get('model_type', 'mlp')
        current_index = 0
        for i, (model_code, _) in enumerate(model_options):
            if model_code == current_model_type:
                current_index = i
                break
        with col1:
            model_type = st.selectbox(
                "딥러닝 모델 타입",
                model_options,
                format_func=lambda x: x[1],
                index=current_index,
                key="dl_model_type",
                help="시계열 데이터에 적합한 모델을 선택하세요"
            )
            if model_type[0] != tab_state.get('model_type'):
                tab_state['model_type'] = model_type[0]
                st.session_state.tab_states['deep_learning'] = tab_state
            st.info(f"**모델 설명:** {model_descriptions[model_type[0]]}")
        with col2:
            st.markdown("**📋 모델별 권장 설정**")
            render_model_recommendations(model_type[0])

        # --- 모델 파라미터 입력 ---
        st.subheader("⚙️ 모델 파라미터 설정")
        render_model_param_sliders(tab_state)
        render_advanced_model_settings(model_type[0])

        # --- 학습/예측 및 팩터 Zoo 저장 ---
        if st.button("🧠 딥러닝 모델 학습 및 팩터 생성", type="primary", key="dl_generate"):
            try:
                # 탭 상태 업데이트
                tab_state['model_type'] = model_type[0]
                tab_state['epochs'] = self.config.model.epochs
                tab_state['learning_rate'] = self.config.model.learning_rate
                tab_state['window_size'] = self.config.model.window_size
                tab_state['dropout_rate'] = self.config.model.dropout_rate
                tab_state['hidden_dim_1'] = self.config.model.hidden_dim_1
                tab_state['hidden_dim_2'] = self.config.model.hidden_dim_2
                tickers = st.session_state.tickers_list
                start_date = st.session_state.start_date
                end_date = st.session_state.end_date
                all_X, all_y, all_dates, all_tickers = [], [], [], []
                with st.spinner("학습 데이터 생성 중..."):
                    for ticker in tickers:
                        df = self.data_handler.download_data(ticker, pd.Timestamp(start_date), pd.Timestamp(end_date))
                        if df is not None and len(df) > (self.config.model.window_size + self.config.model.prediction_horizon):
                            X, y, dates = self.data_handler.create_features_targets(
                                df, self.config.model.window_size, self.config.model.prediction_horizon
                            )
                            all_X.append(X)
                            all_y.append(y)
                            all_dates.extend(dates)
                            all_tickers.extend([ticker] * len(dates))
                    if not all_X:
                        st.error("학습 데이터를 생성할 수 없습니다.")
                        return
                    X_train = np.concatenate(all_X)
                    y_train = np.concatenate(all_y)
                st.success(f"✅ 총 {len(X_train)}개의 학습 데이터 생성 완료")
                # 모델 학습
                self.model_trainer = ModelTrainer(self.config.model, model_type=model_type[0])
                trained_model = self.model_trainer.train_model(X_train, y_train)
                if trained_model:
                    st.success("✅ 딥러닝 모델 학습 완료!")
                    with st.expander("📊 모델 정보", expanded=False):
                        model_summary = self.model_trainer.get_model_summary()
                        st.json(model_summary)
                    with st.spinner("딥러닝 팩터 생성 중..."):
                        predictions = self.model_trainer.predict(X_train)
                        if len(predictions) != len(all_dates):
                            st.warning(f"예측값 길이({len(predictions)})와 날짜 길이({len(all_dates)})가 다릅니다. 길이를 맞춰서 처리합니다.")
                            min_length = min(len(predictions), len(all_dates), len(all_tickers))
                            st.info(f"조정된 길이: {min_length} (예측값: {len(predictions)}, 날짜: {len(all_dates)}, 티커: {len(all_tickers)})")
                            predictions = predictions[:min_length]
                            adjusted_dates = all_dates[:min_length]
                            adjusted_tickers = all_tickers[:min_length]
                        else:
                            adjusted_dates = all_dates
                            adjusted_tickers = all_tickers
                        factor_df = pd.DataFrame({
                            'datetime': adjusted_dates,
                            'instrument': adjusted_tickers,
                            'prediction': predictions
                        }).pivot(index='datetime', columns='instrument', values='prediction')
                        ranked_factor = factor_df.rank(axis=1, pct=True)
                        qlib_factor = self.alpha_engine.convert_to_qlib_format(ranked_factor)
                        st.session_state.custom_factor = qlib_factor
                        st.session_state.combined_factor_df = ranked_factor
                        st.session_state.individual_factors = {"dl_factor": ranked_factor}
                        st.session_state.factor_generated = True
                        st.session_state.data_loaded = True
                    st.success("✅ 딥러닝 알파 팩터 생성 완료!")
                    # --- 팩터 Zoo 자동 저장 ---
                    universe_data = st.session_state.universe_data
                    future_returns = universe_data.pct_change().shift(-1)
                    rolling_ic = self.alpha_engine.calculate_rolling_ic(ranked_factor, future_returns, window=20)
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    factor_label = f"{now_str}_딥러닝_{model_type[0].upper()}"
                    meta = {
                        'factor_name': factor_label,
                        'created_at': now_str,
                        'factor_types': ['딥러닝'],
                        'weight_mode': '비선형',
                        'weights': {},
                        'performance': {},
                        'params': {
                            'model_type': model_type[0],
                            'epochs': self.config.model.epochs,
                            'learning_rate': self.config.model.learning_rate,
                            'window_size': self.config.model.window_size,
                            'dropout_rate': self.config.model.dropout_rate,
                            'hidden_dim_1': self.config.model.hidden_dim_1,
                            'hidden_dim_2': self.config.model.hidden_dim_2
                        },
                        'rolling_ic': rolling_ic
                    }
                    save_factor_to_zoo(factor_label, {'meta': meta, 'factor': ranked_factor})
                    st.info(f"[팩터 Zoo]에 자동 저장됨: {factor_label}")
            except Exception as e:
                st.error(f"딥러닝 팩터 생성 중 오류: {e}")
                import traceback
                st.code(traceback.format_exc())

    def _render_factor_zoo_section(self):
        """
        팩터 Zoo(저장소) UI 섹션: 목록, 상세, 불러오기, 삭제 기능
        """
        st.header("🦁 팩터 Zoo (저장된 팩터 관리)")
        
        factors = load_factors_from_zoo()
        if not factors:
            st.info("저장된 팩터가 없습니다. 팩터를 생성 후 저장해보세요!")
            return
        
        # 팩터 목록 요약
        st.subheader("📊 팩터 Zoo 요약")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 팩터 수", len(factors))
        
        with col2:
            linear_count = sum(1 for f in factors.values() 
                             if f.get('meta', {}).get('weight_mode') in ["IC 기반 동적 가중치", "고정 가중치 직접 입력"])
            st.metric("선형 팩터", linear_count)
        
        with col3:
            nonlinear_count = sum(1 for f in factors.values() 
                                if f.get('meta', {}).get('weight_mode') == '비선형')
            st.metric("비선형 팩터", nonlinear_count)
        
        # 팩터 타입별 분류
        factor_types = {}
        for factor_name, factor_data in factors.items():
            factor_type = factor_data.get('meta', {}).get('weight_mode', 'Unknown')
            if factor_type not in factor_types:
                factor_types[factor_type] = []
            factor_types[factor_type].append(factor_name)
        
        # 팩터 선택
        st.subheader("🔍 팩터 선택 및 분석")
        
        # 탭으로 팩터 타입별 분류
        if len(factor_types) > 1:
            type_tabs = st.tabs(list(factor_types.keys()))
            for i, (factor_type, factor_names) in enumerate(factor_types.items()):
                with type_tabs[i]:
                    selected = st.selectbox(
                        f"{factor_type} 팩터 선택",
                        factor_names,
                        key=f"zoo_select_{i}"
                    )
                    if selected:
                        self._display_factor_details(factors[selected], selected)
        else:
            factor_names = list(factors.keys())
            selected = st.selectbox("저장된 팩터 선택", factor_names)
            if selected:
                self._display_factor_details(factors[selected], selected)
    
    def _display_factor_details(self, factor_data: dict, factor_name: str):
        """
        팩터 상세 정보 표시
        - 메타 정보, 성과 지표, 파라미터, 시계열 시각화, AI 해석, 액션 버튼 등 반복되는 부분 함수화
        - UI/UX(위젯, 출력 포맷 등)는 절대 변경하지 않음
        """
        meta = factor_data.get('meta', {})
        # --- 내부 유틸 함수 분리 ---
        def render_meta_info(meta: dict):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📋 기본 정보**")
                st.write(f"• 생성일: {meta.get('created_at', 'N/A')}")
                st.write(f"• 팩터 타입: {meta.get('weight_mode', 'N/A')}")
                st.write(f"• 팩터 종류: {', '.join(meta.get('factor_types', []))}")
            with col2:
                st.markdown("**📊 성과 지표**")
                performance = meta.get('performance', {})
                if performance:
                    for key, value in performance.items():
                        if isinstance(value, float):
                            if 'ic' in key.lower():
                                st.write(f"• {key}: {value:.4f}")
                            elif 'return' in key.lower():
                                st.write(f"• {key}: {value:.2%}")
                            else:
                                st.write(f"• {key}: {value:.4f}")
                        else:
                            st.write(f"• {key}: {value}")
        def render_rolling_ic(meta: dict):
            if 'rolling_ic' in meta and isinstance(meta['rolling_ic'], dict):
                rolling_ic_data = meta['rolling_ic']
                if 'dates' in rolling_ic_data and 'ic' in rolling_ic_data and 'icir' in rolling_ic_data:
                    st.subheader("📈 Rolling IC/ICIR 시계열")
                    try:
                        dates = rolling_ic_data['dates']
                        ic_values = rolling_ic_data['ic']
                        icir_values = rolling_ic_data['icir']
                        if isinstance(dates[0], str):
                            dates = pd.to_datetime(dates)
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        ax1.plot(dates, ic_values, label='Rolling IC', color='tab:blue', linewidth=2)
                        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                        ax1.set_title(f'{factor_name} - Rolling IC')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        ax2.plot(dates, icir_values, label='Rolling ICIR', color='tab:red', linewidth=2)
                        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                        ax2.set_title(f'{factor_name} - Rolling ICIR')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("평균 IC", f"{np.nanmean(ic_values):.4f}")
                        with col2:
                            st.metric("IC 표준편차", f"{np.nanstd(ic_values):.4f}")
                        with col3:
                            st.metric("평균 ICIR", f"{np.nanmean(icir_values):.4f}")
                        with col4:
                            positive_ic_ratio = np.sum(np.array(ic_values) > 0) / len(ic_values)
                            st.metric("양의 IC 비율", f"{positive_ic_ratio:.2%}")
                    except Exception as e:
                        st.warning(f"Rolling IC/ICIR 시각화 중 오류 발생: {e}")
                        st.info("Rolling IC 데이터 구조를 확인해주세요.")
                else:
                    st.info("Rolling IC 데이터가 올바른 형식이 아닙니다. 필요한 키: 'dates', 'ic', 'icir'")
            else:
                st.info("Rolling IC 데이터가 없습니다.")
        def render_params(meta: dict):
            if 'params' in meta and meta['params']:
                st.subheader("⚙️ 파라미터 정보")
                try:
                    if isinstance(meta['params'], dict):
                        if all(isinstance(v, (str, int, float, bool)) for v in meta['params'].values()):
                            params_df = pd.DataFrame.from_dict(meta['params'], orient='index', columns=['값'])
                            st.dataframe(params_df, use_container_width=True)
                        else:
                            st.write("파라미터 정보 (복잡한 구조):")
                            st.json(meta['params'])
                    else:
                        st.write(f"파라미터 정보: {str(meta['params'])}")
                except Exception as e:
                    st.write(f"파라미터 정보 표시 중 오류: {str(meta['params'])}")
                    st.error(f"파라미터 처리 오류: {e}")
        def render_ai_analysis(meta: dict):
            st.subheader("🤖 AI 성과 해석")
            if 'use_llm_analysis' not in st.session_state:
                st.session_state['use_llm_analysis'] = False
            if st.button('고도화된 AI 해석 적용', key=f'llm_zoo_{factor_name}'):
                st.session_state['use_llm_analysis'] = True
            with st.expander("🤖 AI 해석 결과", expanded=True):
                st.info(analyze_factor_performance_text(
                    meta.get('performance', {}), 
                    llm_api_key=None if not st.session_state['use_llm_analysis'] else 'env'
                ))
        def render_action_buttons():
            st.subheader("🎯 액션")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 이 팩터 불러오기", key=f"load_{factor_name}"):
                    st.session_state.custom_factor = factor_data['factor']
                    st.session_state.combined_factor_df = factor_data['factor']
                    st.session_state.factor_performance = meta.get('performance', {})
                    st.session_state.factor_generated = True
                    st.success(f"{factor_name} 팩터를 불러왔습니다! 분석/백테스트 탭에서 바로 사용 가능합니다.")
            with col2:
                if st.button("📊 백테스트 실행", key=f"backtest_{factor_name}"):
                    st.session_state.custom_factor = factor_data['factor']
                    st.session_state.combined_factor_df = factor_data['factor']
                    st.session_state.factor_performance = meta.get('performance', {})
                    st.session_state.factor_generated = True
                    st.success(f"{factor_name} 팩터로 백테스트를 실행할 수 있습니다!")
                    st.markdown('<script>document.querySelector("[data-testid=stTabs]").scrollIntoView();</script>', unsafe_allow_html=True)
            with col3:
                if st.button("🗑️ 이 팩터 삭제", type="secondary", key=f"delete_{factor_name}"):
                    delete_factor_from_zoo(factor_name)
                    st.warning(f"{factor_name} 팩터가 삭제되었습니다. 새로고침 후 목록이 갱신됩니다.")
                    st.rerun()
        # --- 실제 렌더링 ---
        st.subheader(f"📄 {factor_name} - 상세 정보")
        render_meta_info(meta)
        render_rolling_ic(meta)
        render_params(meta)
        render_ai_analysis(meta)
        render_action_buttons()

    def _render_linear_vs_nonlinear_section(self):
        """
        선형/비선형 팩터 성능 비교 및 Mega-Alpha 신호 생성 섹션
        """
        st.header("⚡ 선형/비선형 팩터 성능 비교")
        
        factors = load_factors_from_zoo()
        if not factors:
            st.info("팩터 Zoo에 저장된 팩터가 없습니다. 먼저 팩터를 생성/저장하세요!")
            return
        
        # 선형/비선형 팩터 분류
        linear_candidates = [k for k, v in factors.items() 
                           if v['meta'].get('weight_mode') in ["IC 기반 동적 가중치", "고정 가중치 직접 입력"]]
        nonlinear_candidates = [k for k, v in factors.items() 
                              if v['meta'].get('weight_mode') == '비선형']
        
        if not linear_candidates:
            st.warning("선형 팩터가 없습니다. 통계/기술적 팩터를 먼저 생성하세요!")
            return
        
        if not nonlinear_candidates:
            st.warning("비선형 팩터가 없습니다. 딥러닝 팩터를 먼저 생성하세요!")
            return
        
        st.info(f"📊 비교 가능한 팩터: 선형 {len(linear_candidates)}개, 비선형 {len(nonlinear_candidates)}개")
        
        # 팩터 선택
        col1, col2 = st.columns(2)
        with col1:
            linear_selected = st.selectbox("선형 팩터 선택", linear_candidates, key='linear_factor')
        with col2:
            nonlinear_selected = st.selectbox("비선형 팩터 선택", nonlinear_candidates, key='nonlinear_factor')
        
        if not linear_selected or not nonlinear_selected:
            st.warning("선형/비선형 팩터를 모두 선택하세요.")
            return
        
        # 선택된 팩터 정보 표시
        st.subheader("📋 선택된 팩터 정보")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 선형 팩터**")
            linear_meta = factors[linear_selected]['meta']
            st.write(f"• 이름: {linear_selected}")
            st.write(f"• 타입: {linear_meta.get('weight_mode', 'N/A')}")
            st.write(f"• 팩터: {', '.join(linear_meta.get('factor_types', []))}")
            if 'performance' in linear_meta:
                perf = linear_meta['performance']
                if 'mean_ic' in perf:
                    st.write(f"• 평균 IC: {perf['mean_ic']:.4f}")
                if 'icir' in perf:
                    st.write(f"• ICIR: {perf['icir']:.4f}")
        
        with col2:
            st.markdown("**🧠 비선형 팩터**")
            nonlinear_meta = factors[nonlinear_selected]['meta']
            st.write(f"• 이름: {nonlinear_selected}")
            st.write(f"• 타입: {nonlinear_meta.get('weight_mode', 'N/A')}")
            st.write(f"• 팩터: {', '.join(nonlinear_meta.get('factor_types', []))}")
            if 'performance' in nonlinear_meta:
                perf = nonlinear_meta['performance']
                if 'mean_ic' in perf:
                    st.write(f"• 평균 IC: {perf['mean_ic']:.4f}")
                if 'icir' in perf:
                    st.write(f"• ICIR: {perf['icir']:.4f}")
        
        # Rolling IC/ICIR 비교 시각화
        st.subheader("📈 Rolling IC/ICIR 비교")
        
        # 공통 기간 찾기
        linear_ic = factors[linear_selected]['meta'].get('rolling_ic', {})
        nonlinear_ic = factors[nonlinear_selected]['meta'].get('rolling_ic', {})
        
        # 데이터 구조 확인
        if (linear_ic and nonlinear_ic and 
            isinstance(linear_ic, dict) and isinstance(nonlinear_ic, dict) and
            'dates' in linear_ic and 'ic' in linear_ic and 'icir' in linear_ic and
            'dates' in nonlinear_ic and 'ic' in nonlinear_ic and 'icir' in nonlinear_ic):
            
            try:
                # 공통 날짜 범위 찾기
                linear_dates = pd.to_datetime(linear_ic['dates'])
                nonlinear_dates = pd.to_datetime(nonlinear_ic['dates'])
                common_dates = linear_dates.intersection(nonlinear_dates)
                
                if len(common_dates) > 0:
                    # 공통 기간의 데이터만 사용
                    linear_mask = linear_dates.isin(common_dates)
                    nonlinear_mask = nonlinear_dates.isin(common_dates)
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                    
                    # IC 비교
                    ax1.plot(common_dates, np.array(linear_ic['ic'])[linear_mask], 
                            label=f'선형: {linear_selected}', color='tab:blue', linewidth=2)
                    ax1.plot(common_dates, np.array(nonlinear_ic['ic'])[nonlinear_mask], 
                            label=f'비선형: {nonlinear_selected}', color='tab:red', linewidth=2)
                    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax1.set_title('Rolling IC 비교')
                    ax1.set_ylabel('IC')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # ICIR 비교
                    ax2.plot(common_dates, np.array(linear_ic['icir'])[linear_mask], 
                            label=f'선형: {linear_selected}', color='tab:blue', linewidth=2)
                    ax2.plot(common_dates, np.array(nonlinear_ic['icir'])[nonlinear_mask], 
                            label=f'비선형: {nonlinear_selected}', color='tab:red', linewidth=2)
                    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax2.set_title('Rolling ICIR 비교')
                    ax2.set_ylabel('ICIR')
                    ax2.set_xlabel('날짜')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # 성과 지표 비교 테이블
                    st.subheader("📊 성과 지표 비교")
                    
                    comparison_data = {
                        '지표': ['평균 IC', 'IC 표준편차', '평균 ICIR', '양의 IC 비율'],
                        '선형 팩터': [
                            f"{np.nanmean(np.array(linear_ic['ic'])[linear_mask]):.4f}",
                            f"{np.nanstd(np.array(linear_ic['ic'])[linear_mask]):.4f}",
                            f"{np.nanmean(np.array(linear_ic['icir'])[linear_mask]):.4f}",
                            f"{np.sum(np.array(linear_ic['ic'])[linear_mask] > 0) / len(common_dates):.2%}"
                        ],
                        '비선형 팩터': [
                            f"{np.nanmean(np.array(nonlinear_ic['ic'])[nonlinear_mask]):.4f}",
                            f"{np.nanstd(np.array(nonlinear_ic['ic'])[nonlinear_mask]):.4f}",
                            f"{np.nanmean(np.array(nonlinear_ic['icir'])[nonlinear_mask]):.4f}",
                            f"{np.sum(np.array(nonlinear_ic['ic'])[nonlinear_mask] > 0) / len(common_dates):.2%}"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # 승자 판정
                    st.subheader("🏆 성과 비교 결과")
                    
                    linear_mean_ic = np.nanmean(np.array(linear_ic['ic'])[linear_mask])
                    nonlinear_mean_ic = np.nanmean(np.array(nonlinear_ic['ic'])[nonlinear_mask])
                    linear_icir = np.nanmean(np.array(linear_ic['icir'])[linear_mask])
                    nonlinear_icir = np.nanmean(np.array(nonlinear_ic['icir'])[nonlinear_mask])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if linear_mean_ic > nonlinear_mean_ic:
                            st.success(f"🎯 IC 승자: 선형 팩터 ({linear_mean_ic:.4f} vs {nonlinear_mean_ic:.4f})")
                        else:
                            st.success(f"🎯 IC 승자: 비선형 팩터 ({nonlinear_mean_ic:.4f} vs {linear_mean_ic:.4f})")
                    
                    with col2:
                        if linear_icir > nonlinear_icir:
                            st.success(f"🎯 ICIR 승자: 선형 팩터 ({linear_icir:.4f} vs {nonlinear_icir:.4f})")
                        else:
                            st.success(f"🎯 ICIR 승자: 비선형 팩터 ({nonlinear_icir:.4f} vs {linear_icir:.4f})")
                    
                    with col3:
                        overall_winner = "선형" if (linear_mean_ic + linear_icir) > (nonlinear_mean_ic + nonlinear_icir) else "비선형"
                        st.info(f"🏆 종합 승자: {overall_winner} 팩터")
                else:
                    st.warning("두 팩터의 공통 기간이 없습니다.")
            except Exception as e:
                st.warning(f"Rolling IC/ICIR 비교 중 오류 발생: {e}")
                st.info("Rolling IC 데이터 구조를 확인해주세요.")
        else:
            st.warning("Rolling IC 데이터가 부족하거나 올바른 형식이 아닙니다.")
        
        # Mega-Alpha 신호 생성
        st.subheader("⚡ Mega-Alpha 신호 생성 및 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            combination_method = st.selectbox(
                "결합 방식",
                ["단순 평균", "IC 가중 평균", "동적 가중치"],
                help="두 팩터를 결합하는 방식 선택"
            )
        
        with col2:
            if st.button("🚀 Mega-Alpha 신호 생성", type="primary"):
                try:
                    # 두 팩터 데이터 가져오기
                    lin_df = factors[linear_selected]['factor']
                    nonlin_df = factors[nonlinear_selected]['factor']
                    
                    # 공통 인덱스/컬럼만 결합
                    common_idx = lin_df.index.intersection(nonlin_df.index)
                    common_col = lin_df.columns.intersection(nonlin_df.columns)
                    
                    if len(common_idx) == 0 or len(common_col) == 0:
                        st.error("두 팩터의 공통 구간이 없습니다.")
                        return
                    
                    # 결합 방식에 따른 가중치 계산
                    if combination_method == "단순 평균":
                        mega_alpha = (lin_df.loc[common_idx, common_col] + nonlin_df.loc[common_idx, common_col]) / 2
                        weights = {"선형": 0.5, "비선형": 0.5}
                    elif combination_method == "IC 가중 평균":
                        # IC 기반 가중치
                        linear_ic_weight = abs(linear_mean_ic) / (abs(linear_mean_ic) + abs(nonlinear_mean_ic))
                        nonlinear_ic_weight = abs(nonlinear_mean_ic) / (abs(linear_mean_ic) + abs(nonlinear_mean_ic))
                        mega_alpha = (linear_ic_weight * lin_df.loc[common_idx, common_col] + 
                                    nonlinear_ic_weight * nonlin_df.loc[common_idx, common_col])
                        weights = {"선형": linear_ic_weight, "비선형": nonlinear_ic_weight}
                    else:  # 동적 가중치
                        # 시간에 따른 동적 가중치 (간단한 구현)
                        mega_alpha = (lin_df.loc[common_idx, common_col] + nonlin_df.loc[common_idx, common_col]) / 2
                        weights = {"선형": 0.5, "비선형": 0.5}
                    
                    # Rolling IC/ICIR 계산
                    universe_data = st.session_state.universe_data
                    future_returns = universe_data.pct_change().shift(-1)
                    mega_ic = self.alpha_engine.calculate_rolling_ic(mega_alpha, future_returns, window=20)
                    
                    # 결과 저장
                    st.session_state.mega_alpha_factor = mega_alpha
                    st.session_state.mega_alpha_weights = weights
                    st.session_state.mega_alpha_ic = mega_ic
                    
                    st.success("✅ Mega-Alpha 신호 생성 완료!")
                    
                    # 가중치 표시
                    st.info(f"📊 적용된 가중치: 선형 {weights['선형']:.3f}, 비선형 {weights['비선형']:.3f}")
                    
                except Exception as e:
                    st.error(f"Mega-Alpha 신호 생성 실패: {e}")
        
        # Mega-Alpha 신호 결과 표시
        if 'mega_alpha_ic' in st.session_state:
            st.subheader("📈 Mega-Alpha 신호 성과")
            
            mega_ic = st.session_state.mega_alpha_ic
            
            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(mega_ic['dates'], mega_ic['ic'], label='Mega-Alpha IC', color='tab:green', linewidth=2)
            ax.plot(mega_ic['dates'], mega_ic['icir'], label='Mega-Alpha ICIR', color='tab:olive', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title("Mega-Alpha 신호의 Rolling IC/ICIR")
            ax.set_ylabel('IC/ICIR')
            ax.set_xlabel('날짜')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            # 성과 지표
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균 IC", f"{np.nanmean(mega_ic['ic']):.4f}")
            with col2:
                st.metric("IC 표준편차", f"{np.nanstd(mega_ic['ic']):.4f}")
            with col3:
                st.metric("평균 ICIR", f"{np.nanmean(mega_ic['icir']):.4f}")
            with col4:
                positive_ic_ratio = np.sum(np.array(mega_ic['ic']) > 0) / len(mega_ic['ic'])
                st.metric("양의 IC 비율", f"{positive_ic_ratio:.2%}")
            
            # AI 해석
            st.subheader("🤖 Mega-Alpha 신호 AI 해석")
            
            if st.button('고도화된 AI 해석 적용', key='llm_mega'):
                st.session_state['use_llm_analysis'] = True
            
            with st.expander("🤖 AI 해석 결과", expanded=True):
                st.info(analyze_factor_performance_text({
                    'mean_ic': float(np.nanmean(mega_ic['ic'])) if mega_ic['ic'] else 0,
                    'icir': float(np.nanmean(mega_ic['icir'])) if mega_ic['icir'] else 0,
                    'positive_ic_ratio': positive_ic_ratio
                }, llm_api_key=None if not st.session_state.get('use_llm_analysis', False) else 'env'))
            
            # 백테스트 실행 버튼
            if st.button("📊 Mega-Alpha 신호 백테스트 실행", type="primary"):
                qlib_factor = self.alpha_engine.convert_to_qlib_format(st.session_state.mega_alpha_factor)
                st.session_state.custom_factor = qlib_factor
                st.session_state.combined_factor_df = st.session_state.mega_alpha_factor
                st.session_state.factor_generated = True
                st.success("Mega-Alpha 신호로 백테스트를 실행할 수 있습니다!")

    def _render_formula_factor_section(self):
        """
        공식 기반 알파 팩터 생성 섹션 렌더링
        - 템플릿, 직접 입력, 고급 편집기 등 다양한 공식 입력 방식 지원
        - UI/UX(위젯, 출력 포맷 등)는 절대 변경하지 않음
        """
        st.subheader("📝 공식 기반 팩터 생성")
        if not st.session_state.get('universe_loaded', False):
            st.warning("먼저 투자 유니버스를 구성하세요.")
            return
        st.info("**프로세스:** 수학적 공식을 직접 입력하거나 템플릿을 활용해 커스텀 알파 팩터를 생성합니다.")
        # 공식 입력 방식 선택
        st.subheader("📝 공식 입력 방식")
        input_method = st.radio(
            "공식 입력 방식 선택",
            ["템플릿 사용", "직접 입력", "고급 편집기"],
            help="템플릿을 사용하거나 직접 공식을 입력할 수 있습니다."
        )
        if input_method == "템플릿 사용":
            self._render_formula_templates()
        elif input_method == "직접 입력":
            self._render_direct_formula_input()
        else:
            self._render_advanced_formula_editor()

    def _render_formula_templates(self):
        """
        공식 템플릿 섹션 렌더링
        - 템플릿 선택, 파라미터 입력, 결합 방식, 팩터 생성 등 반복되는 부분 함수화
        """
        st.subheader("📋 공식 템플릿")
        templates = self.formula_pipeline.generator.get_formula_templates()
        # 템플릿 카테고리 선택
        category = st.selectbox(
            "템플릿 카테고리",
            list(templates.keys()),
            help="사용할 템플릿 카테고리를 선택하세요"
        )
        selected_templates = st.multiselect(
            "사용할 템플릿 선택",
            list(templates[category].keys()),
            help="여러 템플릿을 선택하면 자동으로 결합됩니다"
        )
        def render_template_params() -> dict:
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("기본 윈도우 크기", 5, 50, 20)
                momentum_period = st.slider("모멘텀 기간", 5, 30, 10)
                volatility_period = st.slider("변동성 기간", 10, 60, 20)
            with col2:
                rsi_period = st.slider("RSI 기간", 5, 30, 14)
                macd_fast = st.slider("MACD 빠른선", 5, 20, 12)
                macd_slow = st.slider("MACD 느린선", 20, 50, 26)
            return {
                'window': window_size,
                'momentum_period': momentum_period,
                'volatility_period': volatility_period,
                'rsi_period': rsi_period,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow
            }
        def render_combine_method() -> str:
            return st.selectbox(
                "팩터 결합 방식",
                ["ic_weighted", "equal_weight", "none"],
                format_func=lambda x: {
                    "ic_weighted": "IC 기반 가중치",
                    "equal_weight": "동일 가중치",
                    "none": "개별 팩터만"
                }[x],
                help="여러 팩터를 결합하는 방식을 선택하세요"
            )
        if selected_templates:
            st.subheader("📊 선택된 템플릿")
            formulas = {}
            for template_name in selected_templates:
                formula = templates[category][template_name]
                formulas[template_name] = formula
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{template_name}**")
                with col2:
                    st.code(formula, language="python")
            st.subheader("⚙️ 파라미터 설정")
            params = render_template_params()
            combine_method = render_combine_method()
            if st.button("📝 공식 기반 팩터 생성", type="primary"):
                self._generate_formula_factors(formulas, params, combine_method)

    def _render_direct_formula_input(self):
        """
        직접 공식 입력 섹션 렌더링
        - 공식 입력, 파라미터 입력, 생성 버튼 등 반복되는 부분 함수화
        """
        st.subheader("✏️ 직접 공식 입력")
        with st.expander("📚 사용 가능한 함수 및 변수", expanded=False):
            parser = self.formula_pipeline.generator.parser
            functions = parser.get_available_functions()
            st.markdown("**📊 기본 변수:**")
            st.write("- `price`, `close`: 종가 데이터")
            st.write("- `returns`: 수익률")
            st.write("- `log_returns`: 로그 수익률")
            st.write("- `volume`: 거래량")
            st.write("- `high`, `low`, `open`: 고가, 저가, 시가 (현재는 종가와 동일)")
            st.write("- `t`: 시간 인덱스")
            st.write("- `n`: 데이터 포인트 수")
            st.markdown("**🔧 사용 가능한 함수:**")
            for func, desc in functions.items():
                st.write(f"- `{func}`: {desc}")
        formula_name = st.text_input(
            "팩터 이름",
            placeholder="예: 커스텀_모멘텀",
            help="생성할 팩터의 이름을 입력하세요"
        )
        formula = st.text_area(
            "수학 공식",
            placeholder="예: momentum(price, 20) * normalize(volume)",
            height=100,
            help="수학적 공식을 Python 문법으로 입력하세요"
        )
        def render_custom_params() -> dict:
            col1, col2 = st.columns(2)
            with col1:
                custom_param1 = st.text_input("커스텀 파라미터 1", placeholder="예: period")
                custom_value1 = st.number_input("값 1", value=20)
            with col2:
                custom_param2 = st.text_input("커스텀 파라미터 2", placeholder="예: threshold")
                custom_value2 = st.number_input("값 2", value=0.5)
            params = {}
            if custom_param1:
                params[custom_param1] = custom_value1
            if custom_param2:
                params[custom_param2] = custom_value2
            return params
        if formula:
            parser = self.formula_pipeline.generator.parser
            is_valid, message = parser.validate_formula(formula)
            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        st.subheader("⚙️ 추가 파라미터")
        params = render_custom_params()
        if st.button("📝 단일 공식 팩터 생성", type="primary"):
            if formula_name and formula:
                formulas = {formula_name: formula}
                self._generate_formula_factors(formulas, params, "none")
            else:
                st.error("팩터 이름과 공식을 모두 입력해주세요.")

    def _render_advanced_formula_editor(self):
        """
        고급 공식 편집기 섹션 렌더링
        - 여러 공식 입력, 결합 방식, 고급 옵션 등 반복되는 부분 함수화
        """
        st.subheader("🔧 고급 공식 편집기")
        st.info("여러 공식을 한 번에 입력하고 관리할 수 있습니다.")
        formulas = {}
        num_formulas = st.slider("공식 개수", 1, 10, 3)
        for i in range(num_formulas):
            st.markdown(f"**공식 {i+1}**")
            col1, col2 = st.columns([1, 2])
            with col1:
                name = st.text_input(f"팩터 이름 {i+1}", placeholder=f"팩터_{i+1}")
            with col2:
                formula = st.text_area(f"공식 {i+1}", placeholder="momentum(price, 20)", height=80)
            if name and formula:
                formulas[name] = formula
        def render_advanced_combine_method() -> str:
            return st.selectbox(
                "결합 방식",
                ["ic_weighted", "equal_weight", "none"],
                format_func=lambda x: {
                    "ic_weighted": "IC 기반 가중치",
                    "equal_weight": "동일 가중치",
                    "none": "개별 팩터만"
                }[x]
            )
        def render_advanced_options() -> dict:
            col1, col2 = st.columns(2)
            with col1:
                validation_mode = st.checkbox("엄격한 검증 모드", value=True)
            with col2:
                error_handling = st.selectbox(
                    "오류 처리",
                    ["skip", "stop", "warn"],
                    format_func=lambda x: {
                        "skip": "오류 건너뛰기",
                        "stop": "오류 시 중단",
                        "warn": "경고 후 계속"
                    }[x]
                )
                cache_results = st.checkbox("결과 캐싱", value=True)
            return {
                'validation_mode': validation_mode,
                'error_handling': error_handling,
                'cache_results': cache_results
            }
        if formulas:
            st.subheader("📋 입력된 공식들")
            for name, formula in formulas.items():
                st.code(f"{name}: {formula}", language="python")
            st.subheader("⚙️ 고급 설정")
            combine_method = render_advanced_combine_method()
            advanced_options = render_advanced_options()
            if st.button("🔧 고급 공식 팩터 생성", type="primary"):
                self._generate_formula_factors(formulas, {}, combine_method)

    def _generate_formula_factors(self, formulas: Dict[str, str], params: Dict[str, Any], combine_method: str):
        """공식 기반 팩터 생성 실행"""
        try:
            with st.spinner("공식 기반 팩터 생성 중..."):
                # 파이프라인 실행
                result = self.formula_pipeline.run_pipeline(
                    formulas=formulas,
                    universe_data=st.session_state.universe_data,
                    volume_data=st.session_state.get('volume_data'),
                    params=params,
                    combine_method=combine_method
                )
                
                # 결과 저장
                st.session_state.formula_factors = result['individual_factors']
                st.session_state.formula_combined_factor = result['combined_factor']
                st.session_state.formula_performance = result['performance']
                st.session_state.factor_generated = True
                
                st.success("✅ 공식 기반 팩터 생성 완료!")
                
                # 결과 표시
                st.subheader("📊 생성된 팩터 결과")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("생성된 팩터 수", len(result['individual_factors']))
                with col2:
                    st.metric("IC", f"{result['performance']['ic']:.4f}")
                with col3:
                    st.metric("데이터 포인트", f"{result['performance']['data_points']:,}")
                
                # 개별 팩터 성과
                if len(result['individual_factors']) > 1:
                    st.subheader("📈 개별 팩터 성과")
                    
                    performance_data = []
                    for name, factor in result['individual_factors'].items():
                        # 간단한 성과 계산
                        future_returns = st.session_state.universe_data.pct_change().shift(-1)
                        ic = self.formula_pipeline._calculate_ic(factor, future_returns)
                        performance_data.append({
                            '팩터명': name,
                            'IC': ic,
                            'IC_절댓값': abs(ic),
                            '표준편차': factor.std().mean(),
                            '평균': factor.mean().mean()
                        })
                    
                    perf_df = pd.DataFrame(performance_data)
                    st.dataframe(perf_df, use_container_width=True)
                
                # 팩터 Zoo 자동 저장
                now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                factor_label = f"{now_str}_공식기반_{len(formulas)}개팩터"
                
                meta = {
                    'factor_name': factor_label,
                    'created_at': now_str,
                    'factor_types': ['공식기반'],
                    'weight_mode': combine_method,
                    'weights': {},
                    'performance': result['performance'],
                    'params': {
                        'formulas': formulas,
                        'params': params,
                        'combine_method': combine_method
                    },
                    'rolling_ic': {}  # 나중에 계산 가능
                }
                
                from utils import save_factor_to_zoo
                save_factor_to_zoo(factor_label, {
                    'meta': meta, 
                    'factor': result['combined_factor']
                })
                
                st.info(f"📦 [팩터 Zoo]에 자동 저장됨: {factor_label}")
                
        except Exception as e:
            st.error(f"공식 기반 팩터 생성 중 오류: {e}")
            import traceback
            st.code(traceback.format_exc())

# 애플리케이션 실행
if __name__ == "__main__":
    try:
        app = AlphaForgeApp()
        app.run()
    except Exception as e:
        st.error(f"애플리케이션 시작 중 오류 발생: {e}")
        logger.error(f"Application startup error: {e}")
        import traceback
        st.code(traceback.format_exc())

