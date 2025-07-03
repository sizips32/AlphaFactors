import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, Dict, List

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
    logger
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
    
    def run(self):
        """메인 애플리케이션 실행"""
        self._setup_page()
        self._render_sidebar()
        self._render_main_content()
    
    def _setup_page(self):
        """페이지 설정"""
        st.set_page_config(
            page_title=self.config.ui.page_title,
            page_icon=self.config.ui.page_icon,
            layout=self.config.ui.layout
        )
        
        st.title("🚀 AlphaForge: 미국주식 딥러닝 팩터 + Qlib 백테스팅")
        st.markdown("---")
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("📋 프로세스 단계")
            st.markdown("""
            ### 1. 유니버스 구성
            - 여러 종목의 OHLCV 데이터 다운로드
            - 횡단면 분석을 위한 데이터 정렬
            
            ### 2. 알파 팩터 생성
            - 올바른 횡단면 팩터 계산
            - IC 기반 팩터 결합 및 검증
            
            ### 3. 백테스팅
            - Qlib을 통한 포트폴리오 성과 분석
            - 리스크 지표 및 수익률 시각화
            """)
            
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
    
    def _render_main_content(self):
        """메인 컨텐츠 렌더링"""
        # 1. 데이터 준비 섹션
        self._render_data_section()
        
        st.header("2. 🎯 알파 팩터 생성")
        
        tab1, tab2 = st.tabs(["📊 통계/기술적 팩터", "🧠 딥러닝 팩터"])
        
        with tab1:
            self._render_statistical_factor_section()
            
        with tab2:
            self._render_dl_factor_section()
        
        # 3. Qlib 백테스팅 섹션
        self._render_backtest_section()
        
        # 4. 설명 섹션
        self._render_explanation_section()
    
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
        """올바른 알파 팩터 생성 섹션 렌더링"""
        
        if not st.session_state.get('universe_loaded', False):
            st.warning("먼저 투자 유니버스를 구성하세요.")
            return
        
        # 팩터 타입 선택
        st.subheader("📊 팩터 타입 선택")
        
        col1, col2 = st.columns(2)
        
        with col1:
            factor_types = st.multiselect(
                "생성할 팩터 선택",
                ['momentum', 'reversal', 'volatility', 'volume', 'rsi', 'price_to_ma'],
                default=st.session_state.get('selected_factors', ['momentum', 'reversal', 'volatility']),
                help="여러 팩터를 선택하면 IC 가중으로 결합됩니다."
            )
        
        with col2:
            ic_lookback = st.slider(
                "IC 계산 기간 (일)",
                min_value=20, max_value=120,
                value=st.session_state.get('ic_lookback', 60),
                help="Information Coefficient 계산을 위한 과거 기간"
            )
        
        if not factor_types:
            st.error("최소 하나의 팩터를 선택해주세요.")
            return
        
        # 팩터명 매핑 (한글 표시용)
        factor_names_ko = {
            'momentum': '모멘텀',
            'reversal': '반전',
            'volatility': '저변동성',
            'volume': '거래량',
            'rsi': 'RSI',
            'price_to_ma': '이동평균 대비 가격'
        }
        
        selected_names = [factor_names_ko.get(f, f) for f in factor_types]
        st.info(f"선택된 팩터: {', '.join(selected_names)}")
        
        st.subheader("⚙️ 팩터 파라미터 설정")
        with st.expander("파라미터 상세 설정", expanded=False):
            self.config.factor.momentum_lookback = st.slider("모멘텀 기간", 5, 60, self.config.factor.momentum_lookback)
            self.config.factor.reversal_lookback = st.slider("반전 기간", 3, 30, self.config.factor.reversal_lookback)
            self.config.factor.volatility_lookback = st.slider("변동성 기간", 10, 60, self.config.factor.volatility_lookback)
            self.config.factor.rsi_period = st.slider("RSI 기간", 7, 28, self.config.factor.rsi_period)
            self.config.factor.ma_period = st.slider("이동평균 기간", 10, 100, self.config.factor.ma_period)
        
        if st.button("🚀 알파 팩터 생성", type="primary"):
            try:
                universe_data = st.session_state.universe_data
                volume_data = st.session_state.get('volume_data')
                
                with st.spinner("알파 팩터 계산 중..."):
                    # 개별 팩터들 계산
                    factors_dict = self.alpha_engine.calculate_all_factors(
                        universe_data, volume_data, factor_types
                    )
                    
                    if not factors_dict:
                        st.error("팩터 생성에 실패했습니다.")
                        return
                    
                    st.success(f"✅ {len(factors_dict)}개 팩터 생성 완료")
                    
                    # 미래 수익률 계산 (1일 후)
                    future_returns = universe_data.pct_change().shift(-1)
                    
                    # IC 기반 가중 결합
                    if len(factors_dict) > 1:
                        combined_factor, ic_weights = self.alpha_engine.combine_factors_ic_weighted(
                            factors_dict, future_returns, ic_lookback
                        )
                        
                        st.info("여러 팩터를 IC 가중 방식으로 결합했습니다.")
                        
                        # IC 가중치 표시
                        st.subheader("⚖️ IC 기반 팩터 가중치")
                        weights_df = pd.DataFrame.from_dict(
                            {factor_names_ko.get(k, k): [v] for k, v in ic_weights.items()}, 
                            orient='index',
                            columns=['가중치']
                        )
                        st.dataframe(weights_df, use_container_width=True)
                        
                    else:
                        combined_factor = list(factors_dict.values())[0]
                        ic_weights = {list(factors_dict.keys())[0]: 1.0}
                        st.info("단일 팩터를 사용합니다.")
                    
                    if combined_factor.empty:
                        st.error("결합된 팩터가 비어있습니다.")
                        return
                    
                    # 팩터 성능 분석
                    performance = self.alpha_engine.analyze_factor_performance(
                        combined_factor, future_returns
                    )
                    
                    # Qlib 형식으로 변환
                    qlib_factor = self.alpha_engine.convert_to_qlib_format(combined_factor)
                    
                    # 세션 상태 업데이트
                    st.session_state.custom_factor = qlib_factor
                    st.session_state.combined_factor_df = combined_factor
                    st.session_state.individual_factors = factors_dict
                    st.session_state.factor_performance = performance
                    st.session_state.ic_weights = ic_weights
                    st.session_state.factor_generated = True
                    
                    st.success("✅ 올바른 알파 팩터 생성 완료!")
                
                # 결과 시각화
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
    
    def _render_backtest_section(self):
        """백테스팅 섹션 렌더링"""
        st.header("3. 📊 포트폴리오 백테스팅")
        
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
        
        factor_names_ko = {
            'momentum': '모멘텀',
            'reversal': '반전',
            'volatility': '저변동성',
            'volume': '거래량',
            'rsi': 'RSI',
            'price_to_ma': '이동평균 대비 가격'
        }
        
        factor_backtester = FactorBacktester(universe_data, volume_data)
        
        with st.spinner("개별 팩터들 백테스팅 비교 중..."):
            comparison_results = factor_backtester.compare_factors(
                individual_factors, factor_names_ko
            )
        
        if comparison_results:
            st.session_state.factor_comparison_results = comparison_results
            st.success("✅ 개별 팩터 성과 비교 완료!")
    
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
        with st.expander("💡 코드 설명 및 사용법", expanded=False):
            st.markdown(f"""
            ## 🎯 올바른 AlphaForge 시스템
            
            ### ✅ 핵심 개선사항
            1. **진정한 알파 팩터**: 횡단면 순위 기반 팩터 생성
            2. **IC 기반 결합**: Information Coefficient로 팩터 가중치 최적화
            3. **멀티 종목 분석**: 여러 종목을 동시에 분석하여 의미 있는 알파 창출
            4. **성능 검증**: IC, ICIR 등 퀀트 업계 표준 지표로 팩터 품질 검증
            5. **올바른 방법론**: 실제 퀀트 투자에서 사용하는 정통 방식 구현
            
            ### 🔍 알파 팩터 종류
            - **모멘텀**: 과거 수익률 기반 추세 추종
            - **반전**: 단기 반전 효과 포착
            - **저변동성**: 안정적인 종목 선호
            - **거래량**: 비정상적 거래량 증가 감지
            - **RSI**: 기술적 과매도/과매수 구간 활용
            - **이동평균**: 추세선 대비 상대적 위치
            
            ### 📊 사용 방법
            1. **유니버스 구성**: 10개 내외의 종목으로 투자 유니버스 구성
            2. **팩터 생성**: 원하는 팩터들을 선택하여 횡단면 순위 계산
            3. **성능 검증**: IC, ICIR 지표로 팩터 품질 확인
            4. **백테스팅**: Qlib으로 실제 포트폴리오 성과 분석
            
            ### ⚠️ 이전 방식의 문제점
            - ❌ 단일 종목 팩터를 모든 종목에 동일 적용
            - ❌ 횡단면 정보 부재로 순위 차이 없음
            - ❌ 실제 퀀트 투자에서 사용 불가능한 방식
            
            ### ✅ 현재 방식의 장점
            - ✅ 종목별 상대적 순위로 포트폴리오 구성 가능
            - ✅ IC 기반 과학적 팩터 결합
            - ✅ 업계 표준 성능 지표로 검증
            - ✅ 실제 헤지펀드에서 사용하는 정통 방법론
            - ✅ 딥러닝 기반 백테스팅으로 더 유연하고 안정적인 성과 분석
            """)

    def _render_dl_factor_section(self):
        """딥러닝 기반 알파 팩터 생성 섹션"""
        st.subheader("🧠 딥러닝 모델 기반 팩터 생성")

        if not st.session_state.get('universe_loaded', False):
            st.warning("먼저 투자 유니버스를 구성하세요.")
            return

        st.info("**프로세스:** 유니버스 내 모든 종목의 데이터를 사용하여 MLP 모델을 학습하고, 예측값을 새로운 알파 팩터로 사용합니다.")

        # 모델 파라미터 설정
        with st.expander("딥러닝 모델 파라미터 설정"):
            self.config.model.epochs = st.slider("Epochs", 10, 100, self.config.model.epochs, key='dl_epochs')
            self.config.model.window_size = st.slider("Window Size", 5, 30, self.config.model.window_size, key='dl_window_size')
            self.config.model.prediction_horizon = st.slider("Prediction Horizon", 1, 10, self.config.model.prediction_horizon, key='dl_prediction_horizon')

        if st.button("🧠 딥러닝 모델 학습 및 팩터 생성", type="primary"):
            try:
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
                self.model_trainer = ModelTrainer(self.config.model)
                trained_model = self.model_trainer.train_model(X_train, y_train)

                if trained_model:
                    st.success("✅ 딥러닝 모델 학습 완료!")
                    
                    # 팩터 생성 (예측)
                    with st.spinner("딥러닝 팩터 생성 중..."):
                        predictions = self.model_trainer.predict(X_train)
                        
                        # 예측값을 DataFrame으로 변환
                        factor_df = pd.DataFrame({
                            'datetime': all_dates,
                            'instrument': all_tickers,
                            'prediction': predictions
                        }).pivot(index='datetime', columns='instrument', values='prediction')

                        # 횡단면 순위화
                        ranked_factor = factor_df.rank(axis=1, pct=True)
                        
                        # Qlib 형식으로 변환
                        qlib_factor = self.alpha_engine.convert_to_qlib_format(ranked_factor)

                        # 세션 상태 업데이트
                        st.session_state.custom_factor = qlib_factor
                        st.session_state.combined_factor_df = ranked_factor
                        st.session_state.individual_factors = {"dl_factor": ranked_factor}
                        st.session_state.factor_generated = True

                    st.success("✅ 딥러닝 알파 팩터 생성 완료!")

                    # 결과 표시
                    st.subheader("📈 딥러닝 팩터 분석")
                    st.metric("데이터 포인트 수", f"{len(qlib_factor):,}")
                    st.dataframe(ranked_factor.tail(), use_container_width=True)

            except Exception as e:
                st.error(f"딥러닝 팩터 생성 중 오류: {e}")
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

