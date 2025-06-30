"""
app.py의 팩터 생성 부분 수정 버전
Fixed version of the factor generation section in app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from proper_alpha_factors import ProperAlphaFactors

def _render_model_section_fixed(self):
    """수정된 모델 섹션 렌더링"""
    st.header("2. 🧠 올바른 알파 팩터 생성")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("먼저 데이터를 다운로드하세요.")
        return
    
    # 유니버스 선택 옵션 추가
    st.subheader("📊 유니버스 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        universe_type = st.selectbox(
            "유니버스 타입",
            ["단일 종목 (데모용)", "미국 대형주", "기술주", "커스텀"],
            help="알파 팩터는 여러 종목에서 계산되어야 의미가 있습니다."
        )
    
    with col2:
        if universe_type == "커스텀":
            custom_tickers = st.text_input(
                "종목 리스트 (쉼표 구분)",
                value="AAPL,GOOGL,MSFT,TSLA,AMZN",
                help="예: AAPL,GOOGL,MSFT,TSLA,AMZN"
            )
        else:
            # 미리 정의된 유니버스
            universe_tickers = {
                "단일 종목 (데모용)": [st.session_state.get('ticker_name', 'AAPL')],
                "미국 대형주": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
                "기술주": ["AAPL", "GOOGL", "MSFT", "NVDA", "AMD", "INTC", "CRM"]
            }
            custom_tickers = ",".join(universe_tickers[universe_type])
    
    tickers_list = [t.strip().upper() for t in custom_tickers.split(",")]
    
    if universe_type == "단일 종목 (데모용)":
        st.warning("""
        ⚠️ **중요한 알림**: 
        - 단일 종목으로는 진정한 알파 팩터를 생성할 수 없습니다.
        - 이는 데모 목적으로만 제공되며, 실제 투자에서는 사용하지 마세요.
        - 진정한 알파 팩터는 **여러 종목 간의 상대적 순위**를 기반으로 합니다.
        """)
    
    st.info(f"선택된 유니버스: {', '.join(tickers_list)} ({len(tickers_list)}개 종목)")
    
    # 팩터 타입 선택
    st.subheader("🔧 팩터 타입 선택")
    factor_types = st.multiselect(
        "생성할 팩터 선택",
        ["모멘텀", "반전", "변동성", "거래량", "기술적"],
        default=["모멘텀", "반전"],
        help="여러 팩터를 선택하면 IC 가중으로 결합됩니다."
    )
    
    if not factor_types:
        st.error("최소 하나의 팩터를 선택해주세요.")
        return
    
    if st.button("🚀 올바른 알파 팩터 생성", type="primary"):
        try:
            # 1. 유니버스 데이터 다운로드
            with st.spinner("유니버스 데이터 다운로드 중..."):
                universe_data = {}
                volume_data = {}
                
                progress_bar = st.progress(0)
                for i, ticker in enumerate(tickers_list):
                    progress_bar.progress((i + 1) / len(tickers_list))
                    
                    # 기존 세션에 있으면 사용, 없으면 다운로드
                    if ticker == st.session_state.get('ticker_name') and 'ticker_data' in st.session_state:
                        df = st.session_state.ticker_data
                    else:
                        start_ts = pd.Timestamp("2022-01-01")
                        end_ts = pd.Timestamp("2023-12-31")
                        df = self.data_handler.download_data(ticker, start_ts, end_ts)
                    
                    if df is not None:
                        universe_data[ticker] = df['Close']
                        volume_data[ticker] = df['Volume']
                
                progress_bar.empty()
                
                if not universe_data:
                    st.error("다운로드된 데이터가 없습니다.")
                    return
                
                # DataFrame으로 변환
                universe_df = pd.DataFrame(universe_data).fillna(method='ffill')
                volume_df = pd.DataFrame(volume_data).fillna(method='ffill')
                
                st.success(f"{len(universe_data)}개 종목 데이터 준비 완료")
            
            # 2. 알파 팩터 생성
            with st.spinner("알파 팩터 계산 중..."):
                alpha_gen = ProperAlphaFactors()
                factors_dict = {}
                
                if "모멘텀" in factor_types:
                    factors_dict['momentum'] = alpha_gen.calculate_momentum_factor(universe_df, lookback=20)
                
                if "반전" in factor_types:
                    factors_dict['reversal'] = alpha_gen.calculate_reversal_factor(universe_df, lookback=5)
                
                if "변동성" in factor_types:
                    factors_dict['volatility'] = alpha_gen.calculate_volatility_factor(universe_df, lookback=20)
                
                if "거래량" in factor_types:
                    factors_dict['volume'] = alpha_gen.calculate_volume_factor(universe_df, volume_df, lookback=20)
                
                if "기술적" in factor_types:
                    factors_dict['technical'] = alpha_gen.calculate_technical_factor(universe_df, rsi_period=14)
                
                # 미래 수익률 계산
                future_returns = universe_df.pct_change().shift(-1)
                
                # IC 기반 가중 결합
                if len(factors_dict) > 1:
                    combined_factor = alpha_gen.combine_factors_ic_weighted(
                        factors_dict, future_returns, lookback=60
                    )
                    st.info("여러 팩터를 IC 가중 방식으로 결합했습니다.")
                else:
                    combined_factor = list(factors_dict.values())[0]
                    st.info("단일 팩터를 사용합니다.")
                
                # Qlib 형식으로 변환
                qlib_factor = alpha_gen.convert_to_qlib_format(combined_factor)
                
                # 세션 상태 업데이트
                st.session_state.custom_factor = qlib_factor
                st.session_state.combined_factor_df = combined_factor
                st.session_state.individual_factors = factors_dict
                st.session_state.factor_generated = True
                st.session_state.universe_data = universe_df
                
                st.success("✅ 올바른 알파 팩터 생성 완료!")
            
            # 3. 결과 시각화
            st.subheader("📈 생성된 알파 팩터 분석")
            
            # 개별 팩터들 시각화
            if len(factors_dict) > 1:
                st.write("**개별 팩터별 성능:**")
                
                # 각 팩터의 IC 시계열
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, (factor_name, factor_data) in enumerate(factors_dict.items()):
                    if i >= 4:  # 최대 4개까지만 표시
                        break
                    
                    # 팩터 값의 시계열 (첫 번째 종목)
                    first_ticker = factor_data.columns[0]
                    factor_data[first_ticker].plot(ax=axes[i], title=f"{factor_name} 팩터 ({first_ticker})")
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # 결합된 팩터 시각화
            st.write("**결합된 최종 알파 팩터:**")
            
            # 히트맵으로 팩터 값 표시
            if len(tickers_list) <= 10:  # 종목이 많지 않을 때만 히트맵 표시
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 최근 30일 데이터만 표시
                recent_data = combined_factor.tail(30)
                
                import seaborn as sns
                sns.heatmap(
                    recent_data.T, 
                    ax=ax, 
                    cmap='RdYlBu_r', 
                    center=0.5,
                    cbar_kws={'label': 'Factor Score (0-1)'}
                )
                ax.set_title("최근 30일 종목별 알파 팩터 점수")
                ax.set_xlabel("날짜")
                ax.set_ylabel("종목")
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # 팩터 통계
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("총 데이터 포인트", f"{len(qlib_factor):,}")
            
            with col2:
                st.metric("팩터 값 범위", f"{qlib_factor.min():.3f} ~ {qlib_factor.max():.3f}")
            
            with col3:
                st.metric("평균 팩터 값", f"{qlib_factor.mean():.3f}")
            
            # 핵심 설명 추가
            with st.expander("💡 올바른 알파 팩터의 특징", expanded=True):
                st.markdown("""
                ### ✅ 올바른 알파 팩터란?
                
                1. **횡단면 순위**: 각 시점에서 종목들 간의 상대적 순위를 매김 (0~1 백분위수)
                2. **의미 있는 분산**: 종목별로 서로 다른 값을 가져야 포트폴리오 구성 가능
                3. **예측력**: 미래 수익률과 상관관계가 있어야 함 (Information Coefficient)
                4. **안정성**: 시간이 지나도 일관된 신호를 제공
                
                ### 🔍 현재 생성된 팩터의 의미
                - **모멘텀**: 과거 수익률이 높은 종목에 높은 점수
                - **반전**: 최근 하락한 종목에 높은 점수 (단기 반등 기대)
                - **변동성**: 변동성이 낮은 안정적인 종목에 높은 점수
                - **거래량**: 거래량이 급증한 종목에 높은 점수
                - **기술적**: RSI 등 기술적 지표 기반 점수
                
                ### ⚠️ 이전 방식의 문제점
                - 단일 종목의 값을 모든 종목에 동일 적용 → 순위 차이 없음
                - 횡단면 정보 부재 → 의미 있는 포트폴리오 구성 불가
                - 실제 퀀트 투자에서는 절대 사용하지 않는 방식
                """)
                
        except Exception as e:
            st.error(f"알파 팩터 생성 중 오류: {e}")
            import traceback
            st.code(traceback.format_exc())

# 이 함수를 app.py의 기존 _render_model_section 대신 사용해야 함