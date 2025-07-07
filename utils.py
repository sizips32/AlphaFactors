import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Optional, Union
import logging
from datetime import datetime
import os
import pickle
import openai
from dotenv import load_dotenv

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
    df_clean[required_columns] = df_clean[required_columns].ffill().bfill()
    
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
        
        if isinstance(df, pd.DataFrame):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
        st.dataframe(df.describe(), use_container_width=True)

def display_error_with_suggestions(error_msg: str, suggestions: List[str] = None):
    """에러 메시지와 해결 방안 표시"""
    st.error(error_msg)
    
    if suggestions:
        with st.expander("💡 해결 방안", expanded=True):
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")

def show_success_with_details(success_msg: str, details: dict = None):
    """성공 메시지와 상세 정보 표시"""
    st.success(success_msg)
    
    if details:
        with st.expander("📊 상세 정보", expanded=False):
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    if 'ratio' in key.lower() or 'sharpe' in key.lower():
                        st.write(f"**{key}**: {value:.4f}")
                    elif 'return' in key.lower():
                        st.write(f"**{key}**: {value:.2%}")
                    else:
                        st.write(f"**{key}**: {value:.4f}")
                else:
                    st.write(f"**{key}**: {value}")

def show_warning_with_help(warning_msg: str, help_text: str = None):
    """경고 메시지와 도움말 표시"""
    st.warning(warning_msg)
    
    if help_text:
        with st.expander("❓ 도움말", expanded=False):
            st.info(help_text)

def validate_user_input(input_data: dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """사용자 입력 검증"""
    errors = []
    
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            errors.append(f"'{field}' 필드가 필요합니다.")
        elif isinstance(input_data[field], str) and not input_data[field].strip():
            errors.append(f"'{field}' 필드가 비어있습니다.")
        elif isinstance(input_data[field], (list, tuple)) and len(input_data[field]) == 0:
            errors.append(f"'{field}' 필드에 최소 하나의 항목이 필요합니다.")
    
    return len(errors) == 0, errors

def show_progress_with_status(message: str, progress_bar=None):
    """진행 상황 표시"""
    if progress_bar is None:
        progress_bar = st.progress(0)
    
    return progress_bar

def display_performance_summary(performance_metrics: dict, title: str = "성과 요약"):
    """성과 지표 요약 표시"""
    st.subheader(title)
    
    # 주요 지표들을 카테고리별로 분류
    return_metrics = {k: v for k, v in performance_metrics.items() if 'return' in k.lower()}
    risk_metrics = {k: v for k, v in performance_metrics.items() if any(x in k.lower() for x in ['volatility', 'drawdown', 'var'])}
    ratio_metrics = {k: v for k, v in performance_metrics.items() if any(x in k.lower() for x in ['sharpe', 'sortino', 'calmar'])}
    ic_metrics = {k: v for k, v in performance_metrics.items() if 'ic' in k.lower()}
    
    # 수익률 지표
    if return_metrics:
        st.markdown("**📈 수익률 지표**")
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(return_metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(key, f"{value:.2%}")
    
    # 리스크 지표
    if risk_metrics:
        st.markdown("**⚠️ 리스크 지표**")
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(risk_metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(key, f"{value:.4f}")
    
    # 비율 지표
    if ratio_metrics:
        st.markdown("**📊 비율 지표**")
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(ratio_metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(key, f"{value:.4f}")
    
    # IC 지표
    if ic_metrics:
        st.markdown("**🎯 IC 지표**")
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(ic_metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(key, f"{value:.4f}")

def create_download_link(data, filename: str, file_type: str = "csv"):
    """다운로드 링크 생성"""
    import base64
    
    if file_type == "csv":
        csv = data.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 CSV 다운로드</a>'
    elif file_type == "excel":
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            data.to_excel(writer, index=True)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Excel 다운로드</a>'
    
    return href

def show_data_quality_report(df: pd.DataFrame, title: str = "데이터 품질 리포트"):
    """데이터 품질 리포트 표시"""
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 행 수", len(df))
    
    with col2:
        st.metric("총 열 수", len(df.columns))
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("결측값 비율", f"{missing_pct:.2f}%")
    
    with col4:
        duplicate_rows = df.duplicated().sum()
        st.metric("중복 행", duplicate_rows)
    
    # 결측값 상세
    if df.isnull().any().any():
        st.markdown("**🔍 결측값 상세**")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        st.bar_chart(missing_data)
    
    # 데이터 타입 정보
    st.markdown("**📋 데이터 타입 정보**")
    dtype_info = pd.DataFrame({
        '컬럼명': df.columns,
        '데이터 타입': df.dtypes,
        '고유값 수': df.nunique(),
        '결측값 수': df.isnull().sum()
    })
    if isinstance(dtype_info, pd.DataFrame):
        for col in dtype_info.columns:
            if dtype_info[col].dtype == 'object':
                dtype_info[col] = dtype_info[col].astype(str)
    st.dataframe(dtype_info, use_container_width=True)

def cache_key_generator(*args) -> str:
    """캐시 키 생성"""
    return "_".join(str(arg) for arg in args)

@st.cache_data(ttl=3600)  # 1시간 캐시
def cached_calculation(func, *args, **kwargs):
    """계산 결과 캐싱"""
    return func(*args, **kwargs)

def save_factor_to_zoo(factor_name: str, factor_data: dict, zoo_dir: str = 'factor_zoo'):
    """
    팩터와 메타데이터를 factor_zoo/에 pickle로 저장합니다.
    - factor_name: 저장할 파일명(확장자 제외)
    - factor_data: {'meta': 메타데이터 dict, 'factor': 팩터 DataFrame/Series 등}
    - zoo_dir: 저장 폴더
    """
    if not os.path.exists(zoo_dir):
        os.makedirs(zoo_dir)
    file_path = os.path.join(zoo_dir, f"{factor_name}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(factor_data, f)


def load_factors_from_zoo(zoo_dir: str = 'factor_zoo') -> dict:
    """
    factor_zoo/ 내 모든 팩터 목록과 메타데이터를 반환합니다.
    반환: {factor_name: {'meta': ..., 'factor': ...}}
    """
    factors = {}
    if not os.path.exists(zoo_dir):
        return factors
    for fname in os.listdir(zoo_dir):
        if fname.endswith('.pkl'):
            fpath = os.path.join(zoo_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    factors[fname[:-4]] = data
            except Exception as e:
                logger.warning(f"팩터 Zoo 로드 실패: {fname} - {e}")
    return factors


def delete_factor_from_zoo(factor_name: str, zoo_dir: str = 'factor_zoo'):
    """
    factor_zoo/ 내 특정 팩터 파일을 삭제합니다.
    - factor_name: 삭제할 파일명(확장자 제외)
    """
    file_path = os.path.join(zoo_dir, f"{factor_name}.pkl")
    if os.path.exists(file_path):
        os.remove(file_path)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def summarize_with_llm(text: str, prompt: str = "", api_key: str = None) -> str:
    """
    OpenAI gpt-4.1-mini-2025-04-14 기반 자연어 해석 함수. .env에서 키를 읽어 보안 유지.
    - text: 해석할 데이터/지표 요약 텍스트
    - prompt: 추가 프롬프트(옵션)
    - api_key: LLM API 키(옵션, 없으면 .env 사용)
    """
    key = api_key or OPENAI_API_KEY
    if not key:
        return None
    try:
        openai.api_key = key
        system_prompt = prompt or "아래 데이터를 투자 전문가 관점에서 해석/제안해줘."
        user_content = f"분석 데이터:\n{text}"
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # 유효한 GPT-4 mini 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=400,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM 해석 오류] {e}"

def analyze_factor_performance_text(perf: dict, llm_api_key: str = None) -> str:
    """
    팩터 성능(performance dict)을 해석하고 투자적 시사점/제안을 반환합니다.
    LLM API 키가 있으면 LLM 해석을 우선 사용합니다.
    """
    if not perf:
        return "팩터 성능 분석 결과가 없습니다."
    # LLM 해석 우선
    if llm_api_key:
        llm_result = summarize_with_llm(str(perf), prompt="팩터 성능을 투자 관점에서 해석/제안해줘.", api_key=llm_api_key)
        if llm_result:
            return llm_result
    # 이하 rule-based 해석
    mean_ic = perf.get('mean_ic', 0)
    icir = perf.get('icir', 0)
    spread = perf.get('factor_spread', 0)
    turnover = perf.get('factor_turnover', 0)
    msg = []
    # IC 해석
    if mean_ic > 0.05:
        msg.append(f"✅ 평균 IC({mean_ic:.4f})가 0.05 이상으로 예측력이 우수합니다.")
    elif mean_ic > 0.02:
        msg.append(f"ℹ️ 평균 IC({mean_ic:.4f})가 0.02~0.05로 보통 수준입니다. 팩터 개선 여지가 있습니다.")
    else:
        msg.append(f"⚠️ 평균 IC({mean_ic:.4f})가 낮아 예측력이 부족합니다. 팩터 구조/파라미터를 재점검하세요.")
    # ICIR 해석
    if icir > 1:
        msg.append(f"✅ ICIR({icir:.2f})가 1 이상으로 예측력의 일관성/안정성이 매우 높습니다.")
    elif icir > 0.5:
        msg.append(f"ℹ️ ICIR({icir:.2f})가 0.5~1로 양호합니다.")
    else:
        msg.append(f"⚠️ ICIR({icir:.2f})가 낮아 팩터의 일관성이 부족할 수 있습니다.")
    # 분산/턴오버 해석
    if spread < 0.05:
        msg.append(f"⚠️ 팩터 분산({spread:.4f})이 낮아 종목 간 차별화가 부족합니다.")
    if turnover > 0.5:
        msg.append(f"⚠️ 팩터 턴오버({turnover:.2f})가 높아 거래비용 증가 위험이 있습니다.")
    # 종합 제안
    if mean_ic > 0.05 and icir > 1:
        msg.append("🎯 이 팩터는 실제 투자 전략에 바로 적용해볼 만합니다.")
    elif mean_ic > 0.02:
        msg.append("🔍 팩터 파라미터 튜닝, 결합, 딥러닝 팩터와의 조합을 추가 실험해보세요.")
    else:
        msg.append("🛠️ 팩터 구조/데이터/파라미터를 근본적으로 재설계하는 것이 좋습니다.")
    return '\n'.join(msg)

def analyze_backtest_performance_text(report: dict, llm_api_key: str = None) -> str:
    """
    백테스트 성과 리포트(성과지표 dict)를 해석하고 투자적 시사점/제안을 반환합니다.
    LLM API 키가 있으면 LLM 해석을 우선 사용합니다.
    """
    if not report:
        return "백테스트 성과 분석 결과가 없습니다."
    if llm_api_key:
        llm_result = summarize_with_llm(str(report), prompt="백테스트 성과를 투자 관점에서 해석/제안해줘.", api_key=llm_api_key)
        if llm_result:
            return llm_result
    sharpe = report.get('Sharpe', 0)
    ret = report.get('Return', 0)
    maxdd = report.get('MaxDrawdown', 0)
    msg = []
    # Sharpe 해석
    if sharpe > 1.5:
        msg.append(f"✅ 샤프 비율({sharpe:.2f})이 1.5 이상으로 위험 대비 수익이 매우 우수합니다.")
    elif sharpe > 1.0:
        msg.append(f"ℹ️ 샤프 비율({sharpe:.2f})이 1.0~1.5로 양호합니다.")
    else:
        msg.append(f"⚠️ 샤프 비율({sharpe:.2f})이 낮아 전략의 위험 대비 수익이 부족할 수 있습니다.")
    # 수익률 해석
    if ret > 0.2:
        msg.append(f"✅ 연환산 수익률({ret:.2%})이 매우 높습니다.")
    elif ret > 0.05:
        msg.append(f"ℹ️ 연환산 수익률({ret:.2%})이 보통 수준입니다.")
    else:
        msg.append(f"⚠️ 연환산 수익률({ret:.2%})이 낮아 개선이 필요합니다.")
    # MDD 해석
    if maxdd > 0.3:
        msg.append(f"⚠️ 최대 낙폭({maxdd:.2%})이 커서 리스크 관리가 필요합니다.")
    elif maxdd > 0.15:
        msg.append(f"ℹ️ 최대 낙폭({maxdd:.2%})이 다소 큽니다. 방어 전략을 고려하세요.")
    else:
        msg.append(f"✅ 최대 낙폭({maxdd:.2%})이 우수하게 관리되고 있습니다.")
    # 종합 제안
    if sharpe > 1.5 and ret > 0.1 and maxdd < 0.15:
        msg.append("🎯 이 전략은 실제 투자에 매우 적합합니다.")
    elif sharpe > 1.0:
        msg.append("🔍 리밸런싱 주기, 거래비용, 팩터 조합 등 추가 실험을 추천합니다.")
    else:
        msg.append("🛠️ 팩터/전략 구조, 리스크 관리 방안을 재설계해보세요.")
    return '\n'.join(msg)
