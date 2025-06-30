"""
한글 폰트 설정 모듈
Korean Font Configuration for Matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import warnings
warnings.filterwarnings('ignore')

def setup_korean_font():
    """한글 폰트 설정"""
    
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # macOS 기본 한글 폰트들
            font_candidates = [
                "AppleGothic",
                "AppleSDGothicNeo-Regular", 
                "Noto Sans CJK KR",
                "Malgun Gothic",
                "나눔고딕",
                "NanumGothic"
            ]
            
        elif system == "Windows":  # Windows
            font_candidates = [
                "Malgun Gothic",
                "맑은 고딕",
                "Noto Sans CJK KR",
                "나눔고딕",
                "NanumGothic",
                "Dotum"
            ]
            
        else:  # Linux
            font_candidates = [
                "Noto Sans CJK KR",
                "NanumGothic",
                "나눔고딕",
                "DejaVu Sans"
            ]
        
        # 사용 가능한 폰트 찾기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        korean_font = None
        for font in font_candidates:
            if font in available_fonts:
                korean_font = font
                break
        
        if korean_font:
            # matplotlib 폰트 설정
            plt.rcParams['font.family'] = korean_font
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            
            print(f"✅ 한글 폰트 설정 완료: {korean_font}")
            return korean_font
        else:
            # 폰트를 찾지 못한 경우 fallback
            print("⚠️ 한글 폰트를 찾을 수 없어 기본 설정을 사용합니다.")
            plt.rcParams['axes.unicode_minus'] = False
            return None
            
    except Exception as e:
        print(f"⚠️ 폰트 설정 중 오류 발생: {e}")
        plt.rcParams['axes.unicode_minus'] = False
        return None

def get_available_korean_fonts():
    """사용 가능한 한글 폰트 목록 반환"""
    
    korean_keywords = ['gothic', 'malgun', 'nanum', 'noto', 'apple', '고딕', '나눔', '맑은']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    korean_fonts = []
    for font in available_fonts:
        font_lower = font.lower()
        if any(keyword in font_lower for keyword in korean_keywords):
            korean_fonts.append(font)
    
    return list(set(korean_fonts))  # 중복 제거

def test_korean_display():
    """한글 표시 테스트"""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 테스트 데이터
        x = [1, 2, 3, 4, 5]
        y = [10, 15, 8, 20, 12]
        
        ax.plot(x, y, marker='o', linewidth=2)
        ax.set_title("한글 폰트 테스트 - 포트폴리오 수익률")
        ax.set_xlabel("기간 (개월)")
        ax.set_ylabel("수익률 (%)")
        ax.grid(True, alpha=0.3)
        
        # 범례 추가
        ax.legend(["테스트 데이터"], loc='upper right')
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"한글 표시 테스트 실패: {e}")
        return None

def apply_korean_style():
    """한글에 최적화된 matplotlib 스타일 적용"""
    
    plt.style.use('default')  # 기본 스타일로 초기화
    
    # 한글 폰트 설정
    setup_korean_font()
    
    # 차트 스타일 설정
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })

def safe_title(text, fontsize=14, **kwargs):
    """안전한 한글 제목 설정"""
    try:
        plt.title(text, fontsize=fontsize, **kwargs)
    except:
        # 폰트 문제 발생 시 영어로 대체
        english_titles = {
            "포트폴리오 백테스팅 결과": "Portfolio Backtesting Results",
            "누적 수익률 비교": "Cumulative Returns Comparison", 
            "일별 수익률 분포": "Daily Returns Distribution",
            "드로우다운": "Drawdown",
            "월별 수익률": "Monthly Returns",
            "팩터별 누적 수익률 비교": "Factor Cumulative Returns Comparison",
            "팩터별 위험-수익 프로파일": "Risk-Return Profile by Factor"
        }
        english_text = english_titles.get(text, text)
        plt.title(english_text, fontsize=fontsize, **kwargs)

def safe_xlabel(text, fontsize=12, **kwargs):
    """안전한 한글 x축 라벨 설정"""
    try:
        plt.xlabel(text, fontsize=fontsize, **kwargs)
    except:
        english_labels = {
            "기간": "Period",
            "날짜": "Date", 
            "일별 수익률": "Daily Returns",
            "연간 변동성 (위험)": "Annual Volatility (Risk)",
            "월": "Month"
        }
        english_text = english_labels.get(text, text)
        plt.xlabel(english_text, fontsize=fontsize, **kwargs)

def safe_ylabel(text, fontsize=12, **kwargs):
    """안전한 한글 y축 라벨 설정"""
    try:
        plt.ylabel(text, fontsize=fontsize, **kwargs)
    except:
        english_labels = {
            "수익률": "Returns",
            "누적 수익률": "Cumulative Returns",
            "확률 밀도": "Probability Density", 
            "드로우다운 (%)": "Drawdown (%)",
            "연간 수익률": "Annual Returns",
            "연도": "Year"
        }
        english_text = english_labels.get(text, text)
        plt.ylabel(english_text, fontsize=fontsize, **kwargs)

# 모듈 로드 시 자동으로 한글 폰트 설정
setup_korean_font()