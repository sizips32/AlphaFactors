# AlphaFactors: 미국 주식 알파 팩터 + Qlib 백테스팅 플랫폼

## 프로젝트 소개

AlphaFactors는 미국 주식 데이터를 기반으로 다양한 알파 팩터를 생성하고, Qlib과 연동하여 전문적인 포트폴리오 백테스팅을 제공하는 **퀀트 투자 연구 플랫폼**입니다. Streamlit 기반의 웹 UI로 누구나 쉽게 팩터 연구와 전략 검증을 할 수 있습니다.

### ✨ 핵심 특징
- 🎯 **진정한 알파 팩터**: 횡단면 순위 기반 팩터 생성 (실제 퀀트 투자 방식)
- 🧠 **딥러닝 통합**: MLP, LSTM, Transformer 등 다양한 모델 지원
- 📊 **전문 백테스팅**: Qlib 기반 리스크 분석 및 성과 평가
- 🦁 **팩터 Zoo**: 실험 결과 저장/관리/재사용 시스템
- ⚡ **Mega-Alpha 신호**: 선형/비선형 팩터 동적 조합

---

## 주요 기능
- **📊 데이터 관리**: 미국 주식 실시간 데이터 다운로드 및 캐싱 (yfinance, FinanceDataReader)
- **🎯 팩터 생성**: 통계/기술적 팩터, 딥러닝 팩터, 공식 기반 팩터 생성 및 관리
- **📈 백테스팅**: Qlib 기반 전문 백테스팅 및 리스크 분석, 다양한 전략 지원
- **📊 시각화**: 누적 수익률, 리스크 지표, 수익률 분포, Rolling IC/ICIR 등 시각화
- **🦁 팩터 Zoo**: 실험 결과 저장/관리/재사용 시스템
- **⚡ 고급 분석**: 선형/비선형 팩터 성능 비교, Mega-Alpha 신호(동적 조합) 분석
- **🌐 웹 UI**: Streamlit 기반 직관적인 웹 인터페이스

---

## 설치 및 환경 준비

### 1. Python 환경
- Python 3.7 ~ 3.8 권장 (Qlib 호환성)
- 가상환경(venv, conda 등) 사용 권장

### 2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Qlib 미국 데이터셋 다운로드 (최초 1회)
```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```
- 위 명령어로 `~/.qlib/qlib_data/us_data` 경로에 데이터가 저장됩니다.
- 자세한 내용은 [Qlib 공식 설치 문서](https://qlib.readthedocs.io/en/latest/start/installation.html) 참고

---

## 실행 방법

```bash
streamlit run app.py
```
- 웹 브라우저에서 Streamlit UI가 열립니다.

---

## 전체 워크플로우 요약

1. **데이터 준비**: 종목(티커), 기간 입력 → OHLCV 데이터 다운로드 및 캐싱 (`data_handler.py`)
2. **팩터 생성**: 기본/커스텀/딥러닝 팩터 계산 및 결합 (`alpha_factors.py`, `models.py`)
3. **팩터 Zoo 저장/관리**: 팩터와 메타데이터를 pickle로 저장/불러오기/삭제 (`utils.py`)
4. **백테스팅**: Qlib 엔진으로 포트폴리오 성과 분석 (`portfolio_backtester.py`, `qlib_handler.py`)
5. **시각화/리포트**: 누적 수익률, 리스크 지표, rolling IC 등 Streamlit UI로 시각화
6. **선형/비선형 비교 및 Mega-Alpha 신호**: 팩터 Zoo에서 팩터 선택, 동적 조합 성능 분석

---

## 📁 폴더/파일별 역할 요약

| 파일/폴더                | 주요 역할 및 설명                                                                                 | 주요 클래스/함수 예시                      |
|--------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------|
| app.py                   | Streamlit 기반 메인 웹앱, 전체 UI/UX 및 워크플로우 관리                                         | AlphaForgeApp                              |
| config.py                | 환경설정, 공통 설정값 dataclass                                                                | ModelConfig, DataConfig, QlibConfig        |
| data_handler.py          | 주식 데이터 다운로드, 캐싱, 전처리                                                              | DataHandler, download_universe_data        |
| alpha_factors.py         | 알파 팩터 생성 엔진, 팩터별 계산 함수                                                          | AlphaFactorEngine, calculate_all_factors   |
| models.py                | 딥러닝/머신러닝 팩터(MLP, LSTM 등) 학습/예측                                                    | ModelTrainer, ImprovedFactorMLP 등         |
| portfolio_backtester.py  | Qlib 기반 백테스팅, 리스크 분석                                                                | PortfolioBacktester, FactorBacktester      |
| qlib_handler.py          | Qlib 연동, 데이터 변환, 백테스트 실행                                                          | QlibHandler                                |
| utils.py                 | 공통 유틸리티 함수, 팩터 Zoo 관리(저장/불러오기/삭제)                                           | save_factor_to_zoo, load_factors_from_zoo  |
| font_config.py           | 한글 폰트 설정                                                                                  | setup_korean_font 등                       |
| factor_zoo/              | 팩터 및 실험 결과 저장소 (pickle 파일)                                                          | 예: 20250708_084439_..._HYBRID.pkl         |
| local_stock_data/        | 다운로드/캐싱된 주식 데이터                                                                    | cache/, processed/                         |
| requirements.txt         | 의존성 패키지 목록                                                                              | -                                          |
| PRD.md                   | 상세 요구사항/워크플로우 문서                                                                  | -                                          |
| README.md                | 프로젝트 소개, 설치/실행법, 구조 설명                                                          | -                                          |

---

## 🧩 주요 함수/클래스 간단 설명 (주니어 개발자 참고)

- **AlphaForgeApp (app.py)**: 전체 UI/UX 및 워크플로우 관리
- **DataHandler (data_handler.py)**: 데이터 다운로드/캐싱/전처리 담당
- **AlphaFactorEngine (alpha_factors.py)**: 팩터 계산, IC/ICIR, 팩터 결합 등
- **ModelTrainer (models.py)**: 딥러닝/머신러닝 팩터 학습/예측
- **PortfolioBacktester (portfolio_backtester.py)**: Qlib 기반 백테스팅, 리스크 분석
- **QlibHandler (qlib_handler.py)**: Qlib 연동, 데이터 변환, 백테스트 실행
- **save_factor_to_zoo, load_factors_from_zoo (utils.py)**: 팩터 Zoo 저장/불러오기/삭제

---

## ⚠️ 실수 방지 팁 & 자주 하는 실수

- Python 3.7~3.8 환경 권장 (Qlib 호환성)
- Qlib 데이터셋 경로, yfinance/FinanceDataReader API 키 확인
- 팩터 계산 시 데이터 기간/인덱스 일치 여부 확인
- Streamlit 실행 시 requirements.txt 패키지 모두 설치 필요
- factor_zoo/ 및 local_stock_data/ 폴더 권한/경로 확인

---

## 💡 실제 사용 예시 (코드 스니펫)

```python
from data_handler import DataHandler
from alpha_factors import AlphaFactorEngine
from utils import save_factor_to_zoo

# 1. 데이터 준비
handler = DataHandler()
data = handler.download_universe_data(['AAPL', 'MSFT'], '2022-01-01', '2023-12-31')

# 2. 팩터 생성
engine = AlphaFactorEngine()
factors = engine.calculate_all_factors(data)

# 3. 팩터 저장
save_factor_to_zoo('my_factor', factors)
```

---

## 🏃‍♂️ 빠른 시작 요약

1. Python 3.7~3.8 환경 준비 (venv/conda 권장)
2. `pip install -r requirements.txt`
3. Qlib 데이터셋 다운로드 (README 상단 참고)
4. `streamlit run app.py`로 실행

---

## 🧑‍💻 주니어 개발자용 Q&A

- **Q: 팩터 계산이 안 돼요!**
  - 데이터 기간, 인덱스, 결측치 확인
  - requirements.txt 패키지 설치 여부 확인
- **Q: 백테스트 결과가 NaN이에요!**
  - 데이터 품질, 기간, 종목 수, 팩터 값 확인
- **Q: Streamlit UI가 깨져요!**
  - 한글 폰트 설정(font_config.py) 적용 여부 확인

---

## 기여 방법
- 이슈/PR 환영합니다!
- 추가 팩터, 멀티팩터, 고급 백테스팅 모델 등 확장 아이디어 언제든 제안해주세요.

---

## 참고자료
- [Qlib 공식 문서](https://qlib.readthedocs.io/en/latest/)
- [Qlib 미국 데이터셋 준비](https://qlib.readthedocs.io/en/latest/component/data.html)
- [yfinance 문서](https://github.com/ranaroussi/yfinance)

---

## 문의
- 추가 확장(여러 종목, 멀티팩터, 고급 딥러닝 모델 등)이나 에러 발생 시 언제든 문의해 주세요
