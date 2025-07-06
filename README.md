# AlphaFactors: 미국 주식 알파 팩터 + Qlib 백테스팅 플랫폼

## 프로젝트 소개

AlphaFactors는 미국 주식 데이터를 기반으로 다양한 팩터(지표)를 생성하고, Qlib과 연동하여 포트폴리오 백테스팅, 리스크 분석, 시각화를 제공하는 플랫폼입니다. Streamlit 기반의 웹 UI로 누구나 쉽게 팩터 연구와 전략 검증을 할 수 있습니다.

---

## 주요 기능
- 미국 주식 실시간 데이터 다운로드 및 캐싱 (yfinance, FinanceDataReader)
- 커스텀/기본/딥러닝 팩터 생성 및 관리
- Qlib 기반 백테스팅 및 리스크 분석, 다양한 전략 지원
- 누적 수익률, 리스크 지표, 수익률 분포 등 시각화
- 팩터 Zoo(저장소) 기반 실험 결과 관리 및 재사용
- 선형/비선형 팩터 성능 비교, Mega-Alpha 신호(동적 조합) 분석
- Streamlit 기반 웹 UI 제공

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

## 폴더/파일 구조

```
AlphaFactors/
  ├─ app.py                  # Streamlit 메인 앱 (AlphaForgeApp)
  ├─ config.py               # 환경설정 dataclass
  ├─ data_handler.py         # 데이터 다운로드/캐싱/전처리
  ├─ alpha_factors.py        # 팩터 생성 엔진
  ├─ models.py               # 딥러닝 팩터 학습/예측
  ├─ portfolio_backtester.py # Qlib 기반 백테스팅
  ├─ qlib_handler.py         # Qlib 연동/실행
  ├─ utils.py                # 유틸리티 함수/팩터 Zoo 관리
  ├─ font_config.py          # 한글 폰트 설정
  ├─ factor_zoo/             # 팩터/실험 결과 저장소 (pickle)
  ├─ local_stock_data/       # 다운로드/캐싱된 주식 데이터
  │    ├─ cache/
  │    └─ processed/
  ├─ requirements.txt
  ├─ PRD.md                  # 상세 요구사항/워크플로우 문서
  └─ README.md
```

---

## 주요 클래스/함수 역할

- **AlphaForgeApp (app.py)**: 전체 Streamlit UI 및 워크플로우 관리
- **DataHandler (data_handler.py)**: 데이터 다운로드, 캐싱, 전처리, 멀티티커 지원
- **AlphaFactorEngine (alpha_factors.py)**: 팩터 계산, IC/ICIR, 팩터 결합, 성능 분석
- **ModelTrainer (models.py)**: 딥러닝 팩터 학습/예측
- **PortfolioBacktester, FactorBacktester (portfolio_backtester.py)**: Qlib 기반 백테스팅, 전략 비교
- **QlibHandler (qlib_handler.py)**: Qlib 데이터 연동, 백테스트 실행
- **팩터 Zoo 관련 함수 (utils.py)**: 팩터 저장/불러오기/삭제, 성능 해설

---

## 사용 예시 (주요 시나리오)

1. **미국 주식 데이터 준비**
   - 티커(예: AAPL), 시작일, 종료일 입력 후 **데이터 다운로드**
2. **팩터 생성 및 관리**
   - 기본/커스텀/딥러닝 팩터 선택 및 생성, IC 기반 결합
3. **팩터 Zoo 활용**
   - 팩터 저장/불러오기/삭제, rolling IC/ICIR, 성능지표 확인
4. **백테스팅 실행**
   - Qlib 기반 포트폴리오 백테스트, 누적 수익률/Sharpe/IC 등 시각화
5. **선형/비선형 비교 및 Mega-Alpha 신호**
   - 팩터 Zoo에서 팩터 선택, 동적 조합 신호 성능 분석

---

## 자주 묻는 질문 (FAQ) 및 주니어 개발자 안내

- **Qlib 데이터셋 경로 오류**: 데이터가 올바른 경로에 있는지 확인
- **커스텀 팩터 날짜/인덱스 오류**: 팩터의 인덱스가 Qlib 데이터와 일치해야 함
- **'This type of signal is not supported'**: 팩터가 Series(MultiIndex: datetime, instrument)인지 확인
- **Python 3.9 이상**: Qlib 일부 기능이 제한될 수 있음 (3.7~3.8 권장)
- **코드/데이터 흐름**: 각 함수/클래스에 주석과 docstring이 상세히 작성되어 있음
- **에러 발생 시**: Streamlit UI에서 상세 메시지와 해결 방법 안내

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
