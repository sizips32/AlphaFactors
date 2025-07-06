# PRD: AlphaFactors - 미국 주식 알파 팩터 분석 및 백테스팅 플랫폼

- **문서 버전**: 1.2
- **작성일**: 2025년 6월 30일
- **작성자**: AlphaFactors 개발팀

---

## 1. 개요 (Overview)

AlphaFactors는 미국 주식 데이터를 기반으로 다양한 팩터(지표)를 생성하고, Qlib과 연동하여 포트폴리오 백테스팅, 리스크 분석, 시각화를 제공하는 플랫폼입니다. Streamlit 기반의 웹 UI로 팩터 연구와 전략 검증을 누구나 쉽게 할 수 있습니다.

---

## 2. 해결하고자 하는 문제 (Problem Statement)

- 기존 팩터 마이닝 및 백테스팅 환경의 진입장벽(복잡한 환경설정, 데이터 준비, 코드 작성 등)이 높음
- 팩터와 전략의 효과를 직관적으로 비교/분석하기 어려움
- 리스크 지표, 누적 수익률 등 핵심 성과지표를 한눈에 보기 어려움
- 팩터 실험 결과의 재현성, 관리, 비교가 어려움

---

## 3. 제안하는 솔루션 (Proposed Solution)

- 미국 주식 실시간 데이터 다운로드 및 관리
- 커스텀/기본/딥러닝 팩터 생성 및 관리
- Qlib 기반 백테스팅 및 리스크 분석 자동화
- 누적 수익률, 리스크 지표, 수익률 분포 등 시각화
- 팩터 Zoo(저장소) 기반 실험 결과 관리 및 재사용
- 선형/비선형 팩터 성능 비교, Mega-Alpha 신호(동적 조합) 분석
- Streamlit 기반 웹 UI 제공

---

## 4. 핵심 기능 요구사항 (Key Features)

### F1: 데이터 준비 및 관리
- F1.1: 사용자는 분석할 주식의 티커, 기간을 입력할 수 있다.
- F1.2: yfinance로 지정한 종목의 OHLCV 데이터를 다운로드한다.
- F1.3: 다운로드한 데이터는 `local_stock_data/`에 저장되어 재사용 가능하다.

### F2: 팩터 생성 및 관리
- F2.1: 기본 팩터(모멘텀, 반전, 변동성, 거래량, RSI, 이동평균 등) 또는 커스텀/딥러닝 팩터를 선택/생성할 수 있다.
- F2.2: 팩터 결합 시 IC 기반 동적 가중치 또는 사용자가 직접 입력한 고정 가중치 방식을 선택할 수 있다.
- F2.3: 팩터별/조합별 rolling IC/ICIR(예측력/안정성) 시계열을 자동 분석/시각화한다.
- F2.4: 팩터 생성 시 rolling IC/ICIR, 성능, 파라미터 등 메타데이터와 함께 팩터 Zoo(저장소)에 자동 저장된다.
- F2.5: 팩터 Zoo에서 저장된 팩터를 목록/상세/불러오기/삭제할 수 있다.

### F3: 백테스팅 및 리스크 분석
- F3.1: Qlib Alpha158 데이터와 팩터를 연동하여 백테스팅을 실행한다.
- F3.2: TopkDropoutStrategy 등 Qlib 전략을 적용한다.
- F3.3: 누적 수익률, 일별 수익률, Sharpe, IC 등 리스크 지표를 계산한다.
- F3.4: 여러 팩터(전략) 비교 기능을 제공한다.

### F4: 시각화 및 UI/UX
- F4.1: 누적 수익률 그래프, 리스크 지표 테이블, 일별 수익률 분포 등 다양한 시각화 제공
- F4.2: 팩터별/조합별 rolling IC/ICIR 시계열 그래프 제공
- F4.3: Streamlit 기반의 직관적인 웹 UI 제공 (탭: 데이터 준비, 팩터 생성, 딥러닝 팩터, 팩터 Zoo, 선형/비선형 비교)
- F4.4: 팩터 생성~백테스트~비교~조합까지 전체 워크플로우를 UI에서 원스톱으로 지원

### F5: 팩터 Zoo (실험 결과 저장/관리)
- F5.1: 팩터 생성 시 rolling IC/ICIR, 성능, 파라미터 등 메타데이터와 함께 자동 저장
- F5.2: 팩터 Zoo에서 저장된 팩터 목록, 상세 정보, rolling IC/ICIR, 파라미터, 성능지표 등 확인 가능
- F5.3: 팩터 불러오기(분석/백테스트 재사용), 삭제 기능 제공

### F6: 선형/비선형 비교 및 Mega-Alpha 신호
- F6.1: 팩터 Zoo에서 선형(통계/기술적)과 비선형(딥러닝) 팩터를 각각 선택해 rolling IC/ICIR, 백테스트 결과를 한 화면에서 비교
- F6.2: 두 팩터를 동적으로 조합(Mega-Alpha 신호)하여 rolling IC/ICIR, 성능, 시각화, 리포트 제공

---

## 5. 사용법 (How to Use)

### 1. 데이터 준비
- "투자 유니버스 구성" 탭에서 분석할 종목(티커)과 기간을 입력 후 데이터 다운로드

### 2. 팩터 생성
- "통계/기술적 팩터" 탭에서 팩터 종류, 결합 방식(동적/고정 가중치), 파라미터를 선택 후 팩터 생성
- "딥러닝 팩터" 탭에서 딥러닝 모델 파라미터를 설정 후 팩터 생성
- 팩터 생성 시 rolling IC/ICIR, 성능, 파라미터 등 메타데이터와 함께 팩터 Zoo에 자동 저장

### 3. 팩터 Zoo 활용
- "팩터 Zoo" 탭에서 저장된 팩터 목록, 상세 정보, rolling IC/ICIR, 파라미터, 성능지표 등 확인
- 원하는 팩터를 불러와 분석/백테스트에 재사용하거나, 삭제 가능

### 4. 선형/비선형 비교 및 Mega-Alpha 신호
- "선형/비선형 비교" 탭에서 팩터 Zoo의 선형/비선형 팩터를 각각 선택
- rolling IC/ICIR, 백테스트 결과를 한 화면에서 비교
- "Mega-Alpha 신호 생성/분석" 버튼으로 두 팩터의 동적 조합 신호 성능을 즉시 확인

### 5. 백테스팅 및 리포트
- "포트폴리오 백테스팅" 탭에서 팩터를 선택해 Qlib 기반 백테스트 실행
- 누적 수익률, Sharpe, IC 등 리스크 지표와 시각화 리포트 자동 제공

## 5-1. 전체 워크플로우 (Workflow)

AlphaFactors의 전체 워크플로우는 다음과 같이 구성됩니다. 각 단계별로 주요 파일, 클래스, 함수, 데이터 흐름을 상세히 설명합니다.

### 1. 데이터 준비 및 유니버스 구성
- **주요 파일/클래스**: `app.py`(AlphaForgeApp), `data_handler.py`(DataHandler)
- **주요 함수**: `DataHandler.download_universe_data`, `DataHandler.download_data`
- **설명**:
  - 사용자가 Streamlit UI에서 종목(티커)과 기간을 입력하면, `DataHandler`가 FinanceDataReader/yfinance를 통해 OHLCV 데이터를 다운로드합니다.
  - 데이터는 `local_stock_data/cache/`에 캐싱되어 재사용됩니다.
  - 데이터 전처리, 결측치 처리, 표준화 등은 `utils.py`의 함수들이 활용됩니다.

### 2. 알파 팩터 생성
- **주요 파일/클래스**: `app.py`, `alpha_factors.py`(AlphaFactorEngine)
- **주요 함수**: `AlphaFactorEngine.calculate_all_factors`, `calculate_momentum_factor`, `calculate_reversal_factor` 등
- **설명**:
  - 사용자가 팩터 종류, 파라미터, 결합 방식을 선택하면, `AlphaFactorEngine`이 각 팩터를 계산합니다.
  - 팩터별로 횡단면 순위화, 표준화, IC(Information Coefficient) 계산이 수행됩니다.
  - 팩터 결합은 IC 기반 동적 가중치(`combine_factors_ic_weighted`) 또는 고정 가중치(`combine_factors_fixed_weights`)로 처리됩니다.
  - 팩터별/조합별 rolling IC/ICIR, 성능 분석 결과가 반환됩니다.

### 3. 팩터 Zoo 저장 및 관리
- **주요 파일/클래스**: `app.py`, `utils.py`
- **주요 함수**: `save_factor_to_zoo`, `load_factors_from_zoo`, `delete_factor_from_zoo`
- **설명**:
  - 생성된 팩터와 메타데이터(rolling IC/ICIR, 파라미터, 성능 등)는 `factor_zoo/`에 pickle 파일로 저장됩니다.
  - 팩터 Zoo 탭에서 저장된 팩터를 불러오거나 삭제할 수 있습니다.

### 4. 딥러닝 팩터 생성
- **주요 파일/클래스**: `app.py`, `models.py`(ModelTrainer)
- **주요 함수**: `ModelTrainer.train`, `ModelTrainer.predict`
- **설명**:
  - 딥러닝 팩터 탭에서 모델 구조, 파라미터를 설정 후 학습/예측을 수행합니다.
  - 학습된 딥러닝 팩터 역시 팩터 Zoo에 저장됩니다.

### 5. 백테스팅 및 리스크 분석
- **주요 파일/클래스**: `app.py`, `portfolio_backtester.py`(PortfolioBacktester, FactorBacktester), `qlib_handler.py`(QlibHandler)
- **주요 함수**: `PortfolioBacktester.run_backtest`, `QlibHandler.run_qlib_backtest`
- **설명**:
  - 선택한 팩터(또는 조합)를 기반으로 Qlib 엔진을 통해 포트폴리오 백테스트를 실행합니다.
  - TopkDropoutStrategy 등 다양한 전략을 적용할 수 있습니다.
  - 누적 수익률, Sharpe, IC 등 리스크 지표가 계산되어 반환됩니다.

### 6. 시각화 및 리포트
- **주요 파일/클래스**: `app.py`, `utils.py`
- **주요 함수**: `analyze_factor_performance_text`, `analyze_backtest_performance_text`, Streamlit 내장 시각화 함수
- **설명**:
  - 백테스트 결과, 팩터 성능, 리스크 지표 등은 Streamlit UI에서 그래프, 표, 텍스트로 시각화됩니다.
  - 주요 지표는 한글로 해설이 제공되어 주니어 개발자도 쉽게 이해할 수 있습니다.

### 7. 선형/비선형 비교 및 Mega-Alpha 신호
- **주요 파일/클래스**: `app.py`, `alpha_factors.py`, `models.py`
- **주요 함수**: `AlphaFactorEngine.combine_factors_ic_weighted`, `ModelTrainer.train`, `analyze_factor_performance_text`
- **설명**:
  - 팩터 Zoo에서 선형/비선형 팩터를 각각 선택해 rolling IC/ICIR, 백테스트 결과를 비교합니다.
  - 두 팩터를 동적으로 조합(Mega-Alpha 신호)하여 성능을 즉시 분석할 수 있습니다.

---

## 6. 기술 스택 (Technology Stack)
- Python 3.7~3.8 (Qlib 호환)
- Streamlit (UI)
- pandas, numpy (데이터 처리)
- yfinance (미국 주식 데이터)
- pyqlib (백테스팅/리스크 분석)
- matplotlib, seaborn (시각화)
- torch (딥러닝 팩터)
- pickle (팩터 Zoo 저장)

---

## 7. 성공 지표 (Success Metrics)
- Qlib 백테스팅 결과 누적 수익률, Sharpe, IC 등 주요 지표가 UI에 정상적으로 시각화됨
- 커스텀/딥러닝/조합 팩터의 전략 비교가 가능함
- 팩터 Zoo를 통한 실험 결과 관리, 재현, 비교가 원활함
- 데이터 준비~팩터 생성~백테스팅~시각화까지 전체 흐름이 UI에서 원활하게 동작함

---

## 8. 향후 확장 계획 (Future Scope)
- 멀티팩터, 멀티종목, 고급 딥러닝 팩터 등 확장
- 사용자 정의 전략/팩터 업로드 기능
- 백테스팅 결과 자동 리포트/엑셀 내보내기
- 다양한 리스크 지표 및 시각화 추가
- Mega-Alpha 신호의 동적 가중치 최적화, 앙상블 등 고도화

---

## 9. 주니어 개발자 참고
- 각 기능별 목적, 사용법, 예상 결과를 코드와 UI에서 쉽게 확인할 수 있도록 설계
- 에러 발생 시 상세 메시지와 해결 방법을 UI에 안내
- 코드 구조와 데이터 흐름을 주석과 문서로 명확히 설명
- 팩터 Zoo, 선형/비선형 비교, Mega-Alpha 신호 등 고급 기능도 상세 주석과 예시로 안내
