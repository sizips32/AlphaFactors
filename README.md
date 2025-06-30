# AlphaFactors: 미국 주식 알파 팩터 + Qlib 백테스팅 플랫폼

## 프로젝트 소개

AlphaFactors는 미국 주식 데이터를 기반으로 다양한 팩터(지표)를 생성하고, Qlib과 연동하여 포트폴리오 백테스팅 및 리스크 분석, 시각화를 제공하는 플랫폼입니다. Streamlit 기반의 웹 UI로 누구나 쉽게 팩터 연구와 전략 검증을 할 수 있습니다.

---

## 주요 기능
- **미국 주식 실시간 데이터 다운로드 (yfinance)**
- **커스텀/기본 팩터 생성 및 관리**
- **Qlib 기반 백테스팅 및 리스크 분석**
- **누적 수익률, 리스크 지표, 수익률 분포 등 시각화**
- **Streamlit 기반 웹 UI 제공**

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

## 사용법 (주요 시나리오)

### 1. 미국 주식 데이터 준비
- 티커(예: AAPL), 시작일, 종료일 입력 후 **데이터 다운로드** 버튼 클릭
- yfinance로 일별 OHLCV 데이터를 불러옵니다.

### 2. 팩터 생성 및 관리
- 기본 팩터(예: RESI5) 또는 커스텀 팩터를 선택/생성
- 커스텀 팩터는 날짜 인덱스와 Qlib 데이터 날짜가 일치해야 합니다.

### 3. 백테스팅 실행
- **Qlib 백테스팅 실행** 버튼 클릭
- Qlib Alpha158 데이터와 팩터를 연동하여 포트폴리오 누적 수익률, 리스크 지표(Sharpe, IC 등)를 시각화합니다.

### 4. 결과 시각화 및 분석
- 누적 수익률 그래프, 리스크 지표 테이블, 일별 수익률 분포 등 다양한 시각화 제공
- 여러 전략(팩터) 비교 기능 지원

---

## 에러 대처법
- Qlib 데이터셋 경로 오류: 데이터가 올바른 경로에 있는지 확인
- 커스텀 팩터 날짜/인덱스 오류: 팩터의 인덱스가 Qlib 데이터와 일치하는지 확인
- "This type of signal is not supported": 팩터가 Series(MultiIndex: datetime, instrument)인지 확인
- Python 3.9 이상에서는 Qlib 일부 기능이 제한될 수 있음

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
- 추가 확장(여러 종목, 멀티팩터, 고급 딥러닝 모델 등)이나 에러 발생 시 언제든 문의해 주세요!
