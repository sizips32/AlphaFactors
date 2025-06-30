# AlphaForge: 미국주식 딥러닝 팩터 + Qlib 백테스팅 데모

## 프로젝트 소개

이 프로젝트는 미국 주식 데이터를 기반으로 딥러닝(MLP)으로 팩터를 생성하고, Qlib의 커스텀 데이터 연동 및 백테스팅을 Streamlit UI로 제공합니다.

- **미국 주식 실시간 데이터 다운로드 (yfinance)**
- **딥러닝 팩터 마이닝 (PyTorch MLP)**
- **Qlib 미국 데이터셋 기반 커스텀 팩터 백테스팅**
- **Streamlit 기반 웹 UI**

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
Qlib 공식 데이터셋을 반드시 다운로드해야 합니다.

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

---

## 주요 기능 및 사용법

### 1. 미국 주식 데이터 준비
- 티커(예: AAPL), 시작일, 종료일 입력 후 **데이터 다운로드** 버튼 클릭
- yfinance로 미국 주식 일별 OHLCV 데이터를 불러옵니다.

### 2. 딥러닝 팩터 마이닝 (MLP)
- **딥러닝 팩터 학습/생성** 버튼 클릭
- 최근 시계열 데이터를 입력으로 미래 수익률을 예측하는 MLP 모델을 학습합니다.
- 예측값(=커스텀 팩터)을 시계열로 시각화합니다.

### 3. Qlib 미국 백테스팅
- **Qlib 백테스팅 실행** 버튼 클릭
- Qlib 미국 데이터셋(Alpha158)과 커스텀 팩터를 연동하여 포트폴리오 누적 수익률, 리스크 지표(샤프지수 등)를 시각화합니다.

---

## requirements.txt 예시
```
streamlit
pandas
numpy
FinanceDataReader
torch
pyqlib
scikit-learn
matplotlib
yfinance
```

---

## 주의사항 및 팁
- Qlib는 반드시 `pip install pyqlib`로 설치해야 하며, 데이터셋도 공식 명령어로 다운로드해야 합니다.
- Qlib 관련 코드는 Qlib 소스 디렉토리(qlib/) 밖에서 실행해야 import 에러가 발생하지 않습니다.
- 딥러닝 팩터의 인덱스(날짜)가 Qlib 데이터와 일치해야 하며, 교집합만 사용해야 에러가 발생하지 않습니다.
- Python 3.9 이상에서는 일부 Qlib 기능이 제한될 수 있습니다. (3.7~3.8 권장)
- 데이터가 없거나 기간이 짧으면 학습/백테스팅이 불가하니 충분한 기간을 선택하세요.

---

## 참고 자료
- [Qlib 공식 문서](https://qlib.readthedocs.io/en/latest/)
- [Qlib 미국 데이터셋 준비](https://qlib.readthedocs.io/en/latest/component/data.html)
- [yfinance 문서](https://github.com/ranaroussi/yfinance)

---

## 문의
- 추가 확장(여러 종목, 멀티팩터, 고급 딥러닝 모델 등)이나 에러 발생 시 언제든 문의해 주세요!
