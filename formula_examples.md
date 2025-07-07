# 공식 기반 팩터 생성 파이프라인 가이드

## 📝 개요

공식 기반 팩터 생성 파이프라인은 사용자가 수학적 공식을 직접 입력하여 완전히 커스텀한 알파 팩터를 생성할 수 있는 강력한 도구입니다.

## 🚀 주요 기능

### 1. **템플릿 기반 생성**
- 미리 정의된 공식 템플릿 사용
- 카테고리별 분류 (기본, 기술적 지표, 고급, 복합)
- 빠른 팩터 생성

### 2. **직접 공식 입력**
- Python 문법으로 수학적 공식 작성
- 실시간 공식 검증
- 커스텀 파라미터 설정

### 3. **고급 편집기**
- 여러 공식 동시 관리
- 다양한 결합 방식
- 고급 설정 옵션

## 📊 사용 가능한 변수

### 기본 변수
- `price`, `close`: 종가 데이터
- `returns`: 수익률 (pct_change)
- `log_returns`: 로그 수익률
- `volume`: 거래량
- `high`, `low`, `open`: 고가, 저가, 시가 (현재는 종가와 동일)
- `t`: 시간 인덱스
- `n`: 데이터 포인트 수

### 사용자 정의 변수
- `window`: 기본 윈도우 크기
- `momentum_period`: 모멘텀 기간
- `volatility_period`: 변동성 기간
- `rsi_period`: RSI 기간
- `macd_fast`, `macd_slow`: MACD 파라미터

## 🔧 사용 가능한 함수

### 수학 함수
```python
abs(x)           # 절댓값
sqrt(x)          # 제곱근
log(x)           # 자연로그
log10(x)         # 상용로그
exp(x)           # 지수함수
sin(x), cos(x), tan(x)  # 삼각함수
```

### 통계 함수
```python
mean(x)          # 평균
std(x)           # 표준편차
sum(x)           # 합계
max(x, y), min(x, y)  # 최대값, 최소값
rank(x)          # 순위 (백분위)
```

### 시계열 함수
```python
rolling_mean(x, window)     # 이동평균
rolling_std(x, window)      # 이동표준편차
rolling_max(x, window)      # 이동최대값
rolling_min(x, window)      # 이동최소값
pct_change(x, periods)      # 변화율
diff(x, periods)            # 차분
shift(x, periods)           # 시프트
ewm_mean(x, span)           # 지수가중이동평균
ewm_std(x, span)            # 지수가중이동표준편차
```

### 정규화 함수
```python
zscore(x)        # Z-점수 정규화
normalize(x)     # 0-1 정규화
winsorize(x, limits)  # 윈저라이제이션
```

### 기술적 지표
```python
momentum(x, period)         # 모멘텀
volatility(x, period)       # 변동성
rsi(x, period)              # RSI
macd(x, fast, slow, signal) # MACD
bollinger_bands(x, period, std_dev)  # 볼린저 밴드
stochastic(x, k_period, d_period)    # 스토캐스틱
williams_r(x, period)       # Williams %R
cci(x, period)              # CCI
```

## 📋 공식 예제

### 1. 기본 팩터

#### 모멘텀 팩터
```python
momentum(price, 20)
```

#### 반전 팩터
```python
-momentum(price, 5)
```

#### 변동성 팩터
```python
-volatility(price, 20)
```

#### 거래량 가중 모멘텀
```python
momentum(price, 20) * normalize(volume)
```

### 2. 기술적 지표

#### RSI 기반 팩터
```python
rsi(price, 14)
```

#### MACD 기반 팩터
```python
macd(price, 12, 26, 9)
```

#### 볼린저 밴드 기반 팩터
```python
bollinger_bands(price, 20, 2)
```

### 3. 고급 팩터

#### 이동평균 크로스오버
```python
rolling_mean(price, 5) - rolling_mean(price, 20)
```

#### 변동성 브레이크아웃
```python
(price - rolling_mean(price, 20)) / rolling_std(price, 20)
```

#### 거래량 가중 가격
```python
price * normalize(volume)
```

#### 모멘텀 가속도
```python
momentum(price, 5) - momentum(price, 20)
```

### 4. 복합 팩터

#### 멀티 타임프레임 모멘텀
```python
(momentum(price, 5) + momentum(price, 10) + momentum(price, 20)) / 3
```

#### 변동성 조정 모멘텀
```python
momentum(price, 20) / (rolling_std(returns, 20) + 1e-8)
```

#### 거래량 가중 RSI
```python
rsi(price, 14) * normalize(volume)
```

#### 통합 기술적 지표
```python
(rsi(price, 14) + normalize(macd(price, 12, 26, 9)) + bollinger_bands(price, 20, 2)) / 3
```

## 🎯 고급 사용법

### 1. 조건부 팩터
```python
# 조건부 모멘텀
momentum(price, 20) * (returns > 0)
```

### 2. 다중 조건 팩터
```python
# 거래량이 높고 모멘텀이 양수일 때
momentum(price, 20) * (normalize(volume) > 0.5) * (momentum(price, 20) > 0)
```

### 3. 복잡한 수학적 표현
```python
# 지수 가중 모멘텀
exp(-0.1 * t) * momentum(price, 20)
```

### 4. 통계적 정규화
```python
# Z-점수 기반 모멘텀
zscore(momentum(price, 20))
```

## ⚙️ 파라미터 설정

### 기본 파라미터
```python
params = {
    'window': 20,           # 기본 윈도우 크기
    'momentum_period': 10,  # 모멘텀 기간
    'volatility_period': 20, # 변동성 기간
    'rsi_period': 14,       # RSI 기간
    'macd_fast': 12,        # MACD 빠른선
    'macd_slow': 26         # MACD 느린선
}
```

### 커스텀 파라미터
```python
params = {
    'custom_period': 30,
    'threshold': 0.05,
    'alpha': 0.1
}
```

## 🔄 결합 방식

### 1. IC 기반 가중치 (ic_weighted)
- 각 팩터의 Information Coefficient를 계산
- IC 절댓값에 비례하여 가중치 할당
- 가장 예측력이 높은 팩터에 높은 가중치

### 2. 동일 가중치 (equal_weight)
- 모든 팩터에 동일한 가중치 적용
- 간단하고 직관적

### 3. 개별 팩터만 (none)
- 팩터를 결합하지 않고 개별적으로 사용
- 각 팩터의 독립적인 성능 분석 가능

## 📈 성과 분석

### 생성되는 지표
- **IC (Information Coefficient)**: 팩터와 미래 수익률의 상관계수
- **IC 절댓값**: 예측력의 절댓값
- **팩터 표준편차**: 팩터의 변동성
- **팩터 평균**: 팩터의 중앙값
- **데이터 포인트 수**: 계산된 데이터의 총 개수

### 성과 해석
- **IC > 0.05**: 강한 예측력
- **IC > 0.02**: 중간 예측력
- **IC < 0.01**: 약한 예측력
- **IC < 0**: 역방향 예측력

## 🛡️ 보안 기능

### 공식 검증
- 문법 오류 검사
- 지원되지 않는 함수 검사
- 위험한 함수 사용 금지

### 안전한 실행 환경
- 제한된 함수만 사용 가능
- 시스템 명령어 실행 금지
- 메모리 사용량 제한

## 💡 팁과 모범 사례

### 1. 공식 작성 팁
- 간단하고 명확한 공식 사용
- 과적합 방지를 위한 적절한 윈도우 크기 설정
- 거래량 정보 활용으로 신호 강화

### 2. 성능 최적화
- 복잡한 공식은 단계별로 분해
- 캐싱 기능 활용
- 배치 처리로 효율성 향상

### 3. 위험 관리
- 극단값 처리 (winsorize 함수 활용)
- 정규화로 스케일 통일
- 다중 팩터로 분산 투자 효과

## 🔍 문제 해결

### 일반적인 오류
1. **문법 오류**: Python 문법 확인
2. **함수명 오류**: 지원되는 함수 목록 확인
3. **변수명 오류**: 사용 가능한 변수 확인
4. **나누기 오류**: 0으로 나누기 방지

### 디버깅 방법
1. 공식 검증 기능 활용
2. 단계별 계산 확인
3. 샘플 데이터로 테스트
4. 로그 메시지 확인

## 📚 추가 리소스

### 관련 문서
- [AlphaFactors 메인 가이드](../README.md)
- [팩터 Zoo 사용법](factor_zoo_guide.md)
- [백테스팅 가이드](backtesting_guide.md)

### 참고 자료
- [Quantitative Investment Strategies](https://www.quantitativeinvestment.com)
- [Technical Analysis](https://www.investopedia.com/technical-analysis)
- [Factor Investing](https://www.factorinvesting.com) 
