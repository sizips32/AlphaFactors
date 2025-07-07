import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """딥러닝 모델 설정"""
    # 기본 모델 설정
    model_type: str = "mlp"  # "mlp", "lstm", "gru", "transformer", "cnn1d", "hybrid"
    
    # 네트워크 구조
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    
    # 학습 설정
    learning_rate: float = 0.001
    epochs: int = 20
    window_size: int = 10
    prediction_horizon: int = 5
    dropout_rate: float = 0.2
    batch_size: int = 64
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    
    # 시계열 모델 설정
    lstm_layers: int = 2
    lstm_bidirectional: bool = True
    gru_layers: int = 2
    gru_bidirectional: bool = True
    
    # Transformer 설정
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_ff_dim: int = 256
    
    # CNN 설정
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # 앙상블 설정
    ensemble_size: int = 3
    ensemble_method: str = "average"  # "average", "weighted", "voting"
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    qlib_data_path: str = os.path.expanduser("~/.qlib/qlib_data/us_data")
    local_data_path: str = "local_stock_data"
    default_ticker: str = "AAPL"
    default_start_date: str = "2020-01-01"
    default_end_date: str = "2023-01-01"
    required_columns: list = None
    cache_ttl: int = 3600  # 캐시 유효 시간 (초)
    max_cache_size: int = 1024 * 1024 * 100  # 최대 캐시 크기 (100MB)
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

@dataclass
class QlibConfig:
    """Qlib 백테스팅 설정"""
    default_instrument: str = "sp500"
    available_instruments: list = None
    topk: int = 10
    n_drop: int = 2
    default_benchmark: str = "sp500"
    
    def __post_init__(self):
        if self.available_instruments is None:
            self.available_instruments = ["sp500", "nasdaq100"]

@dataclass
class UIConfig:
    """UI 관련 설정"""
    page_title: str = "AlphaForge"
    page_icon: str = "📈"
    layout: str = "wide"
    sidebar_width: int = 300
    theme: str = "light"
    show_progress: bool = True
    auto_refresh: bool = False
    refresh_interval: int = 30  # 자동 새로고침 간격 (초)
    
    # 사용자 가이드 설정
    show_guide: bool = True
    guide_expanded: bool = True
    show_tooltips: bool = True
    
    # 시각화 설정
    chart_theme: str = "default"
    chart_height: int = 400
    chart_width: int = 800
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

@dataclass
class FactorConfig:
    """알파 팩터 생성 관련 설정"""
    momentum_lookback: int = 20
    reversal_lookback: int = 5
    volatility_lookback: int = 20
    volume_lookback: int = 20
    rsi_period: int = 14
    ma_period: int = 50
    ic_lookback: int = 60
    
    # 새로운 기술적 지표 파라미터
    bollinger_band_period: int = 20
    bollinger_band_std_dev: float = 2.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    stochastic_k_period: int = 14
    stochastic_d_period: int = 3
    
    williams_r_period: int = 14
    
    cci_period: int = 20
    
    money_flow_period: int = 14
    
    aroon_period: int = 25
    
    obv_period: int = 20
    
    volume_price_trend_period: int = 20
    
    chaikin_money_flow_period: int = 20
    
    force_index_period: int = 13
    
    ease_of_movement_period: int = 14
    
    accumulation_distribution_period: int = 20
    
    # 팩터 결합 설정
    default_weight_mode: str = "IC 기반 동적 가중치"
    min_ic_threshold: float = 0.01
    max_factor_weight: float = 0.5
    
    # 팩터 Zoo 설정
    zoo_auto_save: bool = True
    zoo_max_factors: int = 50
    zoo_cleanup_days: int = 30

@dataclass
class BacktestConfig:
    """백테스팅 관련 설정"""
    default_rebalance_freq: str = "M"  # 월간
    default_transaction_cost: float = 0.001  # 0.1%
    default_max_position: float = 0.1  # 10%
    default_long_only: bool = True
    
    # 리스크 지표 설정
    risk_free_rate: float = 0.02  # 2%
    confidence_level: float = 0.95  # VaR 계산용
    
    # 성과 지표 설정
    benchmark_ticker: str = "^GSPC"  # S&P 500
    performance_metrics: List[str] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = [
                'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
                'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta'
            ]

@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    model: ModelConfig = None
    data: DataConfig = None
    qlib: QlibConfig = None
    ui: UIConfig = None
    factor: FactorConfig = None
    backtest: BacktestConfig = None
    
    # 개발/디버그 설정
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_profiling: bool = False
    
    # 성능 설정
    max_workers: int = 4
    chunk_size: int = 1000
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.qlib is None:
            self.qlib = QlibConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.factor is None:
            self.factor = FactorConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
    
    def get_config_summary(self) -> dict:
        """설정 요약 반환"""
        return {
            'model': {
                'hidden_dims': [self.model.hidden_dim_1, self.model.hidden_dim_2],
                'learning_rate': self.model.learning_rate,
                'epochs': self.model.epochs
            },
            'data': {
                'cache_ttl': self.data.cache_ttl,
                'max_cache_size': self.data.max_cache_size
            },
            'factor': {
                'ic_lookback': self.factor.ic_lookback,
                'default_weight_mode': self.factor.default_weight_mode
            },
            'backtest': {
                'rebalance_freq': self.backtest.default_rebalance_freq,
                'transaction_cost': self.backtest.default_transaction_cost
            },
            'ui': {
                'theme': self.ui.theme,
                'show_guide': self.ui.show_guide
            }
        }
