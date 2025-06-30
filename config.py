import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """딥러닝 모델 설정"""
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    learning_rate: float = 0.001
    epochs: int = 20
    window_size: int = 10
    prediction_horizon: int = 5
    dropout_rate: float = 0.2
    batch_size: int = 64
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    qlib_data_path: str = os.path.expanduser("~/.qlib/qlib_data/us_data")
    local_data_path: str = "local_stock_data"
    default_ticker: str = "AAPL"
    default_start_date: str = "2020-01-01"
    default_end_date: str = "2023-01-01"
    required_columns: list = None
    
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

@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    model: ModelConfig = None
    data: DataConfig = None
    qlib: QlibConfig = None
    ui: UIConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.qlib is None:
            self.qlib = QlibConfig()
        if self.ui is None:
            self.ui = UIConfig()