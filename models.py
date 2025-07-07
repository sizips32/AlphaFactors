"""
딥러닝 팩터 모델 및 학습 파이프라인 (개선된 버전)
Deep Learning Factor Models and Training Pipeline (Improved Version)
"""

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict, Type
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import ModelConfig
from utils import logger

# --- 1. 기본 모델 추상화 ---
class BaseFactorModel(nn.Module):
    """모든 팩터 모델의 기반이 되는 추상 클래스"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

    def _initialize_weights(self):
        """공통 가중치 초기화 로직"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

# --- 2. 개별 모델 아키텍처 정의 ---
class ImprovedFactorMLP(BaseFactorModel):
    """개선된 팩터 예측 MLP 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim_1),
            nn.BatchNorm1d(config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.hidden_dim_2, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

class LSTMFactorModel(BaseFactorModel):
    """LSTM 기반 팩터 예측 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim_1,
            num_layers=config.lstm_layers,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=config.lstm_bidirectional
        )
        hidden_dim = config.hidden_dim_1 * 2 if config.lstm_bidirectional else config.hidden_dim_1
        self.fc = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :] # 마지막 시점의 출력 사용
        return self.fc(last_output)

class GRUFactorModel(BaseFactorModel):
    """GRU 기반 팩터 예측 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim_1,
            num_layers=config.gru_layers,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=config.gru_bidirectional
        )
        hidden_dim = config.hidden_dim_1 * 2 if config.gru_bidirectional else config.hidden_dim_1
        self.fc = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.fc(last_output)

class TransformerFactorModel(BaseFactorModel):
    """Transformer 기반 팩터 예측 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        self.embedding = nn.Linear(input_dim, config.hidden_dim_1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim_1,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.transformer_layers)
        self.fc = nn.Linear(config.hidden_dim_1, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        return self.fc(pooled)

class CNN1DFactorModel(BaseFactorModel):
    """1D CNN 기반 팩터 예측 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        
        layers = []
        in_channels = 1  # 시계열 데이터를 1차원으로 취급
        
        for i, (out_channels, kernel_size) in enumerate(zip(config.cnn_channels, config.cnn_kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling으로 고정 크기 출력
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config.cnn_channels[-1], 1)
        self._initialize_weights()

    def forward(self, x):
        # (batch, seq_len, features) -> (batch, 1, seq_len * features)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        return self.fc(x)

class HybridFactorModel(BaseFactorModel):
    """LSTM + CNN 하이브리드 팩터 예측 모델"""
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__(config, input_dim)
        
        # LSTM 분기
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim_1 // 2,
            num_layers=config.lstm_layers,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=config.lstm_bidirectional
        )
        
        # CNN 분기
        self.conv1d = nn.Conv1d(1, config.hidden_dim_1 // 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(config.hidden_dim_1 // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 결합 레이어
        lstm_out_dim = config.hidden_dim_1 if config.lstm_bidirectional else config.hidden_dim_1 // 2
        combined_dim = lstm_out_dim + config.hidden_dim_1 // 2
        
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM 분기
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        # CNN 분기
        cnn_input = x.view(batch_size, 1, -1)
        cnn_out = self.conv1d(cnn_input)
        cnn_out = self.bn(cnn_out)
        cnn_out = torch.relu(cnn_out)
        cnn_features = self.pool(cnn_out).view(batch_size, -1)
        
        # 특징 결합
        combined = torch.cat([lstm_features, cnn_features], dim=1)
        return self.fc(combined)

# --- 3. 모델 트레이너 리팩토링 ---
class ModelTrainer:
    """모델 학습을 담당하는 클래스 (리팩토링된 버전)"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: nn.Module = None
        self.training_history = {}
        self._model_map = self._get_model_map()

    def _get_model_map(self) -> Dict[str, Type[BaseFactorModel]]:
        """모델 이름과 클래스를 매핑"""
        return {
            "mlp": ImprovedFactorMLP,
            "lstm": LSTMFactorModel,
            "gru": GRUFactorModel,
            "transformer": TransformerFactorModel,
            "cnn1d": CNN1DFactorModel,
            "hybrid": HybridFactorModel
        }

    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """데이터 전처리 및 DataLoader 생성"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, random_state=42, shuffle=False
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 시계열 모델을 위한 3D 텐서 변환
        if self.config.model_type in ["lstm", "gru", "transformer", "cnn1d", "hybrid"]:
            # (샘플 수, 특징 수) -> (샘플 수, 윈도우 크기, 특징 수)
            # 이 예제에서는 윈도우가 이미 데이터에 적용되었다고 가정하고, 차원만 추가합니다.
            X_train_scaled = X_train_scaled[:, np.newaxis, :]
            X_val_scaled = X_val_scaled[:, np.newaxis, :]

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader

    def _run_train_epoch(self, loader, optimizer, loss_fn) -> float:
        """한 에포크 훈련 실행"""
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _run_val_epoch(self, loader, loss_fn) -> float:
        """한 에포크 검증 실행"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = loss_fn(predictions, batch_y)
                total_loss += loss.item()
        return total_loss / len(loader)

    def train_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """모델 학습 파이프라인"""
        if len(X) == 0: return None

        train_loader, val_loader = self._prepare_data(X, y)
        
        model_class = self._model_map.get(self.config.model_type, ImprovedFactorMLP)
        input_dim = X.shape[1]
        self.model = model_class(self.config, input_dim).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.lr_scheduler_patience)
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(self.config.epochs):
            train_loss = self._run_train_epoch(train_loader, optimizer, loss_fn)
            val_loss = self._run_val_epoch(val_loader, loss_fn)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            progress_bar.progress((epoch + 1) / self.config.epochs)
            status_text.text(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"조기 종료: {epoch+1} 에포크에서 중단")
                break
        
        self.model.load_state_dict(best_model_state)
        self.training_history = {'train_losses': train_losses, 'val_losses': val_losses}
        status_text.text(f"학습 완료! 최고 검증 손실: {best_val_loss:.6f}")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 생성"""
        if self.model is None: raise ValueError("모델이 학습되지 않았습니다.")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        if self.config.model_type in ["lstm", "gru", "transformer", "cnn1d", "hybrid"]:
            X_scaled = X_scaled[:, np.newaxis, :]

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy().flatten()

    # ... (get_model_summary, save_model, load_model 등 나머지 함수는 기존과 유사하게 유지) ...