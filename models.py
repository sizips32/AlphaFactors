"""
딥러닝 팩터 모델 및 학습 파이프라인 (개선된 버전)
Deep Learning Factor Models and Training Pipeline (Improved Version)
"""

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from typing import Tuple, List, Dict, Type, Any
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
        batch_size = x.size(0)
        
        # 입력 차원에 따라 처리 방식 결정
        if x.dim() == 2:
            # 2D input (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # 3D input (batch, seq_len, features) -> (batch, 1, seq_len * features)
            x = x.view(batch_size, 1, -1)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
        
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
        
        # LSTM 분기 (3D input 필요)
        if x.dim() == 2:
            # 2D input을 3D로 변환
            x_lstm = x.unsqueeze(1)
        else:
            x_lstm = x
        lstm_out, _ = self.lstm(x_lstm)
        lstm_features = lstm_out[:, -1, :]
        
        # CNN 분기 (1D Conv를 위한 처리)
        if x.dim() == 2:
            # 2D input (batch, features) -> (batch, 1, features)
            cnn_input = x.unsqueeze(1)
        elif x.dim() == 3:
            # 3D input (batch, seq_len, features) -> (batch, 1, seq_len * features)
            cnn_input = x.view(batch_size, 1, -1)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
        
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
        if self.config.model_type in ["lstm", "gru", "transformer", "hybrid"]:
            # 시계열 모델의 경우 (batch_size, seq_len, features)로 reshape
            # 하지만 현재 데이터는 이미 플래튼된 특징이므로 단순히 차원 추가
            X_train_scaled = X_train_scaled[:, np.newaxis, :]  # (batch_size, 1, features)
            X_val_scaled = X_val_scaled[:, np.newaxis, :]
        elif self.config.model_type == "cnn1d":
            # CNN1D는 2D input을 유지 (batch_size, features)
            pass

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
        
        if self.config.model_type in ["lstm", "gru", "transformer", "hybrid"]:
            # 학습 시와 동일한 방식으로 reshape
            X_scaled = X_scaled[:, np.newaxis, :]  # (batch_size, 1, features)
        elif self.config.model_type == "cnn1d":
            # CNN1D는 2D input을 유지
            pass

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy().flatten()

    def get_model_summary(self) -> Dict[str, Any]:
        """모델 정보 요약 반환"""
        if self.model is None:
            return {"error": "모델이 학습되지 않았습니다."}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            "model_type": self.config.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_dim": getattr(self.model, 'input_dim', 'Unknown'),
            "hidden_dims": [self.config.hidden_dim_1, self.config.hidden_dim_2],
            "learning_rate": self.config.learning_rate,
            "dropout_rate": self.config.dropout_rate
        }
        
        # 모델별 특정 정보 추가
        if hasattr(self.model, 'lstm'):
            summary["lstm_layers"] = self.config.lstm_layers
            summary["lstm_bidirectional"] = self.config.lstm_bidirectional
        elif hasattr(self.model, 'gru'):
            summary["gru_layers"] = self.config.gru_layers
            summary["gru_bidirectional"] = self.config.gru_bidirectional
        elif hasattr(self.model, 'transformer'):
            summary["transformer_layers"] = self.config.transformer_layers
            summary["transformer_heads"] = self.config.transformer_heads
        
        return summary
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'scaler': self.scaler,
            'training_history': self.training_history
        }
        
        torch.save(save_data, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        save_data = torch.load(filepath, map_location=self.device)
        
        # Config 복원
        for key, value in save_data['config'].items():
            setattr(self.config, key, value)
        
        # 모델 재구성
        model_class = self._model_map.get(self.config.model_type, ImprovedFactorMLP)
        
        # 모델 타입에 따라 input_dim 추출 방법 다름
        state_dict = save_data['model_state_dict']
        if self.config.model_type == "mlp":
            input_dim = state_dict['layers.0.weight'].shape[1]
        elif self.config.model_type in ["lstm", "gru"]:
            input_dim = state_dict['lstm.weight_ih_l0'].shape[1] if 'lstm.weight_ih_l0' in state_dict else state_dict['gru.weight_ih_l0'].shape[1]
        elif self.config.model_type == "transformer":
            input_dim = state_dict['embedding.weight'].shape[1]
        elif self.config.model_type == "cnn1d":
            # CNN1D의 경우 첫 번째 conv1d layer의 입력 차원에서 추정
            input_dim = 100  # 기본값으로 설정 (실제로는 더 정교한 방법 필요)
        elif self.config.model_type == "hybrid":
            input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        else:
            input_dim = 100  # 기본값
        
        self.model = model_class(self.config, input_dim).to(self.device)
        
        # 상태 복원
        self.model.load_state_dict(save_data['model_state_dict'])
        self.scaler = save_data['scaler']
        self.training_history = save_data.get('training_history', {})