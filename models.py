import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from config import ModelConfig
from utils import logger

class ImprovedFactorMLP(nn.Module):
    """개선된 팩터 예측 MLP 모델"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 드롭아웃 비율 설정
        dropout1 = config.dropout_rate
        dropout2 = config.dropout_rate * 0.5
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim_1),
            nn.BatchNorm1d(config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout1)
        )
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout2)
        )
        
        self.output_layer = nn.Linear(config.hidden_dim_2, 1)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        hidden = self.hidden_layers(features)
        output = self.output_layer(hidden)
        return output
    
    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """특징 중요도 계산 (그래디언트 기반)"""
        self.eval()
        x.requires_grad_(True)
        
        output = self.forward(x)
        gradients = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
        
        # 절댓값의 평균으로 중요도 계산
        importance = torch.abs(gradients).mean(dim=0).detach().cpu().numpy()
        return importance

class LSTMFactorModel(nn.Module):
    """LSTM 기반 팩터 예측 모델"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim_1,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim_1 * 2,  # bidirectional
            num_heads=4,
            dropout=config.dropout_rate
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1 * 2, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    # 2차원 이상의 텐서에만 xavier_uniform_ 적용
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Attention 적용
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # 마지막 시점의 출력 사용
        last_output = attn_out[:, -1, :]
        
        # FC 레이어
        output = self.fc_layers(last_output)
        return output

class GRUFactorModel(nn.Module):
    """GRU 기반 팩터 예측 모델"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim_1,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1 * 2, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    # 2차원 이상의 텐서에만 xavier_uniform_ 적용
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        
        # 마지막 시점의 출력 사용
        last_output = gru_out[:, -1, :]
        
        # FC 레이어
        output = self.fc_layers(last_output)
        return output

class TransformerFactorModel(nn.Module):
    """Transformer 기반 팩터 예측 모델"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 입력 임베딩
        self.input_projection = nn.Linear(input_dim, config.hidden_dim_1)
        
        # Positional Encoding - 충분히 큰 max_len 설정
        self.pos_encoder = PositionalEncoding(config.hidden_dim_1, max_len=2000)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim_1,
            nhead=8,
            dim_feedforward=config.hidden_dim_1 * 4,
            dropout=config.dropout_rate,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )
        
        # 출력 레이어
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 입력 프로젝션
        x = self.input_projection(x)
        
        # Positional Encoding 추가
        x = self.pos_encoder(x)
        
        # Transformer 인코더
        transformer_out = self.transformer(x)
        
        # 마지막 시점의 출력 사용
        last_output = transformer_out[:, -1, :]
        
        # 출력 레이어
        output = self.output_layers(last_output)
        return output

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (max_len, d_model) 형태로 저장
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (max_len, d_model)
        # x.size(1)은 시퀀스 길이
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class CNN1DFactorModel(nn.Module):
    """1D CNN 기반 팩터 예측 모델"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 1D CNN 레이어들
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(128, config.hidden_dim_1, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC 레이어들
        self.fc_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # CNN을 위해 차원 변환: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN 레이어들
        conv_out = self.conv_layers(x)
        
        # Global Average Pooling
        pooled = self.global_pool(conv_out).squeeze(-1)
        
        # FC 레이어들
        output = self.fc_layers(pooled)
        return output

class HybridFactorModel(nn.Module):
    """하이브리드 모델 (CNN + LSTM)"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # CNN 부분
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # LSTM 부분
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=config.hidden_dim_1,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # 출력 레이어
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_dim_1 * 2, config.hidden_dim_2),
            nn.BatchNorm1d(config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim_2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # CNN을 위해 차원 변환
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # CNN 레이어들
        cnn_out = self.cnn_layers(x)
        
        # LSTM을 위해 차원 변환
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_len, 64)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)
        
        # 마지막 시점의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # 출력 레이어
        output = self.output_layers(last_output)
        return output

class ModelTrainer:
    """모델 학습을 담당하는 클래스"""
    
    def __init__(self, config: ModelConfig, model_type: str = "mlp"):
        self.config = config
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        
        if torch.cuda.is_available():
            logger.info(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            logger.info("CPU 사용")
    
    def create_model(self, input_dim: int) -> nn.Module:
        """모델 타입에 따라 적절한 모델 생성"""
        if self.model_type == "mlp":
            return ImprovedFactorMLP(self.config, input_dim)
        elif self.model_type == "lstm":
            return LSTMFactorModel(self.config, input_dim)
        elif self.model_type == "gru":
            return GRUFactorModel(self.config, input_dim)
        elif self.model_type == "transformer":
            return TransformerFactorModel(self.config, input_dim)
        elif self.model_type == "cnn1d":
            return CNN1DFactorModel(self.config, input_dim)
        elif self.model_type == "hybrid":
            return HybridFactorModel(self.config, input_dim)
        else:
            logger.warning(f"알 수 없는 모델 타입: {self.model_type}. MLP를 사용합니다.")
            return ImprovedFactorMLP(self.config, input_dim)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, StandardScaler]:
        """데이터 전처리 및 DataLoader 생성"""
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, 
            random_state=42, shuffle=False  # 시계열 데이터이므로 shuffle=False
        )
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 텐서 변환
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # 시계열 모델의 경우 3D 텐서로 변환
        if self.model_type in ["lstm", "gru", "transformer", "cnn1d", "hybrid"]:
            # 시계열 데이터로 재구성 (window_size x features)
            X_train_reshaped, y_train_reshaped = self._reshape_for_sequence(X_train_tensor, y_train_tensor)
            X_val_reshaped, y_val_reshaped = self._reshape_for_sequence(X_val_tensor, y_val_tensor)
            
            X_train_tensor = X_train_reshaped
            y_train_tensor = y_train_reshaped
            X_val_tensor = X_val_reshaped
            y_val_tensor = y_val_reshaped
        
        # 텐서 크기 검증
        assert X_train_tensor.shape[0] == y_train_tensor.shape[0], f"X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}"
        assert X_val_tensor.shape[0] == y_val_tensor.shape[0], f"X_val: {X_val_tensor.shape}, y_val: {y_val_tensor.shape}"
        
        # DataLoader 생성
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, self.scaler
    
    def _reshape_for_sequence(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """시계열 모델을 위해 데이터를 3D로 재구성"""
        # (batch_size, features) -> (batch_size, seq_len, features)
        batch_size = X.shape[0]
        features = X.shape[1]
        
        # 시퀀스 길이 계산 (window_size 사용)
        seq_len = min(self.config.window_size, batch_size)
        
        # 패딩이 필요한 경우
        if batch_size < seq_len:
            # 패딩으로 seq_len 맞추기
            padding = torch.zeros(seq_len - batch_size, features)
            X = torch.cat([padding, X], dim=0)
            y_padding = torch.zeros(seq_len - batch_size, 1)
            y = torch.cat([y_padding, y], dim=0)
            batch_size = seq_len
        
        # 시퀀스로 재구성
        X_sequences = []
        y_sequences = []
        
        for i in range(batch_size - seq_len + 1):
            X_sequence = X[i:i+seq_len]
            y_sequence = y[i+seq_len-1]  # 시퀀스의 마지막 시점의 y값 사용
            
            X_sequences.append(X_sequence)
            y_sequences.append(y_sequence)
        
        if X_sequences:
            X_reshaped = torch.stack(X_sequences)
            y_reshaped = torch.stack(y_sequences)
            return X_reshaped, y_reshaped
        else:
            # 최소한의 시퀀스 생성
            X_reshaped = X.unsqueeze(0)
            y_reshaped = y.unsqueeze(0)
            return X_reshaped, y_reshaped
    
    def train_model(self, X: np.ndarray, y: np.ndarray, show_progress: bool = True) -> nn.Module:
        """모델 학습"""
        
        if len(X) == 0:
            st.error("학습 데이터가 없습니다.")
            return None
        
        # 데이터 준비
        train_loader, val_loader, scaler = self.prepare_data(X, y)
        
        # 모델 생성
        input_dim = X.shape[1] if self.model_type == "mlp" else X.shape[1]
        self.model = self.create_model(input_dim).to(self.device)
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.lr_scheduler_patience, 
            factor=0.5
        )
        
        loss_fn = nn.MSELoss()
        
        # 학습 기록
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 진행 상황 표시
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
        
        for epoch in range(self.config.epochs):
            # 훈련
            train_loss = self._train_epoch(train_loader, optimizer, loss_fn)
            train_losses.append(train_loss)
            
            # 검증
            val_loss = self._validate_epoch(val_loader, loss_fn)
            val_losses.append(val_loss)
            
            # 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 진행 상황 업데이트
            if show_progress:
                progress = (epoch + 1) / self.config.epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
                
                # 손실 차트 업데이트 (10 에포크마다)
                if (epoch + 1) % 10 == 0:
                    self._plot_training_progress(train_losses, val_losses, loss_chart)
            
            # 조기 종료
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"조기 종료: {epoch+1} 에포크에서 중단")
                break
        
        # 최고 모델 복원
        self.model.load_state_dict(best_model_state)
        
        # 학습 기록 저장
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text(f"학습 완료! 최고 검증 손실: {best_val_loss:.6f}")
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, loss_fn) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, loss_fn) -> float:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = loss_fn(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _plot_training_progress(self, train_losses: List[float], val_losses: List[float], 
                               container):
        """학습 진행 상황 시각화"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        container.pyplot(fig)
        plt.close(fig)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 생성"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 시계열 모델의 경우 3D 텐서로 변환
        if self.model_type in ["lstm", "gru", "transformer", "cnn1d", "hybrid"]:
            # 예측용 더미 y 텐서 생성 (실제로는 사용되지 않음)
            y_dummy = torch.zeros(X_tensor.shape[0], 1, dtype=torch.float32).to(self.device)
            X_tensor, _ = self._reshape_for_sequence(X_tensor, y_dummy)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """모델 요약 정보"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'model_type': 'ImprovedFactorMLP',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'input_dim': self.model.input_dim,
            'hidden_dims': [self.config.hidden_dim_1, self.config.hidden_dim_2],
            'dropout_rate': self.config.dropout_rate
        }
        
        if self.training_history:
            summary.update({
                'epochs_trained': self.training_history['epochs_trained'],
                'best_val_loss': self.training_history['best_val_loss'],
                'final_train_loss': self.training_history['train_losses'][-1],
                'final_val_loss': self.training_history['val_losses'][-1]
            })
        
        return summary
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            st.error("저장할 모델이 없습니다.")
            return
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'scaler': self.scaler,
                'training_history': self.training_history
            }, filepath)
            st.success(f"모델이 저장되었습니다: {filepath}")
        except Exception as e:
            st.error(f"모델 저장 실패: {e}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.config = checkpoint['config']
            self.scaler = checkpoint['scaler']
            self.training_history = checkpoint.get('training_history', {})
            
            # 모델 생성 및 가중치 로드
            input_dim = self.scaler.n_features_in_
            self.model = self.create_model(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            st.success(f"모델이 로드되었습니다: {filepath}")
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")

class EnsembleModel:
    """앙상블 모델 클래스"""
    
    def __init__(self, models: List[ModelTrainer]):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # 균등 가중치
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """앙상블 예측"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # 가중 평균
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """검증 데이터로 앙상블 가중치 최적화"""
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / np.sum(weights)  # 정규화
            pred = self.predict_with_weights(X_val, weights)
            mse = np.mean((pred - y_val) ** 2)
            return mse
        
        initial_weights = np.ones(len(self.models))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.weights = result.x
            logger.info(f"앙상블 가중치 최적화 완료: {self.weights}")
        else:
            logger.warning("앙상블 가중치 최적화 실패")
    
    def predict_with_weights(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """특정 가중치로 예측"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
