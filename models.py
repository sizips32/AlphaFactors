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

class ModelTrainer:
    """모델 학습을 담당하는 클래스"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {}
        
        if torch.cuda.is_available():
            logger.info(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            logger.info("CPU 사용")
    
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
    
    def train_model(self, X: np.ndarray, y: np.ndarray, show_progress: bool = True) -> ImprovedFactorMLP:
        """모델 학습"""
        
        if len(X) == 0:
            st.error("학습 데이터가 없습니다.")
            return None
        
        # 데이터 준비
        train_loader, val_loader, scaler = self.prepare_data(X, y)
        
        # 모델 생성
        self.model = ImprovedFactorMLP(self.config, X.shape[1]).to(self.device)
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.lr_scheduler_patience, 
            factor=0.5, verbose=False
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
            progress_bar.empty()
            status_text.success(f"학습 완료! 최종 검증 손실: {best_val_loss:.6f}")
            self._plot_training_progress(train_losses, val_losses, loss_chart)
        
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
            self.model = ImprovedFactorMLP(self.config, input_dim).to(self.device)
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