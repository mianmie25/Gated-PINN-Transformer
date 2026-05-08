import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. 配置超参数 ======================
config = {
    # 数据相关
    'data_dir': r'preprocessed_data_30%life_fft_downsample',
    'train_ratio': 0.7,
    'batch_size': 16,
    # 模型相关（Transformer参数）
    'hidden_size': 128,  # Transformer隐藏层维度
    'num_layers': 2,  # Transformer编码器层数
    'dropout_rate': 0.2,  # Dropout率
    'nhead': 8,  # 多头注意力头数（需整除hidden_size）
    # 训练相关
    'epochs': 2000,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'patience': 30000,
    'factor': 0.7,
    # 贝叶斯推断（MCD）相关
    'mc_samples': 30,
    # 设备相关
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 保存目录
    'save_metrics_dir': 'transformer_bayes_metrics',
    'random_seed': 30
}

# 创建结果保存目录
os.makedirs(config['save_metrics_dir'], exist_ok=True)
np.random.seed(config['random_seed'])
torch.manual_seed(config['random_seed'])


# ====================== 2. 自定义数据集 ======================
class FatigueDataset(Dataset):
    def __init__(self, data_dir, sample_ids=None):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, 'features')
        self.metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

        if sample_ids is not None:
            self.metadata = self.metadata[self.metadata['sample_id'].isin(sample_ids)].reset_index(drop=True)

        self.features = []
        self.labels = []
        for _, row in self.metadata.iterrows():
            feat_path = os.path.join(self.features_dir, row['feature_file'])
            feat = pd.read_csv(feat_path, usecols=['normalized_force']).values
            self.features.append(feat)
            self.labels.append(row['log_remaining_life'])

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx]).unsqueeze(0)

# ====================== 3. Transformer回归模型 ======================
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_size=128, num_layers=2, dropout_rate=0.2, nhead=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.nhead = nhead

        # 输入投影层（将1维特征映射到hidden_size维）
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # 位置编码（可学习）- 修改为(seq_len, hidden_size)格式
        self.positional_encoding = nn.Parameter(torch.zeros(1000, hidden_size))  # 假设最大序列长度1000
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            batch_first=False  # 保持与LSTM相同的输入格式 (seq_len, batch, hidden_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层（与原LSTM结构完全一致）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # 只初始化2D及以上参数
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # 输入x形状: (batch_size, seq_len, input_dim)
        # 转换为Transformer期望的格式: (seq_len, batch, hidden_size)
        x = x.permute(1, 0, 2)  # (seq_len, batch, input_dim)
        x = self.input_proj(x)  # (seq_len, batch, hidden_size)

        # 添加位置编码 - 修复维度问题
        seq_len = x.size(0)
        pos_enc = self.positional_encoding[:seq_len, :]  # (seq_len, hidden_size)
        pos_enc = pos_enc.unsqueeze(1)  # (seq_len, 1, hidden_size) - 添加batch维度以便广播
        x = x + pos_enc  # 广播到 (seq_len, batch, hidden_size)

        # Transformer编码
        transformer_out = self.transformer_encoder(x)

        # 取最后一个时间步的输出（与LSTM行为一致）
        last_output = transformer_out[-1, :, :]

        # 全连接层
        output = self.fc(last_output)
        return output

# ====================== 4. 训练函数 ======================
def train_model(model, train_loader, test_loader, config):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['factor'], patience=config['patience'] // 2
    )

    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0.0001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float('inf')
            self.best_model_state = None

        def __call__(self, val_loss, model):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    early_stopping = EarlyStopping(patience=config['patience'], min_delta=1e-4)
    train_history = {'train_loss': [], 'test_loss': []}
    device = config['device']

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_feat, batch_label in train_loader:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            outputs = model(batch_feat)
            loss = criterion(outputs, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_feat.size(0)

        train_loss /= len(train_loader.dataset)
        train_history['train_loss'].append(train_loss)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_feat, batch_label in test_loader:
                batch_feat = batch_feat.to(device)
                batch_label = batch_label.to(device)
                outputs = model(batch_feat)
                loss = criterion(outputs, batch_label)
                test_loss += loss.item() * batch_feat.size(0)

        test_loss /= len(test_loader.dataset)
        train_history['test_loss'].append(test_loss)

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        if early_stopping(test_loss, model):
            print(f'Early stopping at epoch {epoch + 1}, best test loss: {early_stopping.best_loss:.6f}')
            model.load_state_dict(early_stopping.best_model_state)
            break

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return model, train_history


# ====================== 5. 贝叶斯预测函数 ======================
def bayesian_predict(model, test_loader, config):
    model.eval()
    device = config['device']
    y_true = []
    y_pred_samples = []

    # 强制开启所有Dropout层
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            m.p = config['dropout_rate']

    with torch.no_grad():
        for batch_feat, batch_label in test_loader:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            batch_samples = []
            for _ in range(config['mc_samples']):
                output = model(batch_feat)
                batch_samples.append(output.cpu().numpy())

            batch_samples = np.array(batch_samples)
            y_pred_samples.append(batch_samples)
            y_true.append(batch_label.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred_samples = np.concatenate(y_pred_samples, axis=1)
    y_pred_samples = y_pred_samples.squeeze(-1)

    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0) + 1e-6

    # 转换为实际寿命
    y_true_actual = 10 ** y_true
    y_pred_mean_actual = 10 ** y_pred_mean
    y_pred_actual_samples = 10 ** y_pred_samples
    y_pred_std_actual = np.std(y_pred_actual_samples, axis=0) + 1e-6

    return y_true, y_pred_mean, y_pred_std, y_pred_samples, y_true_actual, y_pred_mean_actual, y_pred_std_actual


# ====================== 6. 指标计算 ======================
def calculate_metrics(y_true, y_pred_mean, y_true_actual, y_pred_mean_actual):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Log10尺度指标
    mae_log = mean_absolute_error(y_true, y_pred_mean)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2_log = r2_score(y_true, y_pred_mean)

    # 实际寿命尺度指标
    mae_actual = mean_absolute_error(y_true_actual, y_pred_mean_actual)
    rmse_actual = np.sqrt(mean_squared_error(y_true_actual, y_pred_mean_actual))
    r2_actual = r2_score(y_true_actual, y_pred_mean_actual)

    # 相对误差
    relative_error = np.mean(np.abs((y_pred_mean_actual - y_true_actual) / (y_true_actual + 1e-8))) * 100

    metrics = {
        'mae_log': mae_log,
        'rmse_log': rmse_log,
        'r2_log': r2_log,
        'mae_actual': mae_actual,
        'rmse_actual': rmse_actual,
        'r2_actual': r2_actual,
        'relative_error_pct': relative_error
    }

    return metrics


def analyze_uncertainty(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual):
    from scipy.stats import pearsonr

    print("\n===== 不确定性可靠性分析 =====")

    error_log = np.abs(y_pred_mean - y_true)
    error_actual = np.abs(y_pred_mean_actual - y_true_actual)

    def clean_nan_inf(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred_std = clean_nan_inf(y_pred_std)
    error_log = clean_nan_inf(error_log)
    y_pred_std_actual = clean_nan_inf(y_pred_std_actual)
    error_actual = clean_nan_inf(error_actual)

    def safe_pearsonr(x, y):
        if np.var(x) < 1e-8 or np.var(y) < 1e-8:
            return np.nan, np.nan
        mask = (x != x[0]) | (y != y[0])
        if np.sum(mask) < 2:
            return np.nan, np.nan
        return pearsonr(x[mask], y[mask])

    corr_log, p_value_log = safe_pearsonr(y_pred_std, error_log)
    corr_actual, p_value_actual = safe_pearsonr(y_pred_std_actual, error_actual)

    print(f"Log10尺度：标准差与实际误差的相关系数 = {corr_log:.4f}" if not np.isnan(
        corr_log) else "Log10尺度：相关系数计算失败")
    print(f"实际寿命尺度：标准差与实际误差的相关系数 = {corr_actual:.4f}" if not np.isnan(
        corr_actual) else "实际寿命尺度：相关系数计算失败")

    uncertainty_metrics = {
        'corr_log': corr_log if not np.isnan(corr_log) else -999,
        'p_value_log': p_value_log if not np.isnan(p_value_log) else -999,
        'corr_actual': corr_actual if not np.isnan(corr_actual) else -999,
        'p_value_actual': p_value_actual if not np.isnan(p_value_actual) else -999
    }

    return uncertainty_metrics


def save_metrics(metrics, config, filename):
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(config['save_metrics_dir'], f'{filename}.csv'), index=False)
    print(f"\n指标已保存到 {config['save_metrics_dir']}/{filename}.csv")


def print_metrics(metrics):
    print(f"Log10尺度 - MAE: {metrics['mae_log']:.4f}, RMSE: {metrics['rmse_log']:.4f}, R²: {metrics['r2_log']:.4f}")
    print(
        f"实际寿命尺度 - MAE: {metrics['mae_actual']:.2f}, RMSE: {metrics['rmse_actual']:.2f}, R²: {metrics['r2_actual']:.4f}")
    print(f"实际寿命相对误差: {metrics['relative_error_pct']:.2f}%")


# ====================== 7. 主流程 ======================
if __name__ == '__main__':
    # 加载数据
    full_dataset = FatigueDataset(config['data_dir'])
    n_samples = len(full_dataset)
    sample_ids = np.arange(n_samples)

    # 划分训练/测试集
    from sklearn.model_selection import train_test_split

    train_ids, test_ids = train_test_split(
        sample_ids, train_size=config['train_ratio'], random_state=config['random_seed']
    )
    train_dataset = FatigueDataset(config['data_dir'], sample_ids=train_ids)
    test_dataset = FatigueDataset(config['data_dir'], sample_ids=test_ids)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 初始化Transformer模型（替换原LSTM）
    model = TransformerRegressor(
        input_dim=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        nhead=config['nhead']
    ).to(config['device'])

    print(f"\nModel initialized on device: {config['device']}")
    print(f"Model parameters: hidden_size={config['hidden_size']}, num_layers={config['num_layers']}")

    # 训练模型
    print("\n===== 开始训练模型 =====")
    model, train_history = train_model(model, train_loader, test_loader, config)

    # 贝叶斯预测
    print("\n===== 开始贝叶斯预测 =====")
    results = bayesian_predict(model, test_loader, config)
    y_true, y_pred_mean, y_pred_std, y_pred_samples, y_true_actual, y_pred_mean_actual, y_pred_std_actual = results

    # 计算指标
    print("\n===== 基础性能指标 =====")
    metrics = calculate_metrics(y_true, y_pred_mean, y_true_actual, y_pred_mean_actual)
    print_metrics(metrics)
    save_metrics(metrics, config, 'base_metrics')

    # 不确定性分析
    print("\n===== 不确定性可靠性分析 =====")
    uncertainty_metrics = analyze_uncertainty(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual,
                                              y_pred_std_actual)
    save_metrics(uncertainty_metrics, config, 'uncertainty_metrics')

    # 保存预测结果
    results_df = pd.DataFrame({
        'true_log_life': y_true,
        'pred_log_life_mean': y_pred_mean,
        'pred_log_life_std': y_pred_std,
        'true_actual_life': y_true_actual,
        'pred_actual_life_mean': y_pred_mean_actual,
        'pred_actual_life_std': y_pred_std_actual
    })
    results_df.to_csv(os.path.join(config['save_metrics_dir'], 'prediction_results.csv'), index=False)
    print(f"\n预测结果已保存到 {config['save_metrics_dir']}/prediction_results.csv")
    print("\n===== 分析完成 =====")
