import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import os
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. 配置超参数（优化：调整超参数以提升性能） ======================
config = {
    # 数据相关
    'data_dir': r'preprocessed_data_30%life_fft_downsample',
    'train_ratio': 0.7,
    'batch_size': 16,
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 4,
    'dim_feedforward': 128,
    'dropout_rate': 0.2,
    'gate_attention': False,
    'epochs': 35000,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'patience': 35000,
    'factor': 0.7,
    'mc_samples': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_fig_dir': 'transformer_bayes_gated_stage3_results',
    'save_metrics_dir': 'stage3_metrics',
    'visualize_attention': True,  # 是否可视化注意力权重
    'uncertainty_analysis': True,  # 是否进行不确定性可靠性分析
    'random_seed': 42,  # 固定随机种子保证可复现
    'cross_section_area': 256,        # 试件横截面积 (mm²)
    'elastic_modulus': 210e3,         # 弹性模量 (MPa，210GPa = 210×10³ MPa)
    'poisson_ratio': 0.3,             # 泊松比
    'yield_strength': 980,            # 屈服强度 (MPa)
    'pinn_loss_weight': 0.01,          # 物理损失权重（可调整，平衡MSE损失和物理损失）
}

# 创建结果保存目录
os.makedirs(config['save_fig_dir'], exist_ok=True)
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
            # ========== 修改：读取两列（归一化轴向力 + 归一化塑性应变） ==========
            feat = pd.read_csv(feat_path, usecols=['normalized_force', 'normalized_plastic_strain']).values
            self.features.append(feat)
            self.labels.append(row['log_remaining_life'])

        # 现在features形状为 (n_samples, 1000, 2)，兼容原有维度逻辑
        self.features = np.array(self.features, dtype=np.float32)  # (n_samples, 1000, 2)
        self.labels = np.array(self.labels, dtype=np.float32)  # (n_samples,)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx]).unsqueeze(0)




# ====================== 4. 位置编码 ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


# ====================== 5. 带门控注意力的Transformer模型 ======================
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=config['nhead'], num_encoder_layers=config['num_encoder_layers'],
                 dim_feedforward=config['dim_feedforward'], dropout_rate=config['dropout_rate'], gate_attention=True):
        super().__init__()
        self.d_model = d_model
        self.gate_attention = gate_attention  # 保留参数但不再使用
        self.dropout_rate = dropout_rate  # 统一Dropout概率

        # 输入嵌入 + Dropout（input_dim=2，适配两列特征：归一化轴向力+归一化塑性应变）
        self.embedding = nn.Linear(input_dim, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)  # 嵌入层Dropout

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # Transformer Encoder（强化Dropout）
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,  # Encoder层Dropout
            batch_first=False,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 池化后添加Dropout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool_dropout = nn.Dropout(dropout_rate)  # 池化层Dropout

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # ========== 核心修复：补充 _init_weights 方法 ==========
        self._init_weights()

    # ========== 定义权重初始化函数 ==========
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层用xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                # 层归一化初始化
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # 前向传播（移除所有门控注意力逻辑）
    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = self.embedding_dropout(x)  # 嵌入层Dropout
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)

        attn_weights = None
        gate_values = None  # 保留变量避免调用处报错，始终为None

        # 【完全删除】门控注意力调用逻辑
        # 直接执行Transformer Encoder
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(-1)
        x = self.pool_dropout(x)  # 池化后Dropout

        out = self.fc(x)
        if return_attention:
            return out, attn_weights, gate_values  # gate_values始终为None
        else:
            return out


# ====================== PINN物理约束损失函数 ======================
def pinn_physical_loss(normalized_force, normalized_plastic_strain, config):
    device = config['device']
    # 固定物理参数（不变）
    A = torch.tensor(config['cross_section_area'], device=device, dtype=torch.float32)  # 256 mm²
    E = torch.tensor(config['elastic_modulus'], device=device, dtype=torch.float32)    # 210e3 MPa
    sigma_y = torch.tensor(config['yield_strength'], device=device, dtype=torch.float32)# 980 MPa

    # ====================== 关键：直接用归一化数据计算 ======================
    # 归一化应力 = 归一化力 / 截面积
    stress = normalized_force / A

    # 归一化弹性应变
    elastic_strain = stress / E

    # 损失1：弹性阶段（应力≤屈服强度）→ 塑性应变应≈0
    elastic_mask = (stress <= sigma_y).float()
    loss_elastic = torch.mean((normalized_plastic_strain * elastic_mask) ** 2)

    # 损失2：塑性阶段（应力>屈服强度）→ 本构关系约束
    plastic_mask = (stress > sigma_y).float()
    stress_pred = E * (elastic_strain + normalized_plastic_strain)
    loss_plastic = torch.mean(((stress - stress_pred) * plastic_mask) ** 2)

    # 损失3：塑性应变非负（物理约束）
    loss_strain_non_neg = torch.mean(torch.relu(-normalized_plastic_strain) ** 2)

    # 总物理损失
    total_physical_loss = loss_elastic + loss_plastic + loss_strain_non_neg
    return total_physical_loss


# ====================== 6. 训练函数 ======================
def train_model(model, train_loader, test_loader, config):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['factor'], patience=config['patience'] // 2
    )

    # 早停机制
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

            # 原有MSE预测损失
            mse_loss = criterion(outputs, batch_label)

            # 提取归一化轴向力、塑性应变
            normalized_force = batch_feat[..., 0:1]
            normalized_plastic_strain = batch_feat[..., 1:2]

            # 计算PINN物理约束损失
            physical_loss = pinn_physical_loss(normalized_force, normalized_plastic_strain, config)

            # 训练总损失 = MSE + 权重×PINN损失
            loss = mse_loss + config['pinn_loss_weight'] * physical_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_feat.size(0)

        # 训练损失归一化
        train_loss /= len(train_loader.dataset)
        train_history['train_loss'].append(train_loss)

        # ====================== 测试阶段加入PINN损失 ======================
        model.eval()
        test_loss = 0.0
        with torch.no_grad():  # 仍保留no_grad，避免测试阶段计算梯度
            for batch_feat, batch_label in test_loader:
                batch_feat = batch_feat.to(device)
                batch_label = batch_label.to(device)
                outputs = model(batch_feat)

                # 1. 计算测试MSE损失（纯预测损失）
                test_mse_loss = criterion(outputs, batch_label)

                # 2. 提取归一化轴向力、塑性应变（和训练阶段一致）
                normalized_force = batch_feat[..., 0:1]
                normalized_plastic_strain = batch_feat[..., 1:2]

                # 3. 计算测试PINN物理损失
                test_physical_loss = pinn_physical_loss(normalized_force, normalized_plastic_strain, config)

                # 4. 测试总损失 = 测试MSE + 权重×测试PINN损失（权重和训练一致）
                test_total_loss = test_mse_loss + config['pinn_loss_weight'] * test_physical_loss

                # 5. 累加测试总损失（替代原有仅累加MSE）
                test_loss += test_total_loss.item() * batch_feat.size(0)
        # ==========================================================================

        # 测试损失归一化（逻辑不变，只是test_loss现在包含PINN）
        test_loss /= len(test_loader.dataset)
        train_history['test_loss'].append(test_loss)

        # 学习率调度（基于包含PINN的测试损失，也可仍用纯MSE，按需调整）
        scheduler.step(test_loss)

        # 打印训练信息（不变）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        # 早停机制（基于包含PINN的测试损失，也可改回纯MSE，按需调整）
        if early_stopping(test_loss, model):
            print(f'Early stopping at epoch {epoch + 1}, best test loss: {early_stopping.best_loss:.6f}')
            model.load_state_dict(early_stopping.best_model_state)
            break

    # 加载最优模型（不变）
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return model, train_history


# ====================== 7. 贝叶斯预测函数 ======================
def bayesian_predict(model, test_loader, config):
    model.eval()
    device = config['device']
    y_true = []
    y_pred_samples = []

    # 强制开启所有Dropout层（MCD核心，增强随机性）
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            m.p = config['dropout_rate']  # 强制设置Dropout概率（避免eval模式下失效）

    with torch.no_grad():
        for batch_feat, batch_label in test_loader:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            # MCD多次采样（建议mc_samples≥100，增强分布多样性）
            batch_samples = []
            for _ in range(config['mc_samples']):  # 至少100次采样
                output = model(batch_feat)
                batch_samples.append(output.cpu().numpy())

            batch_samples = np.array(batch_samples)  # (mc_samples, batch_size, 1)
            y_pred_samples.append(batch_samples)
            y_true.append(batch_label.cpu().numpy())

    # 处理结果
    y_true = np.concatenate(y_true, axis=0).squeeze()  # (n_test,)
    y_pred_samples = np.concatenate(y_pred_samples, axis=1)  # (mc_samples, n_test, 1)
    y_pred_samples = y_pred_samples.squeeze(-1)  # (mc_samples, n_test)

    # 计算均值和标准差（添加微小噪声，避免方差为0）
    y_pred_mean = np.mean(y_pred_samples, axis=0)  # (n_test,)
    y_pred_std = np.std(y_pred_samples, axis=0) + 1e-6  # 加微小噪声，避免方差为0

    # 转换为实际寿命（反log10，优化标准差计算）
    y_true_actual = 10 ** y_true
    y_pred_mean_actual = 10 ** y_pred_mean
    # 更准确的实际寿命标准差计算（避免近似误差）
    y_pred_actual_samples = 10 ** y_pred_samples  # 先转换所有采样值，再算标准差
    y_pred_std_actual = np.std(y_pred_actual_samples, axis=0) + 1e-6

    return y_true, y_pred_mean, y_pred_std, y_pred_samples, y_true_actual, y_pred_mean_actual, y_pred_std_actual


def calculate_metrics(y_true, y_pred_mean, y_true_actual, y_pred_mean_actual):
    # Log10尺度指标
    mae_log = mean_absolute_error(y_true, y_pred_mean)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2_log = r2_score(y_true, y_pred_mean)

    # 实际寿命尺度指标
    mae_actual = mean_absolute_error(y_true_actual, y_pred_mean_actual)
    rmse_actual = np.sqrt(mean_squared_error(y_true_actual, y_pred_mean_actual))
    r2_actual = r2_score(y_true_actual, y_pred_mean_actual)

    # 计算相对误差（工程常用）
    relative_error = np.mean(np.abs((y_pred_mean_actual - y_true_actual) / (y_true_actual + 1e-8))) * 100  # 百分比

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


def print_metrics(metrics):
    print(f"Log10尺度 - MAE: {metrics['mae_log']:.4f}, RMSE: {metrics['rmse_log']:.4f}, R²: {metrics['r2_log']:.4f}")
    print(
        f"实际寿命尺度 - MAE: {metrics['mae_actual']:.2f}, RMSE: {metrics['rmse_actual']:.2f}, R²: {metrics['r2_actual']:.4f}")
    print(f"实际寿命相对误差: {metrics['relative_error_pct']:.2f}%")


# ====================== save_metrics函数，支持字典和DataFrame ======================
def save_metrics(metrics, config, filename):
    if isinstance(metrics, dict):
        # 处理字典类型（基础指标、不确定性指标）
        df = pd.DataFrame([metrics])
    elif isinstance(metrics, pd.DataFrame):
        # 处理DataFrame类型（鲁棒性指标）
        df = metrics
    else:
        raise TypeError("metrics must be a dict or pandas DataFrame")

    df.to_csv(os.path.join(config['save_metrics_dir'], f'{filename}.csv'), index=False)
    print(f"\n指标已保存到 {config['save_metrics_dir']}/{filename}.csv")


def plot_base_results(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual,
                      y_pred_samples, train_history, config):
    # 1. 测试集预测准确度图
    plt.figure(figsize=(10, 8))
    plt.errorbar(
        y_true, y_pred_mean, yerr=y_pred_std, fmt='o', ecolor='lightgray',
        capsize=3, alpha=0.7, label='Predictions'
    )
    min_val = min(min(y_true), min(y_pred_mean))
    max_val = max(max(y_true), max(y_pred_mean))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    # 添加指标文本
    mae_log = mean_absolute_error(y_true, y_pred_mean)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2_log = r2_score(y_true, y_pred_mean)
    plt.text(
        0.05, 0.95,
        f'MAE = {mae_log:.4f}\nRMSE = {rmse_log:.4f}\nR² = {r2_log:.4f}',
        transform=plt.gca().transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel('True Log10(Remaining Life)')
    plt.ylabel('Predicted Log10(Remaining Life)')
    plt.title('Transformer + Bayesian (Gated Attention) Prediction Results (Log10 Scale)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'test_accuracy_log10.png'), dpi=300)
    plt.show()

    # 2. 实际寿命尺度预测准确度图
    plt.figure(figsize=(10, 8))
    plt.errorbar(
        y_true_actual, y_pred_mean_actual, yerr=y_pred_std_actual, fmt='o', ecolor='lightgray',
        capsize=3, alpha=0.7, label='Predictions'
    )
    min_val = min(min(y_true_actual), min(y_pred_mean_actual))
    max_val = max(max(y_true_actual), max(y_pred_mean_actual))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    # 添加指标文本
    mae_actual = mean_absolute_error(y_true_actual, y_pred_mean_actual)
    rmse_actual = np.sqrt(mean_squared_error(y_true_actual, y_pred_mean_actual))
    r2_actual = r2_score(y_true_actual, y_pred_mean_actual)
    plt.text(
        0.05, 0.95,
        f'MAE = {mae_actual:.2f}\nRMSE = {rmse_actual:.2f}\nR² = {r2_actual:.4f}',
        transform=plt.gca().transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.xlabel('True Remaining Life (Cycles)')
    plt.ylabel('Predicted Remaining Life (Cycles)')
    plt.title('Transformer + Bayesian (Gated Attention) Prediction Results (Actual Scale)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'test_accuracy_actual.png'), dpi=300)
    plt.show()

    # 3. 寿命预测分布直方图
    plt.figure(figsize=(12, 6))
    sample_indices = [0, len(y_true) // 2, -1]
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 3, i + 1)
        sns.histplot(y_pred_samples[:, idx], kde=True, bins=15, label='Prediction Distribution', color='skyblue')
        plt.axvline(y_true[idx], color='r', linestyle='--', label=f'True: {y_true[idx]:.4f}')
        plt.axvline(y_pred_mean[idx], color='g', linestyle='-', label=f'Mean: {y_pred_mean[idx]:.4f}')
        plt.xlabel('Log10(Remaining Life)')
        plt.ylabel('Probability Density')
        plt.title(f'Sample {idx + 1}')
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'life_distributions.png'), dpi=300)
    plt.show()

    # 4. 训练/测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_history['train_loss'], label='Train Loss', color='blue')
    plt.plot(train_history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'loss_curve.png'), dpi=300)
    plt.show()

    # 5. 实际寿命预测误差分布
    error = y_pred_mean_actual - y_true_actual
    plt.figure(figsize=(10, 6))
    sns.histplot(error, kde=True, bins=20, color='orange', edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Prediction Error (Cycles)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Actual Life Scale)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'error_distribution.png'), dpi=300)
    plt.show()


def analyze_uncertainty(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual, config):
    print("\n===== 不确定性可靠性分析 =====")
    # 计算实际误差（Log10尺度和实际尺度）
    error_log = np.abs(y_pred_mean - y_true)  # Log10尺度绝对误差
    error_actual = np.abs(y_pred_mean_actual - y_true_actual)  # 实际尺度绝对误差

    # 修复1：清理异常值（nan/inf）
    def clean_nan_inf(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    y_pred_std = clean_nan_inf(y_pred_std)
    error_log = clean_nan_inf(error_log)
    y_pred_std_actual = clean_nan_inf(y_pred_std_actual)
    error_actual = clean_nan_inf(error_actual)

    # 修复2：检查方差是否为0，避免pearsonr返回nan
    def safe_pearsonr(x, y):
        if np.var(x) < 1e-8 or np.var(y) < 1e-8:
            print(f"警告：输入数据方差为0，无法计算相关系数")
            return np.nan, np.nan
        # 过滤掉x/y中值相同的情况（仅保留有波动的样本）
        mask = (x != x[0]) | (y != y[0])
        if np.sum(mask) < 2:  # 至少需要2个不同样本才能计算相关系数
            print(f"警告：有效样本数不足，无法计算相关系数")
            return np.nan, np.nan
        return pearsonr(x[mask], y[mask])

    # 计算皮尔逊相关系数
    corr_log, p_value_log = safe_pearsonr(y_pred_std, error_log)
    corr_actual, p_value_actual = safe_pearsonr(y_pred_std_actual, error_actual)

    # 打印结果（兼容nan）
    print(f"Log10尺度：标准差与实际误差的相关系数 = {corr_log:.4f}" if not np.isnan(corr_log) else "Log10尺度：相关系数计算失败（方差为0/样本不足）")
    print(f"实际寿命尺度：标准差与实际误差的相关系数 = {corr_actual:.4f}" if not np.isnan(corr_actual) else "实际寿命尺度：相关系数计算失败（方差为0/样本不足）")
    print("说明：相关系数越接近1，说明模型的不确定性评估越可靠（标准差大→误差大）")

    # 可视化（兼容nan，避免绘图报错）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Log10尺度
    ax1.scatter(y_pred_std, error_log, alpha=0.7, color='blue')
    ax1.set_xlabel('Predicted Std (Log10 Scale)')
    ax1.set_ylabel('Absolute Error (Log10 Scale)')
    ax1_title = f'Log10 Scale (Corr = {corr_log:.4f})' if not np.isnan(corr_log) else 'Log10 Scale (Corr = N/A)'
    ax1.set_title(ax1_title)
    ax1.grid(alpha=0.3)
    # 仅当相关系数有效时添加拟合线
    if not np.isnan(corr_log):
        z = np.polyfit(y_pred_std, error_log, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred_std, p(y_pred_std), "r--")

    # 实际寿命尺度
    ax2.scatter(y_pred_std_actual, error_actual, alpha=0.7, color='orange')
    ax2.set_xlabel('Predicted Std (Actual Scale, Cycles)')
    ax2.set_ylabel('Absolute Error (Actual Scale, Cycles)')
    ax2_title = f'Actual Scale (Corr = {corr_actual:.4f})' if not np.isnan(corr_actual) else 'Actual Scale (Corr = N/A)'
    ax2.set_title(ax2_title)
    ax2.grid(alpha=0.3)
    # 仅当相关系数有效时添加拟合线
    if not np.isnan(corr_actual):
        z = np.polyfit(y_pred_std_actual, error_actual, 1)
        p = np.poly1d(z)
        ax2.plot(y_pred_std_actual, p(y_pred_std_actual), "r--")

    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'uncertainty_reliability.png'), dpi=300)
    plt.show()

    # 返回不确定性指标（兼容nan）
    uncertainty_metrics = {
        'corr_log': corr_log if not np.isnan(corr_log) else -999,  # 用-999标记无效值
        'p_value_log': p_value_log if not np.isnan(p_value_log) else -999,
        'corr_actual': corr_actual if not np.isnan(corr_actual) else -999,
        'p_value_actual': p_value_actual if not np.isnan(p_value_actual) else -999
    }

    return uncertainty_metrics


def visualize_attention_weights(model, test_loader, config):
    print("\n===== 注意力权重可视化（无门控） =====")
    model.eval()
    device = config['device']

    # 取第一个批次的第一个样本
    for batch_feat, _ in test_loader:
        batch_feat = batch_feat.to(device)
        with torch.no_grad():
            # 仅返回输出和注意力权重，gate_strength始终为None
            _, attn_weights, _ = model(batch_feat[:1], return_attention=True)

        # 若没有注意力权重（原生TransformerEncoder不返回权重），提示并退出
        if attn_weights is None:
            print("提示：原生TransformerEncoder未返回注意力权重，跳过可视化")
            break

        # 压缩注意力权重维度（去掉batch维度）
        attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (1000, 1000)

        # 1. 1000步注意力热力图（仅保留纯注意力权重可视化）
        fig = plt.figure(figsize=(30, 28))
        ax_main = fig.add_axes([0.08, 0.08, 0.80, 0.88])
        ax_cbar = fig.add_axes([0.90, 0.08, 0.02, 0.88])

        im = ax_main.imshow(
            attn_weights,
            cmap='coolwarm',
            vmin=np.percentile(attn_weights, 5),
            vmax=np.percentile(attn_weights, 95)
        )
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label('Attention Weight', fontsize=30, labelpad=30)
        cbar.ax.tick_params(labelsize=30)

        ax_main.set_title('Attention Weights (No Gated)', fontsize=30, pad=40)
        ax_main.set_xlabel('Key Time Step', fontsize=30, labelpad=30)
        ax_main.set_ylabel('Query Time Step', fontsize=30, labelpad=30)
        ax_main.set_xticks(np.arange(0, 1001, 100))
        ax_main.set_yticks(np.arange(0, 1001, 100))
        ax_main.set_xticklabels(np.arange(0, 1001, 100), fontsize=30)
        ax_main.set_yticklabels(np.arange(0, 1001, 100), fontsize=30)
        ax_main.tick_params(axis='x', rotation=45, pad=15)
        ax_main.tick_params(axis='y', pad=15)

        plt.savefig(
            os.path.join(config['save_fig_dir'], 'full_1000_attention_heatmap.png'),
            dpi=200,
            bbox_inches=None,
            facecolor='white'
        )
        plt.show()

        # 【完全删除】所有门控强度相关的代码（曲线绘制、统计信息等）
        break



# ====================== 分组注意力分析 ======================
def group_attention_analysis(model, test_loader, config, group_size=100):
    print("\n===== 分组注意力分析 =====")
    model.eval()
    device = config['device']

    # 取第一个批次的第一个样本
    for batch_feat, _ in test_loader:
        batch_feat = batch_feat.to(device)
        with torch.no_grad():
            _, attn_weights, _ = model(batch_feat[:1], return_attention=True)  # (1, seq_len, seq_len)
        attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (1000, 1000)
        seq_len = attn_weights.shape[0]
        num_groups = seq_len // group_size

        # 1. 计算组内平均注意力权重
        group_attn_mean = []
        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            group_attn = attn_weights[start:end, start:end]  # 组内自注意力
            group_attn_mean.append(np.mean(group_attn))

        # 2. 可视化组间注意力权重差异
        plt.figure(figsize=(12, 6))
        x_ticks = [f'Group {i + 1}\n({i * group_size}-{(i + 1) * group_size - 1})' for i in range(num_groups)]
        # 修正：移除marker='o'，新增edgecolor美化
        plt.bar(x_ticks, group_attn_mean, color='teal', edgecolor='black', alpha=0.8)
        plt.xlabel('Feature Groups (Time Steps)')
        plt.ylabel('Average Attention Weight')
        plt.title('Group-wise Attention权重平均值对比')
        plt.grid(alpha=0.3, axis='y')  # 仅显示y轴网格，更清晰
        plt.tight_layout()
        plt.savefig(os.path.join(config['save_fig_dir'], 'group_attention_comparison.png'), dpi=300)
        plt.show()

        # 3. 绘制组间注意力热力图
        if num_groups >= 3:
            plt.figure(figsize=(15, 5))
            for i in range(3):
                start = i * group_size
                end = start + group_size
                plt.subplot(1, 3, i + 1)
                sns.heatmap(attn_weights[start:end, start:end], cmap='coolwarm', cbar=False)
                plt.title(f'Group {i + 1} Attention Heatmap')
                plt.xlabel('Within Group Time Step')
                plt.ylabel('Within Group Time Step')
            plt.tight_layout()
            plt.savefig(os.path.join(config['save_fig_dir'], 'group_attention_heatmaps.png'), dpi=300)
            plt.show()
        break


# ====================== 特征贡献度计算 ======================
def calculate_feature_contribution(model, test_loader, config):
    print("\n===== 特征贡献度分析 =====")
    model.eval()
    device = config['device']

    # 1. 基于注意力权重的贡献度（平均注意力权重）
    attn_contribution = []
    # 2. 基于梯度的贡献度（输入特征的梯度绝对值）
    grad_contribution = []

    for batch_feat, batch_label in test_loader:
        # 启用输入特征的梯度计算（必须）
        batch_feat = batch_feat.to(device).requires_grad_(True)
        batch_label = batch_label.to(device)

        # 移除整体torch.no_grad()，仅对注意力权重的提取使用no_grad（避免影响梯度）
        # 前向传播获取输出和注意力权重（保留计算图用于梯度计算）
        outputs, attn_weights, _ = model(batch_feat, return_attention=True)

        # 处理注意力权重（无需梯度，单独包裹no_grad）
        with torch.no_grad():
            # 计算注意力贡献度（平均查询-键注意力）
            attn_mean = attn_weights.mean(dim=1).mean(dim=0)  # (seq_len,)
            attn_contribution.append(attn_mean.cpu().numpy())

        # 计算梯度贡献度（输出对输入的梯度）
        model.zero_grad()  # 清空之前的梯度
        # 计算损失（用MSE损失代替mean，更贴合任务，且梯度更合理）
        loss = torch.nn.functional.mse_loss(outputs, batch_label)
        loss.backward(retain_graph=False)  # 无需保留图，计算后释放

        # 提取输入特征的梯度
        if batch_feat.grad is not None:
            grad = batch_feat.grad  # (batch, seq_len, 1)
            grad_mean = grad.abs().mean(dim=0).squeeze().cpu().numpy()  # (seq_len,)
            grad_contribution.append(grad_mean)
        else:
            print("警告：输入特征梯度为None，可能是模型结构问题")
            grad_mean = np.zeros(batch_feat.shape[1])  # 兜底
            grad_contribution.append(grad_mean)

        break  # 仅用第一个批次计算

    # 聚合贡献度
    attn_contribution = np.mean(attn_contribution, axis=0)  # (1000,)
    grad_contribution = np.mean(grad_contribution, axis=0)  # (1000,)

    # 归一化（避免除零错误）
    attn_max = attn_contribution.max()
    attn_min = attn_contribution.min()
    if attn_max - attn_min < 1e-8:
        attn_contribution = np.zeros_like(attn_contribution)
    else:
        attn_contribution = (attn_contribution - attn_min) / (attn_max - attn_min)

    grad_max = grad_contribution.max()
    grad_min = grad_contribution.min()
    if grad_max - grad_min < 1e-8:
        grad_contribution = np.zeros_like(grad_contribution)
    else:
        grad_contribution = (grad_contribution - grad_min) / (grad_max - grad_min)

    # 可视化两种贡献度对比
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(attn_contribution, color='blue', alpha=0.7, marker='.', markersize=2)
    plt.title('特征贡献度（基于注意力权重）')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Contribution')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(grad_contribution, color='red', alpha=0.7, marker='.', markersize=2)
    plt.title('特征贡献度（基于梯度）')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Contribution')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config['save_fig_dir'], 'feature_contribution_comparison.png'), dpi=300)
    plt.show()
    # 输出Top10重要时间步
    top_attn_idx = np.argsort(attn_contribution)[-10:][::-1]
    top_grad_idx = np.argsort(grad_contribution)[-10:][::-1]
    print(f"注意力权重Top10时间步：{top_attn_idx}")
    print(f"梯度Top10时间步：{top_grad_idx}")
    return attn_contribution, grad_contribution


# 在stage3_analysis函数中添加调用
def stage3_analysis(model, test_loader, y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual, y_pred_samples, train_history, config):
    # 1. 计算多尺度评估指标
    metrics = calculate_metrics(y_true, y_pred_mean, y_true_actual, y_pred_mean_actual)
    # 保存指标到CSV
    save_metrics(metrics, config, 'base_metrics')
    print("\n===== 基础性能指标 =====")
    print_metrics(metrics)

    # 2. 基础可视化（预测准确度、损失曲线、寿命分布）
    plot_base_results(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual, y_pred_samples, train_history, config)

    # 3. 不确定性可靠性分析（阶段三关键）
    if config['uncertainty_analysis']:
        uncertainty_metrics = analyze_uncertainty(y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual, y_pred_std_actual, config)
        save_metrics(uncertainty_metrics, config, 'uncertainty_metrics')

    # 4. 门控注意力权重可视化（阶段三关键）
    if config['visualize_attention'] and config['gate_attention']:
        visualize_attention_weights(model, test_loader, config)
    return metrics
# ====================== 9. 主流程执行（衔接阶段二与阶段三） ======================
if __name__ == '__main__':
    # 加载数据
    full_dataset = FatigueDataset(config['data_dir'])
    n_samples = len(full_dataset)
    sample_ids = np.arange(n_samples)

    # 划分训练/测试集（用户可通过config修改比例）
    train_ids, test_ids = train_test_split(
        sample_ids, train_size=config['train_ratio'], random_state=config['random_seed']
    )
    train_dataset = FatigueDataset(config['data_dir'], sample_ids=train_ids)
    test_dataset = FatigueDataset(config['data_dir'], sample_ids=test_ids)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 初始化带门控注意力的Transformer模型
    model = TransformerRegressor(
        input_dim=2,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout_rate=config['dropout_rate'],
        gate_attention=config['gate_attention']
    ).to(config['device'])
    print(f"\nModel initialized on device: {config['device']}")
    print(f"Using gated attention: {config['gate_attention']}")

    # 训练模型（阶段二）
    print("\n===== 开始训练模型 =====")
    model, train_history = train_model(model, train_loader, test_loader, config)

    # 贝叶斯预测（阶段二）
    print("\n===== 开始贝叶斯预测 =====")
    results = bayesian_predict(model, test_loader, config)
    y_true, y_pred_mean, y_pred_std, y_pred_samples, y_true_actual, y_pred_mean_actual, y_pred_std_actual = results

    # 阶段三：多维度性能验证与可视化
    print("\n===== 开始阶段三：模型性能多维度验证 =====")
    metrics = stage3_analysis(model, test_loader, y_true, y_pred_mean, y_pred_std, y_true_actual, y_pred_mean_actual,
                              y_pred_std_actual, y_pred_samples, train_history, config)

    # 保存预测结果
    results_df = pd.DataFrame({
        'true_log_life': y_true,
        'pred_log_life_mean': y_pred_mean,
        'pred_log_life_std': y_pred_std,
        'true_actual_life': y_true_actual,
        'pred_actual_life_mean': y_pred_mean_actual,
        'pred_actual_life_std': y_pred_std_actual
    })
    results_df.to_csv(os.path.join(config['save_fig_dir'], 'prediction_results.csv'), index=False)
    print(f"\n预测结果已保存到 {config['save_fig_dir']}/prediction_results.csv")
    print("\n===== 阶段三分析完成 =====")