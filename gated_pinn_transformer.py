import os
import re
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
config = {
    # -------------------- data --------------------
    "data_dir": r"preprocessed_data_30%life_fft_downsample",
    "train_ratio": 0.7,
    "batch_size": 8,
    "target_length": 1000,
    "cycle_ratio": 0.5,
    "random_seed": 100,
    # -------------------- model --------------------
    "d_model": 64,
    "nhead": 2,
    "num_encoder_layers": 2,
    "dim_feedforward": 64,
    "dropout_rate": 0.3,
    "gate_attention": True,
    # -------------------- train --------------------
    "epochs": 10000,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 10000,
    "factor": 0.7,
    "mc_samples": 50,
    # -------------------- device --------------------
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # -------------------- output --------------------
    "save_fig_dir": "transformer_bayes_gated_cdm_results",
    "save_metrics_dir": "cdm_metrics",
    "visualize_attention": True,
    "uncertainty_analysis": True,
    # -------------------- physics --------------------
    "cross_section_area": ,      # mm^2
    "elastic_modulus": ,         # MPa
    "yield_strength": ,          # MPa
    # CDM parameters
    "physics_loss_weight": ,
    "cdm_alpha": ,
    "cdm_beta": ,
    "w_ref_mpa": ,
    "strain_nonneg_weight": ,
    "yield_penalty_weight": ,
}
BUTT_PATHS = {
  
}

CROSS_PATHS = {

}

RAW_PATHS = {}
RAW_PATHS.update(BUTT_PATHS)
RAW_PATHS.update(CROSS_PATHS)
os.makedirs(config["save_fig_dir"], exist_ok=True)
os.makedirs(config["save_metrics_dir"], exist_ok=True)
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

def base_sample_name(sample_name):
    return re.sub(r"_(original|aug\d+)$", "", sample_name)

def get_variant_name(sample_name):
    m = re.search(r"_(original|aug\d+)$", sample_name)
    return m.group(1) if m else "original"

def get_closest_recorded_cycle(theoretical_cycle, recorded_cycles):
    unique_cycles = np.unique(recorded_cycles)
    unique_cycles.sort()

    if theoretical_cycle <= unique_cycles[0]:
        return unique_cycles[0]

    if theoretical_cycle >= unique_cycles[-1]:
        return unique_cycles[-1]

    for i in range(len(unique_cycles) - 1):
        if unique_cycles[i] <= theoretical_cycle <= unique_cycles[i + 1]:
            diff_left = theoretical_cycle - unique_cycles[i]
            diff_right = unique_cycles[i + 1] - theoretical_cycle

            right_is_round = (
                unique_cycles[i + 1] % 1000 == 0
                or unique_cycles[i + 1] % 100 == 0
                or unique_cycles[i + 1] % 10 == 0
            )
            left_is_round = (
                unique_cycles[i] % 1000 == 0
                or unique_cycles[i] % 100 == 0
                or unique_cycles[i] % 10 == 0
            )

            if right_is_round and not left_is_round:
                return unique_cycles[i + 1]
            if left_is_round and not right_is_round:
                return unique_cycles[i]

            return unique_cycles[i] if diff_left <= diff_right else unique_cycles[i + 1]

    return unique_cycles[np.argmin(np.abs(unique_cycles - theoretical_cycle))]

def extract_cycles_data(cycles, axial_force, plastic_strain, target_cycles):
    indices = np.where(cycles <= target_cycles)[0]

    if len(indices) == 0:
        return axial_force[:1000], plastic_strain[:1000], target_cycles

    axial_extracted = axial_force[indices]
    strain_extracted = plastic_strain[indices]
    actual_cycles_used = np.max(cycles[indices])

    return axial_extracted, strain_extracted, actual_cycles_used

def downsample_sequence(sequence, target_length=1000):
    n_original = len(sequence)

    if n_original <= target_length:
        return sequence.astype(np.float32)

    window = np.hamming(n_original)
    sequence_windowed = sequence * window
    fft_spectrum = np.fft.fft(sequence_windowed)

    n_keep = target_length // 2
    fft_downsampled = np.zeros(target_length, dtype=complex)
    fft_downsampled[:n_keep] = fft_spectrum[:n_keep]
    fft_downsampled[-n_keep:] = fft_spectrum[-n_keep:]

    sequence_downsampled = np.fft.ifft(fft_downsampled).real
    scale_factor = np.max(np.abs(sequence)) / (np.max(np.abs(sequence_downsampled)) + 1e-8)
    sequence_downsampled = sequence_downsampled * scale_factor

    return sequence_downsampled.astype(np.float32)

def load_raw_reference_map(raw_paths, cycle_ratio=0.5, target_length=1000):
    raw_map = {}
    print("正在加载原始未归一化数据，用于CDM物理约束...")

    for exp_name, path in raw_paths.items():
        if not os.path.exists(path):
            print(f"  [警告] 原始文件不存在，跳过: {exp_name} -> {path}")
            continue

        try:
            df = pd.read_csv(path, skiprows=1, header=None)
            cycles = df.iloc[:, 1].values.astype(np.float32)
            axial_force = df.iloc[:, 3].values.astype(np.float32)
            plastic_strain = df.iloc[:, 4].values.astype(np.float32)

            total_life = np.max(cycles)
            theoretical_target = total_life * cycle_ratio
            actual_target_cycles = get_closest_recorded_cycle(theoretical_target, cycles)

            force_seg, strain_seg, actual_cycles_used = extract_cycles_data(
                cycles,
                axial_force,
                plastic_strain,
                actual_target_cycles,
            )

            force_down = downsample_sequence(force_seg, target_length)
            strain_down = downsample_sequence(strain_seg, target_length)

            force_mean = float(np.mean(force_down))
            force_std = float(np.std(force_down) + 1e-8)
            strain_mean = float(np.mean(strain_down))
            strain_std = float(np.std(strain_down) + 1e-8)

            raw_map[exp_name] = {
                "force_down": force_down,
                "strain_down": strain_down,
                "force_mean": force_mean,
                "force_std": force_std,
                "strain_mean": strain_mean,
                "strain_std": strain_std,
                "actual_cycles_used": float(actual_cycles_used),
                "total_life": float(total_life),
                "remaining_life": float(total_life - actual_cycles_used),
            }

            print(f"  已加载 {exp_name}")

        except Exception as e:
            print(f"  [失败] {exp_name}: {e}")

    print(f"原始参考样本数: {len(raw_map)}")
    return raw_map

class FatigueDataset(Dataset):
    def __init__(self, data_dir, raw_reference_map, sample_ids=None):
        self.data_dir = data_dir
        self.features_dir = os.path.join(data_dir, "features")
        self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

        if sample_ids is not None:
            self.metadata = self.metadata[self.metadata["sample_id"].isin(sample_ids)].reset_index(drop=True)

        self.features = []
        self.labels = []
        self.raw_forces = []
        self.raw_plastic_strains = []
        self.sample_names = []

        missing_raw = 0

        for _, row in self.metadata.iterrows():
            sample_name = row["sample_name"]
            base_name = base_sample_name(sample_name)

            feat_path = os.path.join(self.features_dir, row["feature_file"])
            feat = pd.read_csv(
                feat_path,
                usecols=["normalized_force", "normalized_plastic_strain"],
            ).values.astype(np.float32)

            if base_name not in raw_reference_map:
                missing_raw += 1
                continue

            ref = raw_reference_map[base_name]

            raw_force = feat[:, 0] * ref["force_std"] + ref["force_mean"]
            raw_plastic = feat[:, 1] * ref["strain_std"] + ref["strain_mean"]

            self.features.append(feat)
            self.labels.append(np.float32(row["log_remaining_life"]))
            self.raw_forces.append(raw_force.astype(np.float32))
            self.raw_plastic_strains.append(raw_plastic.astype(np.float32))
            self.sample_names.append(sample_name)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.raw_forces = np.array(self.raw_forces, dtype=np.float32)
        self.raw_plastic_strains = np.array(self.raw_plastic_strains, dtype=np.float32)

        if missing_raw > 0:
            print(f"[警告] 有 {missing_raw} 个样本因找不到原始参考数据而被跳过")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx]).unsqueeze(0),
            torch.from_numpy(self.raw_forces[idx]).unsqueeze(-1),
            torch.from_numpy(self.raw_plastic_strains[idx]).unsqueeze(-1),
            self.sample_names[idx],
        )

class GatedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout_rate):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads=nhead,
            batch_first=False,
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, attn_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
        )

        attn_output = self.dropout(attn_output)

        gate_input = torch.cat([x, attn_output], dim=-1)
        gate = self.gate(gate_input)
        gate = torch.clamp(gate, 0.1, 0.9)

        output = x * (1 - gate) + attn_output * gate
        output = self.norm(output)

        return output, attn_weights, gate

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout_rate=0.3,
        gate_attention=True,
    ):
        super().__init__()

        self.gate_attention = gate_attention

        self.embedding = nn.Linear(input_dim, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.pos_encoder = PositionalEncoding(
            d_model,
            max_len=config["target_length"],
        )

        if gate_attention:
            self.gated_attn = GatedAttention(
                d_model,
                nhead,
                dropout_rate,
            )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=False,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool_dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)

        attn_weights = None
        gate_values = None

        if self.gate_attention:
            x, attn_weights, gate_values = self.gated_attn(x)

        x = self.transformer_encoder(x)

        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(-1)
        x = self.pool_dropout(x)

        out = self.fc(x)

        if return_attention:
            return out, attn_weights, gate_values

        return out

def cdm_physical_loss(pred_log_life, raw_force_kN, raw_plastic_strain, cfg):
    area = cfg["cross_section_area"]
    sigma_y = cfg["yield_strength"]
    alpha = cfg["cdm_alpha"]
    beta = cfg["cdm_beta"]
    w_ref = cfg["w_ref_mpa"]

    # kN -> N, N/mm^2 = MPa
    stress = raw_force_kN * 1000.0 / area
    abs_stress = torch.abs(stress)

    plastic_pos = torch.relu(raw_plastic_strain)

    wp_density = torch.mean(abs_stress * plastic_pos, dim=1)
    wp_norm = wp_density / (w_ref + 1e-8)

    damage_rate = alpha * torch.pow(wp_norm + 1e-8, beta)
    pred_life = torch.pow(10.0, pred_log_life)

    cdm_residual = pred_life * damage_rate - 1.0
    loss_cdm = torch.mean(cdm_residual ** 2)

    loss_nonneg = torch.mean(torch.relu(-raw_plastic_strain) ** 2)

    elastic_mask = (abs_stress <= sigma_y).float()
    loss_yield = torch.mean((plastic_pos * elastic_mask) ** 2)

    total_loss = (
        loss_cdm
        + cfg["strain_nonneg_weight"] * loss_nonneg
        + cfg["yield_penalty_weight"] * loss_yield
    )

    return total_loss

def create_grouped_train_test_split(metadata, train_ratio=0.7, random_seed=100):
    meta = metadata.copy()
    meta["base_name"] = meta["sample_name"].apply(base_sample_name)

    group_df = (
        meta.groupby("base_name", as_index=False)
        .agg(
            log_remaining_life=("log_remaining_life", "mean"),
            joint_type=("joint_type", "first"),
        )
    )

    n_groups = len(group_df)
    q = min(3, n_groups) if n_groups > 1 else 1

    if q > 1:
        group_df["life_bin"] = pd.qcut(
            group_df["log_remaining_life"],
            q=q,
            duplicates="drop",
        ).astype(str)
    else:
        group_df["life_bin"] = "all"

    group_df["strata"] = (
        group_df["joint_type"].astype(str)
        + "_"
        + group_df["life_bin"].astype(str)
    )

    rng = np.random.RandomState(random_seed)

    train_base_names = []
    test_base_names = []

    for _, sub_df in group_df.groupby("strata"):
        names = sub_df["base_name"].tolist()
        rng.shuffle(names)

        n_train = max(1, int(round(len(names) * train_ratio))) if len(names) > 1 else 1
        n_train = min(n_train, len(names) - 1) if len(names) > 1 else 1

        if len(names) == 1:
            if len(train_base_names) <= len(test_base_names):
                train_base_names.extend(names)
            else:
                test_base_names.extend(names)
        else:
            train_base_names.extend(names[:n_train])
            test_base_names.extend(names[n_train:])

    if len(test_base_names) == 0 and len(train_base_names) > 1:
        test_base_names.append(train_base_names.pop())

    if len(train_base_names) == 0 and len(test_base_names) > 1:
        train_base_names.append(test_base_names.pop())

    train_ids = meta[meta["base_name"].isin(train_base_names)]["sample_id"].tolist()
    test_ids = meta[meta["base_name"].isin(test_base_names)]["sample_id"].tolist()

    return train_ids, test_ids, train_base_names, test_base_names

class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

def train_model(model, train_loader, test_loader, cfg):
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["factor"],
        patience=max(10, cfg["patience"] // 4),
    )

    early_stopping = EarlyStopping(
        patience=cfg["patience"],
        min_delta=1e-4,
    )

    train_history = {
        "train_loss": [],
        "test_loss": [],
        "train_mse": [],
        "test_mse": [],
        "train_phy": [],
        "test_phy": [],
    }

    device = cfg["device"]

    for epoch in range(cfg["epochs"]):
        model.train()

        train_loss = 0.0
        train_mse_sum = 0.0
        train_phy_sum = 0.0

        for batch_feat, batch_label, raw_force, raw_plastic, _ in train_loader:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)
            raw_force = raw_force.to(device)
            raw_plastic = raw_plastic.to(device)

            optimizer.zero_grad()

            outputs = model(batch_feat)

            mse_loss = criterion(outputs, batch_label)
            phy_loss = cdm_physical_loss(outputs, raw_force, raw_plastic, cfg)

            loss = mse_loss + cfg["physics_loss_weight"] * phy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_feat.size(0)
            train_loss += loss.item() * bs
            train_mse_sum += mse_loss.item() * bs
            train_phy_sum += phy_loss.item() * bs

        train_loss /= len(train_loader.dataset)
        train_mse_sum /= len(train_loader.dataset)
        train_phy_sum /= len(train_loader.dataset)

        model.eval()

        test_loss = 0.0
        test_mse_sum = 0.0
        test_phy_sum = 0.0

        with torch.no_grad():
            for batch_feat, batch_label, raw_force, raw_plastic, _ in test_loader:
                batch_feat = batch_feat.to(device)
                batch_label = batch_label.to(device)
                raw_force = raw_force.to(device)
                raw_plastic = raw_plastic.to(device)

                outputs = model(batch_feat)

                mse_loss = criterion(outputs, batch_label)
                phy_loss = cdm_physical_loss(outputs, raw_force, raw_plastic, cfg)
                total_loss = mse_loss + cfg["physics_loss_weight"] * phy_loss

                bs = batch_feat.size(0)
                test_loss += total_loss.item() * bs
                test_mse_sum += mse_loss.item() * bs
                test_phy_sum += phy_loss.item() * bs

        test_loss /= len(test_loader.dataset)
        test_mse_sum /= len(test_loader.dataset)
        test_phy_sum /= len(test_loader.dataset)

        train_history["train_loss"].append(train_loss)
        train_history["test_loss"].append(test_loss)
        train_history["train_mse"].append(train_mse_sum)
        train_history["test_mse"].append(test_mse_sum)
        train_history["train_phy"].append(train_phy_sum)
        train_history["test_phy"].append(test_phy_sum)

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{cfg['epochs']}] "
                f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | "
                f"Train MSE: {train_mse_sum:.6f} | Test MSE: {test_mse_sum:.6f} | "
                f"Train CDM: {train_phy_sum:.6f} | Test CDM: {test_phy_sum:.6f}"
            )

        if early_stopping(test_loss, model):
            print(
                f"Early stopping at epoch {epoch + 1}, "
                f"best test loss: {early_stopping.best_loss:.6f}"
            )
            break

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return model, train_history

def bayesian_predict(model, test_loader, cfg):
    model.eval()

    device = cfg["device"]

    y_true = []
    y_pred_samples = []
    sample_names = []

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            m.p = cfg["dropout_rate"]

    with torch.no_grad():
        for batch_feat, batch_label, _, _, batch_names in test_loader:
            batch_feat = batch_feat.to(device)
            batch_label = batch_label.to(device)

            batch_samples = []

            for _ in range(cfg["mc_samples"]):
                output = model(batch_feat)
                batch_samples.append(output.cpu().numpy())

            batch_samples = np.array(batch_samples)

            y_pred_samples.append(batch_samples)
            y_true.append(batch_label.cpu().numpy())
            sample_names.extend(list(batch_names))

    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred_samples = np.concatenate(y_pred_samples, axis=1).squeeze(-1)

    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0) + 1e-6

    y_true_actual = 10 ** y_true
    y_pred_mean_actual = 10 ** y_pred_mean

    y_pred_actual_samples = 10 ** y_pred_samples
    y_pred_std_actual = np.std(y_pred_actual_samples, axis=0) + 1e-6

    return (
        sample_names,
        y_true,
        y_pred_mean,
        y_pred_std,
        y_pred_samples,
        y_true_actual,
        y_pred_mean_actual,
        y_pred_std_actual,
    )

def calculate_metrics(y_true, y_pred_mean, y_true_actual, y_pred_mean_actual):
    mae_log = mean_absolute_error(y_true, y_pred_mean)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2_log = r2_score(y_true, y_pred_mean)

    mae_actual = mean_absolute_error(y_true_actual, y_pred_mean_actual)
    rmse_actual = np.sqrt(mean_squared_error(y_true_actual, y_pred_mean_actual))
    r2_actual = r2_score(y_true_actual, y_pred_mean_actual)

    relative_error = (
        np.mean(
            np.abs(
                (y_pred_mean_actual - y_true_actual)
                / (y_true_actual + 1e-8)
            )
        )
        * 100.0
    )

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "mae_actual": mae_actual,
        "rmse_actual": rmse_actual,
        "r2_actual": r2_actual,
        "relative_error_pct": relative_error,
    }

def save_metrics(metrics, cfg, filename):
    pd.DataFrame([metrics]).to_csv(
        os.path.join(cfg["save_metrics_dir"], f"{filename}.csv"),
        index=False,
    )

    print(f"指标已保存到 {cfg['save_metrics_dir']}/{filename}.csv")

def print_metrics(metrics):
    print(
        f"Log10尺度 - MAE: {metrics['mae_log']:.4f}, "
        f"RMSE: {metrics['rmse_log']:.4f}, "
        f"R²: {metrics['r2_log']:.4f}"
    )

    print(
        f"实际寿命尺度 - MAE: {metrics['mae_actual']:.2f}, "
        f"RMSE: {metrics['rmse_actual']:.2f}, "
        f"R²: {metrics['r2_actual']:.4f}"
    )

    print(f"实际寿命相对误差: {metrics['relative_error_pct']:.2f}%")

def analyze_uncertainty(
    y_true,
    y_pred_mean,
    y_pred_std,
    y_true_actual,
    y_pred_mean_actual,
    y_pred_std_actual,
    cfg,
):
    """
    不确定性可靠性分析。

    计算 MC Dropout 预测标准差 与 实际预测绝对误差之间的 Pearson 相关系数。

    注意：
    这里计算的 r 不是 true life 和 predicted life 之间的相关系数，
    而是：

        r_log = corr(y_pred_std, |y_pred_mean - y_true|)

        r_actual = corr(y_pred_std_actual, |y_pred_mean_actual - y_true_actual|)

    工程含义：
    如果 r 越接近 1，说明模型给出的预测不确定性越能反映真实预测误差；
    即模型认为“不确定”的样本，实际误差也更大。
    """

    print("\n===== 不确定性可靠性分析 =====")

    # 1. 计算绝对误差
    error_log = np.abs(y_pred_mean - y_true)
    error_actual = np.abs(y_pred_mean_actual - y_true_actual)

    # 2. 清理 nan 和 inf
    def clean_nan_inf(arr):
        arr = np.asarray(arr).astype(np.float64)
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return arr

    y_pred_std = clean_nan_inf(y_pred_std)
    error_log = clean_nan_inf(error_log)
    y_pred_std_actual = clean_nan_inf(y_pred_std_actual)
    error_actual = clean_nan_inf(error_actual)

    # 3. 安全计算 Pearson 相关系数
    def safe_pearsonr(x, y):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) < 2:
            print("警告：有效样本数小于 2，无法计算 Pearson 相关系数")
            return np.nan, np.nan

        if np.var(x) < 1e-12 or np.var(y) < 1e-12:
            print("警告：输入数据方差过小，无法计算 Pearson 相关系数")
            return np.nan, np.nan

        return pearsonr(x, y)

    # 4. 计算相关系数
    corr_log, p_value_log = safe_pearsonr(y_pred_std, error_log)
    corr_actual, p_value_actual = safe_pearsonr(
        y_pred_std_actual,
        error_actual,
    )

    # 5. 打印结果
    if not np.isnan(corr_log):
        print(
            f"Log10尺度：预测标准差与绝对误差的 Pearson r = {corr_log:.4f}, "
            f"p-value = {p_value_log:.4e}"
        )
    else:
        print("Log10尺度：Pearson r 计算失败")

    if not np.isnan(corr_actual):
        print(
            f"实际寿命尺度：预测标准差与绝对误差的 Pearson r = {corr_actual:.4f}, "
            f"p-value = {p_value_actual:.4e}"
        )
    else:
        print("实际寿命尺度：Pearson r 计算失败")

    # 6. 可视化：不确定性 vs 误差
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Log10尺度
    axes[0].scatter(
        y_pred_std,
        error_log,
        alpha=0.75,
        color="blue",
        edgecolor="k",
    )

    axes[0].set_xlabel("Predicted Std (Log10 Scale)")
    axes[0].set_ylabel("Absolute Error (Log10 Scale)")

    if not np.isnan(corr_log):
        axes[0].set_title(f"Log10 Scale: r = {corr_log:.4f}")

        if np.var(y_pred_std) >= 1e-12:
            z = np.polyfit(y_pred_std, error_log, 1)
            p = np.poly1d(z)
            axes[0].plot(
                y_pred_std,
                p(y_pred_std),
                "r--",
                linewidth=2,
            )
    else:
        axes[0].set_title("Log10 Scale: r = N/A")

    axes[0].grid(alpha=0.3)

    # 实际寿命尺度
    axes[1].scatter(
        y_pred_std_actual,
        error_actual,
        alpha=0.75,
        color="orange",
        edgecolor="k",
    )

    axes[1].set_xlabel("Predicted Std (Actual Life Scale, Cycles)")
    axes[1].set_ylabel("Absolute Error (Actual Life Scale, Cycles)")

    if not np.isnan(corr_actual):
        axes[1].set_title(f"Actual Scale: r = {corr_actual:.4f}")

        if np.var(y_pred_std_actual) >= 1e-12:
            z = np.polyfit(y_pred_std_actual, error_actual, 1)
            p = np.poly1d(z)
            axes[1].plot(
                y_pred_std_actual,
                p(y_pred_std_actual),
                "r--",
                linewidth=2,
            )
    else:
        axes[1].set_title("Actual Scale: r = N/A")

    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        cfg["save_fig_dir"],
        "uncertainty_reliability.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        facecolor="white",
    )

    plt.show()

    print(f"不确定性可靠性图已保存至: {save_path}")

    # 7. 返回并保存指标
    uncertainty_metrics = {
        "corr_log": corr_log if not np.isnan(corr_log) else -999,
        "p_value_log": p_value_log if not np.isnan(p_value_log) else -999,
        "corr_actual": corr_actual if not np.isnan(corr_actual) else -999,
        "p_value_actual": p_value_actual if not np.isnan(p_value_actual) else -999,
    }

    return uncertainty_metrics

def visualize_attention_heatmap(model, test_loader, cfg):
    print("\n===== 注意力权重热力图 =====")

    model.eval()
    device = cfg["device"]

    for batch_feat, _, _, _, batch_names in test_loader:
        batch_feat = batch_feat.to(device)
        sample_name = batch_names[0]

        with torch.no_grad():
            _, attn_weights, _ = model(
                batch_feat[:1],
                return_attention=True,
            )

        if attn_weights is None:
            print("当前模型未返回注意力权重，跳过热力图。")
            return

        attn_weights = attn_weights.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(24, 24))

        ax_main = fig.add_axes([0.08, 0.08, 0.80, 0.88])
        ax_cbar = fig.add_axes([0.90, 0.08, 0.02, 0.88])

        im = ax_main.imshow(
            attn_weights,
            cmap="coolwarm",
            vmin=np.percentile(attn_weights, 5),
            vmax=np.percentile(attn_weights, 95),
        )

        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=18)

        ax_main.set_title(
            f"Attention Heatmap - {sample_name}",
            fontsize=20,
            pad=20,
        )

        ax_main.set_xlabel(
            "Key Time Step",
            fontsize=18,
        )

        ax_main.set_ylabel(
            "Query Time Step",
            fontsize=18,
        )

        tick_step = 100
        ticks = np.arange(0, cfg["target_length"] + 1, tick_step)

        ax_main.set_xticks(ticks)
        ax_main.set_yticks(ticks)
        ax_main.set_xticklabels(
            ticks,
            fontsize=12,
            rotation=45,
        )
        ax_main.set_yticklabels(
            ticks,
            fontsize=12,
        )

        save_path = os.path.join(
            cfg["save_fig_dir"],
            "attention_heatmap.png",
        )

        plt.savefig(
            save_path,
            dpi=200,
            facecolor="white",
        )

        plt.show()

        print(f"注意力热力图已保存至: {save_path}")

        break

if __name__ == "__main__":
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    # 1. 加载归一化后元数据
    metadata_path = os.path.join(
        config["data_dir"],
        "metadata.csv",
    )

    metadata = pd.read_csv(metadata_path)

    # 2. 加载原始未归一化参考数据，用于CDM物理损失
    raw_reference_map = load_raw_reference_map(
        RAW_PATHS,
        cycle_ratio=config["cycle_ratio"],
        target_length=config["target_length"],
    )

    # 3. 分组划分训练/测试集，确保同一原始试件及增强样本不泄漏
    train_ids, test_ids, train_base_names, test_base_names = create_grouped_train_test_split(
        metadata,
        train_ratio=config["train_ratio"],
        random_seed=config["random_seed"],
    )

    print(f"\n原始试件分组数: {len(set(metadata['sample_name'].apply(base_sample_name)))}")
    print(f"训练集原始试件数: {len(train_base_names)}")
    print(f"测试集原始试件数: {len(test_base_names)}")
    print(f"训练样本数(含增强): {len(train_ids)}")
    print(f"测试样本数(含增强): {len(test_ids)}")

    # 4. 构建数据集
    train_dataset = FatigueDataset(
        config["data_dir"],
        raw_reference_map,
        sample_ids=train_ids,
    )

    test_dataset = FatigueDataset(
        config["data_dir"],
        raw_reference_map,
        sample_ids=test_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # 5. 初始化模型
    model = TransformerRegressor(
        input_dim=2,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
        gate_attention=config["gate_attention"],
    ).to(config["device"])

    print(f"\n模型初始化完成，运行设备: {config['device']}")
    print(f"是否启用门控注意力: {config['gate_attention']}")
    print(f"CDM物理损失权重: {config['physics_loss_weight']}")

    # 6. 训练
    print("\n===== 开始训练模型（分组划分 + CDM物理约束） =====")

    model, train_history = train_model(
        model,
        train_loader,
        test_loader,
        config,
    )

    # 7. 贝叶斯预测
    print("\n===== 开始贝叶斯预测（MCDropout） =====")

    (
        sample_names,
        y_true,
        y_pred_mean,
        y_pred_std,
        y_pred_samples,
        y_true_actual,
        y_pred_mean_actual,
        y_pred_std_actual,
    ) = bayesian_predict(
        model,
        test_loader,
        config,
    )

    # 8. 计算基础性能指标
    print("\n===== 模型性能指标 =====")

    metrics = calculate_metrics(
        y_true,
        y_pred_mean,
        y_true_actual,
        y_pred_mean_actual,
    )

    print_metrics(metrics)

    save_metrics(
        metrics,
        config,
        "base_metrics",
    )

    # 8.1 计算不确定性标准差与实际误差之间的 Pearson 相关系数
    if config.get("uncertainty_analysis", True):
        uncertainty_metrics = analyze_uncertainty(
            y_true,
            y_pred_mean,
            y_pred_std,
            y_true_actual,
            y_pred_mean_actual,
            y_pred_std_actual,
            config,
        )

        save_metrics(
            uncertainty_metrics,
            config,
            "uncertainty_metrics",
        )

    # 9. 保存预测结果
    results_df = pd.DataFrame(
        {
            "sample_name": sample_names,
            "base_sample_name": [base_sample_name(x) for x in sample_names],
            "variant": [get_variant_name(x) for x in sample_names],
            "true_log_life": y_true,
            "pred_log_life_mean": y_pred_mean,
            "pred_log_life_std": y_pred_std,
            "true_actual_life": y_true_actual,
            "pred_actual_life_mean": y_pred_mean_actual,
            "pred_actual_life_std": y_pred_std_actual,
        }
    )

    results_save_path = os.path.join(
        config["save_fig_dir"],
        "prediction_results_group_split.csv",
    )

    results_df.to_csv(
        results_save_path,
        index=False,
    )
    # 10. 注意力热力图
    if config["visualize_attention"] and config["gate_attention"]:
        visualize_attention_heatmap(
            model,
            test_loader,
            config,
        )

    print("\n===== 全流程执行完成 =====")
