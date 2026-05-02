import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')


class FatigueDataPreprocessor:
    """基于循环次数的疲劳数据预处理类（力值+塑性应变2维特征 + 傅里叶降采样）"""

    def __init__(self, cycle_ratio=0.3, noise_std=0.01,
                 scale_range=(0.98, 1.02)):
        """
        初始化预处理类
        参数:
        cycle_ratio: float, 目标寿命比例（如0.3表示前30%）
        noise_std: float, 高斯噪声的标准差
        scale_range: tuple, 随机缩放的范围
        """
        self.cycle_ratio = cycle_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range

    def load_data(self, butt_paths, cross_paths):
        """加载对接和十字连接数据（现在包含塑性应变）"""
        self.data = {'butt': [], 'cross': []}
        print("正在加载对接连接数据...")
        for exp_name, path in butt_paths.items():
            try:
                df = pd.read_csv(path, skiprows=1, header=None)
                cycles = df.iloc[:, 1].values          # 第二列：当前循环次数
                axial_force = df.iloc[:, 3].values     # 第四列：轴向力(kN)
                plastic_strain = df.iloc[:, 4].values  # 第五列：塑性应变（新增）
                total_life = np.max(cycles)
                self.data['butt'].append({
                    'exp_name': exp_name,
                    'cycles': cycles,
                    'axial_force': axial_force,
                    'plastic_strain': plastic_strain,  # 保存塑性应变
                    'total_life': total_life,
                    'joint_type': 'butt'
                })
                print(f"  {exp_name}: 总寿命={total_life}")
            except Exception as e:
                print(f"  加载失败 {exp_name}: {e}")
        print("\\n正在加载十字连接数据...")
        for exp_name, path in cross_paths.items():
            try:
                df = pd.read_csv(path, skiprows=1, header=None)
                cycles = df.iloc[:, 1].values
                axial_force = df.iloc[:, 3].values
                plastic_strain = df.iloc[:, 4].values  # 第五列：塑性应变
                total_life = np.max(cycles)
                self.data['cross'].append({
                    'exp_name': exp_name,
                    'cycles': cycles,
                    'axial_force': axial_force,
                    'plastic_strain': plastic_strain,
                    'total_life': total_life,
                    'joint_type': 'cross'
                })
                print(f"  {exp_name}: 总寿命={total_life}")
            except Exception as e:
                print(f"  加载失败 {exp_name}: {e}")
        self.print_data_statistics()

    def print_data_statistics(self):
        """打印数据统计信息"""
        print("\\n=== 数据统计 ===")
        for joint_type in ['butt', 'cross']:
            samples = self.data[joint_type]
            if not samples:
                continue
            total_lives = [s['total_life'] for s in samples]
            print(f"\\n{joint_type.upper()} 连接:")
            print(f"  平均疲劳寿命: {np.mean(total_lives):.0f} ± {np.std(total_lives):.0f}")
            print(f"  最小总循环数: {np.min(total_lives)}, 最大总循环数: {np.max(total_lives)}")
            print(f"  试件数量: {len(samples)}")

    def get_closest_recorded_cycle(self, theoretical_cycle, recorded_cycles):
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
                right_is_round = (unique_cycles[i + 1] % 1000 == 0) or (unique_cycles[i + 1] % 100 == 0) or (
                        unique_cycles[i + 1] % 10 == 0)
                left_is_round = (unique_cycles[i] % 1000 == 0) or (unique_cycles[i] % 100 == 0) or (
                        unique_cycles[i] % 10 == 0)
                if right_is_round and not left_is_round:
                    return unique_cycles[i + 1]
                elif left_is_round and not right_is_round:
                    return unique_cycles[i]
                else:
                    return unique_cycles[i] if diff_left <= diff_right else unique_cycles[i + 1]
        return unique_cycles[np.argmin(np.abs(unique_cycles - theoretical_cycle))]

    def extract_cycles_data(self, cycles, axial_force, plastic_strain, target_cycles):
        indices = np.where(cycles <= target_cycles)[0]
        if len(indices) == 0:
            # 若没有符合条件的循环，则取前1000点（应急处理）
            return axial_force[:1000], plastic_strain[:1000], target_cycles
        axial_extracted = axial_force[indices]
        strain_extracted = plastic_strain[indices]
        actual_cycles_used = np.max(cycles[indices])
        return axial_extracted, strain_extracted, actual_cycles_used

    def downsample_sequence(self, sequence, target_length=1000):
        n_original = len(sequence)
        if n_original <= target_length:
            return sequence

        # 加汉明窗
        window = np.hamming(n_original)
        sequence_windowed = sequence * window

        # FFT
        fft_spectrum = np.fft.fft(sequence_windowed)

        # 保留低频成分（对称截断）
        n_keep = target_length // 2
        fft_downsampled = np.zeros(target_length, dtype=complex)
        fft_downsampled[:n_keep] = fft_spectrum[:n_keep]
        fft_downsampled[-n_keep:] = fft_spectrum[-n_keep:]

        # IFFT
        sequence_downsampled = np.fft.ifft(fft_downsampled).real

        # 幅值归一化
        scale_factor = np.max(np.abs(sequence)) / (np.max(np.abs(sequence_downsampled)) + 1e-8)
        sequence_downsampled = sequence_downsampled * scale_factor

        return sequence_downsampled

    def augment_sequence_pair(self, axial_seq, strain_seq, num_augments=2):
        augmented = [(axial_seq.copy(), strain_seq.copy())]  # 原始样本
        for _ in range(num_augments):
            # 随机缩放（同一因子）
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            axial_aug = axial_seq * scale
            strain_aug = strain_seq * scale

            # 添加高斯噪声（独立）
            noise_std_axial = self.noise_std * np.std(axial_aug)
            noise_axial = np.random.normal(0, noise_std_axial, len(axial_aug))
            axial_aug = axial_aug + noise_axial

            noise_std_strain = self.noise_std * np.std(strain_aug)
            noise_strain = np.random.normal(0, noise_std_strain, len(strain_aug))
            strain_aug = strain_aug + noise_strain

            augmented.append((axial_aug, strain_aug))
        return augmented

    def prepare_dataset_by_cycles(self, augment=True, target_length=1000):
        X_data = []          # 每个元素为形状 (target_length, 2) 的数组
        y_data = []
        joint_types = []
        sample_names = []
        print(f"\\n正在提取前{self.cycle_ratio * 100:.0f}%寿命对应的循环数据...")
        for joint_type, samples in self.data.items():
            print(f"  处理 {joint_type} 连接数据...")
            for sample in samples:
                cycles = sample['cycles']
                axial_force = sample['axial_force']
                plastic_strain = sample['plastic_strain']
                total_life = sample['total_life']
                exp_name = sample['exp_name']

                theoretical_target = total_life * self.cycle_ratio
                actual_target_cycles = self.get_closest_recorded_cycle(theoretical_target, cycles)

                if total_life <= actual_target_cycles:
                    print(f"    跳过 {exp_name}: 总寿命({total_life}) <= 目标循环数({actual_target_cycles})")
                    continue

                # 提取前N个循环的力值和应变数据
                axial_seg, strain_seg, actual_cycles_used = self.extract_cycles_data(
                    cycles, axial_force, plastic_strain, actual_target_cycles
                )
                if len(axial_seg) < 50 or len(strain_seg) < 50:
                    print(f"    跳过 {exp_name}: 提取的数据过短({len(axial_seg)})")
                    continue

                # 分别进行傅里叶降采样
                axial_down = self.downsample_sequence(axial_seg, target_length)
                strain_down = self.downsample_sequence(strain_seg, target_length)

                # 标签：对数剩余寿命
                remaining_life = total_life - actual_cycles_used
                log_remaining_life = np.log10(remaining_life)

                # 数据增强（对力和应变同时处理）
                if augment:
                    augmented_pairs = self.augment_sequence_pair(axial_down, strain_down)
                else:
                    augmented_pairs = [(axial_down, strain_down)]

                # 提取特征（标准化后堆叠为2维）
                for aug_idx, (ax_aug, st_aug) in enumerate(augmented_pairs):
                    # 分别标准化
                    ax_mean, ax_std = np.mean(ax_aug), np.std(ax_aug) + 1e-8
                    st_mean, st_std = np.mean(st_aug), np.std(st_aug) + 1e-8
                    ax_norm = (ax_aug - ax_mean) / ax_std
                    st_norm = (st_aug - st_mean) / st_std

                    # 堆叠成 (target_length, 2) 的特征矩阵
                    features = np.column_stack((ax_norm, st_norm))

                    X_data.append(features)
                    y_data.append(log_remaining_life)
                    joint_types.append(joint_type)
                    sample_suffix = "_original" if aug_idx == 0 else f"_aug{aug_idx}"
                    sample_names.append(f"{exp_name}{sample_suffix}")

                    if aug_idx == 0:
                        print(f"    {exp_name}: 总寿命={total_life}, "
                              f"前{self.cycle_ratio * 100:.0f}%理论循环={theoretical_target:.0f}, "
                              f"匹配实际循环={actual_target_cycles}, "
                              f"剩余寿命={remaining_life:.1f}, "
                              f"原始点数={len(axial_seg)}, "
                              f"傅里叶降采样后={target_length}, "
                              f"特征维度=2")

        y_data = np.array(y_data)
        print(f"\\n数据集准备完成:")
        print(f"  总样本数量: {len(X_data)}")
        if len(y_data) > 0:
            print(f"  目标值范围: {y_data.min():.2f} ~ {y_data.max():.2f} (log10)")
            print(f"  实际剩余寿命范围: {10 ** y_data.min():.0f} ~ {10 ** y_data.max():.0f} 循环")
        print(f"  连接类型分布: 对接 {joint_types.count('butt')} 个, 十字 {joint_types.count('cross')} 个")
        print(f"  特征维度: {X_data[0].shape[1] if X_data else 0}")
        print(f"  序列长度: {len(X_data[0]) if X_data else 0}")

        return X_data, y_data, joint_types, sample_names

    def analyze_and_save(self, X_data, y_data, joint_types, sample_names):
        """分析和保存数据（适配2维特征）"""
        if len(y_data) == 0:
            print("错误: 没有有效数据!")
            return

        # 统计分析（同原代码，不变）
        print("\\n=== 统计分析 ===")
        butt_indices = [i for i, t in enumerate(joint_types) if t == 'butt']
        cross_indices = [i for i, t in enumerate(joint_types) if t == 'cross']
        butt_life = y_data[butt_indices] if butt_indices else np.array([])
        cross_life = y_data[cross_indices] if cross_indices else np.array([])

        if len(butt_life) > 0:
            print(f"对接连接剩余寿命(log10): {np.mean(butt_life):.3f} ± {np.std(butt_life):.3f}")
            print(f"  实际剩余寿命: {10 ** np.mean(butt_life):.0f} ± {10 ** np.std(butt_life):.0f} 循环")
        if len(cross_life) > 0:
            print(f"十字连接剩余寿命(log10): {np.mean(cross_life):.3f} ± {np.std(cross_life):.3f}")
            print(f"  实际剩余寿命: {10 ** np.mean(cross_life):.0f} ± {10 ** np.std(cross_life):.0f} 循环")

        if len(butt_life) > 1 and len(cross_life) > 1:
            t_stat, p_value = stats.ttest_ind(butt_life, cross_life, equal_var=False)
            print(f"差异性检验 (Welch's t-test): t={t_stat:.3f}, p={p_value:.4f}")
            print("  结论: 两种连接方式的剩余疲劳寿命存在显著差异 (p<0.05)" if p_value < 0.05 else
                  "  结论: 两种连接方式的剩余疲劳寿命差异不显著")
        else:
            print("  警告: 样本不足，无法进行t检验")

        # 可视化（调整了降采样对比图，显示两个通道）
        self.plot_separate_charts(X_data, y_data, joint_types, sample_names)
        # 保存数据
        self.save_to_csv(X_data, y_data, joint_types, sample_names)

        return butt_life, cross_life

    def plot_separate_charts(self, X_data, y_data, joint_types, sample_names):
        """绘制可视化图表（标注2维特征）"""
        output_dir = f'charts_{self.cycle_ratio * 100:.0f}%life_fft_downsample'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 连接方式对比箱线图（同原代码）
        plt.figure(figsize=(10, 6))
        plot_data = []
        labels = []
        if butt_indices := [i for i, t in enumerate(joint_types) if t == 'butt']:
            plot_data.append(y_data[butt_indices])
            labels.append('Butt Joint')
        if cross_indices := [i for i, t in enumerate(joint_types) if t == 'cross']:
            plot_data.append(y_data[cross_indices])
            labels.append('Cross Joint')
        if plot_data:
            plt.boxplot(plot_data, labels=labels)
            plt.ylabel('Log10(Remaining Life)', fontsize=12)
            plt.title(
                f'Remaining Life (FFT Downsampling to 1000 Steps, 2D Features)\\nAfter {self.cycle_ratio * 100:.0f}% Life Cycles',
                fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'joint_comparison.png'), dpi=300, bbox_inches='tight')
            plt.show()
        plt.close()

        # 2. 目标值分布图（同原代码）
        plt.figure(figsize=(10, 6))
        plt.hist(y_data, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Log10(Remaining Life)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Target Values (FFT Downsampling, 2D Features)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'target_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # 3. 傅里叶降采样前后信号对比（力值和塑性应变各画一个子图）
        if X_data:
            sample_idx = 0
            sample_name = sample_names[sample_idx].split('_')[0]
            joint_type = joint_types[sample_idx]
            original_sample = next(s for s in self.data[joint_type] if s['exp_name'] == sample_name)

            # 提取原始未降采样数据段
            theoretical_target = original_sample['total_life'] * self.cycle_ratio
            actual_target_cycles = self.get_closest_recorded_cycle(theoretical_target, original_sample['cycles'])
            axial_orig, strain_orig, _ = self.extract_cycles_data(
                original_sample['cycles'],
                original_sample['axial_force'],
                original_sample['plastic_strain'],
                actual_target_cycles
            )

            # 降采样后的数据（力和应变）
            axial_down = X_data[sample_idx][:, 0]  # 第一列力
            strain_down = X_data[sample_idx][:, 1]  # 第二列应变

            plt.figure(figsize=(14, 10))

            # 力信号对比
            plt.subplot(2, 2, 1)
            plt.plot(axial_orig[:200], color='gray', linewidth=1, label='Original Force')
            plt.xlabel('Time Step (Original)', fontsize=10)
            plt.ylabel('Axial Force (kN)', fontsize=10)
            plt.title('Original Force (First 200 Steps)', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            plt.plot(axial_down[:200], color='red', linewidth=1.5, label='FFT Downsampled Force')
            plt.xlabel('Time Step (Downsampled)', fontsize=10)
            plt.ylabel('Normalized Force', fontsize=10)
            plt.title('Downsampled Force (First 200 Steps)', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 塑性应变对比
            plt.subplot(2, 2, 3)
            plt.plot(strain_orig[:200], color='gray', linewidth=1, label='Original Strain')
            plt.xlabel('Time Step (Original)', fontsize=10)
            plt.ylabel('Plastic Strain', fontsize=10)
            plt.title('Original Plastic Strain (First 200 Steps)', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            plt.plot(strain_down[:200], color='blue', linewidth=1.5, label='FFT Downsampled Strain')
            plt.xlabel('Time Step (Downsampled)', fontsize=10)
            plt.ylabel('Normalized Strain', fontsize=10)
            plt.title('Downsampled Strain (First 200 Steps)', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fft_downsampling_comparison_2d.png'), dpi=300, bbox_inches='tight')
            plt.show()
        plt.close()

        print(f"\\n已保存所有图表到 {output_dir}/ 目录")

    def save_to_csv(self, X_data, y_data, joint_types, sample_names):
        """保存数据（2维特征）"""
        output_dir = f'preprocessed_data_{self.cycle_ratio * 100:.0f}%life_fft_downsample'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"\\n正在保存数据到目录: {output_dir}")

        # 保存特征文件（列名改为 force 和 strain）
        features_dir = os.path.join(output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        feature_columns = ['normalized_force', 'normalized_plastic_strain']  # 两列
        for i, (features, sample_name) in enumerate(zip(X_data, sample_names)):
            safe_name = sample_name.replace('\\\\', '_').replace('/', '_').replace(':', '_')
            file_path = os.path.join(features_dir, f'sample_{i:03d}_{safe_name}.csv')
            pd.DataFrame(features, columns=feature_columns).to_csv(file_path, index=False)
        print(f"  已保存 {len(X_data)} 个特征文件到 {features_dir}/")

        # 保存元数据（更新特征维度和列信息）
        metadata = []
        for i, (y, joint_type, sample_name) in enumerate(zip(y_data, joint_types, sample_names)):
            safe_name = sample_name.replace('\\\\', '_').replace('/', '_').replace(':', '_')
            metadata.append({
                'sample_id': i,
                'sample_name': sample_name,
                'joint_type': joint_type,
                'log_remaining_life': y,
                'actual_remaining_life': 10 ** y,
                'feature_file': f'sample_{i:03d}_{safe_name}.csv',
                'sequence_length': len(X_data[i]),
                'feature_dimensions': 2,  # 修改为2
                'feature_columns': 'force, strain',
                'cycle_ratio': self.cycle_ratio,
                'downsampling_method': 'FFT (Fourier Transform)',
                'target_length': 1000
            })
        pd.DataFrame(metadata).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        print(f"  已保存元数据到 metadata.csv")

        # 保存数据集统计
        stats = {
            'total_samples': len(X_data),
            'butt_samples': joint_types.count('butt'),
            'cross_samples': joint_types.count('cross'),
            'cycle_ratio': self.cycle_ratio,
            'downsampling_method': 'FFT (Fourier Transform)',
            'target_sequence_length': 1000,
            'feature_dimensions': 2,  # 修改为2
            'min_log_life': np.min(y_data) if len(y_data) > 0 else 0,
            'max_log_life': np.max(y_data) if len(y_data) > 0 else 0,
            'min_actual_life': np.min(10 ** y_data) if len(y_data) > 0 else 0,
            'max_actual_life': np.max(10 ** y_data) if len(y_data) > 0 else 0,
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, 'dataset_statistics.csv'), index=False)
        print(f"  已保存数据集统计信息到 dataset_statistics.csv")
        print(f"\\n所有数据已成功保存到 {output_dir}/")


# 使用示例（路径保持不变）
if __name__ == "__main__":
    # 数据路径配置（根据实际路径修改）
    butt_paths = {
        "对接试验1": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验1_处理后.csv",
        "对接试验2": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验2_处理后.csv",
        "对接试验3": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验3_处理后.csv",
        "对接试验4": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验4_处理后.csv",
        "对接试验5": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验5_处理后.csv",
        "对接试验7": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验7_处理后.csv",
        "对接试验11": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验11_处理后.csv",
        "对接试验12": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验12_处理后.csv",
        "对接试验13": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验13_处理后.csv",
        "对接试验14": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验14_处理后.csv",
        "对接试验15": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验15_处理后.csv",
        "对接试验16": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验16_处理后.csv",
        "对接试验17": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验17_处理后.csv",
        "对接试验18": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验18_处理后.csv",
        "对接试验19": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验19_处理后.csv",
        "对接试验20": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\对接试验20_处理后.csv"
    }

    cross_paths = {
        "十字试验1": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验1_处理后.csv",
        "十字试验2": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验2_处理后.csv",
        "十字试验3": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验3_处理后.csv",
        "十字试验4": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验4_处理后.csv",
        "十字试验5": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验5_处理后.csv",
        "十字试验6": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验6_处理后.csv",
        "十字试验7": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验7_处理后.csv",
        "十字试验8": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验8_处理后.csv",
        "十字试验9": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验9_处理后.csv",
        "十字试验10": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验10_处理后.csv",
        "十字试验11": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验11_处理后.csv",
        "十字试验12": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验12_处理后.csv",
        "十字试验13": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验13_处理后.csv",
        "十字试验14": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验14_处理后.csv",
        "十字试验15": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验15_处理后.csv",
        "十字试验17": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验17_处理后.csv",
        "十字试验18": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验18_处理后.csv",
        "十字试验19": r"D:\\\\Program Files (x86)\\\\phd\\\\AI\\\\python code\\\\transformer based on bayes and mento-carlo\\\\十字试验19_处理后.csv"
    }

    # 1. 创建预处理器
    preprocessor = FatigueDataPreprocessor(
        cycle_ratio=0.3,  # 示例：使用全寿命数据
        noise_std=0.01,
        scale_range=(0.98, 1.02)
    )

    # 2. 加载数据
    preprocessor.load_data(butt_paths, cross_paths)

    # 3. 准备数据集（开启增强，傅里叶降采样到1000长度，特征维度2）
    X_data, y_data, joint_types, sample_names = preprocessor.prepare_dataset_by_cycles(
        augment=True,
        target_length=1000
    )

    # 4. 分析和保存数据
    preprocessor.analyze_and_save(X_data, y_data, joint_types, sample_names)
    print("\\n预处理完成!")