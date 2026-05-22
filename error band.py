import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1. 定义5个CSV文件路径（对应case1-case5）
# ------------------------------------------------------------------
CSV_PATHS = [
    r"D:\Program Files (x86)\phd\AI\python code\transformer based on bayes and mento-carlo\transformer\transformer_bayes_metrics\prediction_results.csv",
    r"D:\Program Files (x86)\phd\AI\python code\transformer based on bayes and mento-carlo\LSTM\lstm_bayes_metrics\prediction_results.csv",
    r"D:\Program Files (x86)\phd\AI\python code\transformer based on bayes and mento-carlo\RNN\rnn_bayes_metrics\prediction_results.csv",
]

# 定义每个case的样式（标记、填充色、边缘色、标签）
case_styles = [
    {"marker": "o", "facecolor": "white", "edgecolor": "blue", "label": "Case 6"},
    {"marker": "s", "facecolor": "white", "edgecolor": "green", "label": "Case 7"},
    {"marker": "^", "facecolor": "white", "edgecolor": "purple", "label": "Case 8"},
]

# ------------------------------------------------------------------
# 2. 绘图基础设置
# ------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

# ------------------------------------------------------------------
# 3. 循环读取每个case的数据并绘制散点
# ------------------------------------------------------------------
for idx, csv_path in enumerate(CSV_PATHS):
    # 读取数据
    df = pd.read_csv(csv_path)

    # 提取并转换寿命数据（log10转实际值）
    true_log = df["true_log_life"].values
    pred_log = df["pred_log_life_mean"].values
    true_life = 10 ** true_log
    pred_life = 10 ** pred_log

    # 绘制当前case的散点
    style = case_styles[idx]
    ax.scatter(
        true_life,
        pred_life,
        s=60,
        marker=style["marker"],
        edgecolor=style["edgecolor"],
        facecolor=style["facecolor"],
        linewidths=0.8,
        label=style["label"]
    )

# ------------------------------------------------------------------
# 4. 绘制误差带和理想预测线
# ------------------------------------------------------------------
min_life = 1000
max_life = 10000
x_range = np.linspace(min_life, max_life, 500)

# 1.5×误差带
ax.plot(x_range, 1.5 * x_range, "--", color="orange", linewidth=1.2, label="1.5× Error Band")
ax.plot(x_range, x_range / 1.5, "--", color="orange", linewidth=1.2)

# 2×误差带
ax.plot(x_range, 2.0 * x_range, "-.", color="red", linewidth=1.2, label="2× Error Band")
ax.plot(x_range, x_range / 2.0, "-.", color="red", linewidth=1.2)

# 理想预测线
ax.plot(x_range, x_range, "-", color="black", linewidth=1.5, label="Ideal Prediction")

# ------------------------------------------------------------------
# 5. 坐标轴与样式设置（严格匹配原要求）
# ------------------------------------------------------------------
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(10 ** 3, 10 ** 4)
ax.set_ylim(10 ** 3, 10 ** 4)
ax.margins(x=0, y=0)
ax.set_aspect('equal')

# 坐标轴标签（字体24）
ax.set_xlabel("Experimental Fatigue Life (cycles)", fontsize=24)
ax.set_ylabel("Predicted Fatigue Life (cycles)", fontsize=24)

# 刻度字体（主次刻度均20）
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

# 网格与图例
ax.grid(True, which="both", linestyle=":", linewidth=0.5)
ax.legend(frameon=True, fontsize=17, loc="best")  # 图例字体14，自动最优位置

plt.tight_layout()
plt.show()
