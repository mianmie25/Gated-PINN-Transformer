import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1. File path (update this to your local path)
# ------------------------------------------------------------------
CSV_PATH = r"D:\Program Files (x86)\phd\AI\python code\transformer based on bayes and mento-carlo\RNN\rnn_bayes_metrics\prediction_results.csv"

# ------------------------------------------------------------------
# 2. Read data
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Extract required columns
true_log = df["true_log_life"].values
pred_log = df["pred_log_life_mean"].values

# Convert log10 life to actual life
true_life = 10 ** true_log
pred_life = 10 ** pred_log

# ------------------------------------------------------------------
# 3. Plot settings (font-safe for English)
# ------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

# Scatter plot
ax.scatter(
    true_life,
    pred_life,
    s=60,
    edgecolor="black",
    facecolor="white",
    linewidths=0.8,
    label="Test Data"
)

# ------------------------------------------------------------------
# 4. Error bands (1.5× and 2× scatter bands)
# ------------------------------------------------------------------
min_life = min(true_life.min(), pred_life.min())
max_life = max(true_life.max(), pred_life.max())

x_range = np.linspace(min_life, max_life, 500)

# 1.5× error band
ax.plot(x_range, 1.5 * x_range, "--", color="orange", linewidth=1.2, label="1.5× Error Band")
ax.plot(x_range, x_range / 1.5, "--", color="orange", linewidth=1.2)

# 2× error band
ax.plot(x_range, 2.0 * x_range, "-.", color="red", linewidth=1.2, label="2× Error Band")
ax.plot(x_range, x_range / 2.0, "-.", color="red", linewidth=1.2)

# Ideal prediction line
ax.plot(x_range, x_range, "-", color="black", linewidth=1.5, label="Ideal Prediction")

# ------------------------------------------------------------------
# 5. Axes & Labels
# ------------------------------------------------------------------
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Experimental Fatigue Life (cycles)", fontsize=12)
ax.set_ylabel("Predicted Fatigue Life (cycles)", fontsize=12)
ax.set_title("Fatigue Life Prediction with 1.5× and 2× Error Bands", fontsize=13)

ax.grid(True, which="both", linestyle=":", linewidth=0.5)
ax.legend(frameon=True, fontsize=10)

plt.tight_layout()
plt.show()