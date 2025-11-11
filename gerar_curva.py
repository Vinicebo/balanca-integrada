import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

df = pd.read_csv("resultados_teste.csv")

thresholds = np.linspace(0.4, 0.95, 12)
f1s = []
coverages = []

for t in thresholds:
    accepted = df[df['p_top1'] >= t]
    coverage = len(accepted) / len(df)
    coverages.append(coverage)

    if len(accepted) == 0:
        f1s.append(np.nan)
    else:
        f1 = f1_score(accepted['gt_label'], accepted['pred_label'], average='macro', zero_division=0)
        f1s.append(f1)

# --- Exibe e plota ---
for t, f1 in zip(thresholds, f1s):
    print(f"Threshold {t:.2f} -> F1 = {f1:.3f}")

plt.plot(thresholds, f1s, marker='o', label='F1 (macro)')
plt.plot(thresholds, coverages, marker='x', linestyle='--', label='Coverage')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Curve F1 vs Threshold (YOLO - Fruit detection)')
plt.grid(True)
plt.legend()
plt.show()
