""
21512采面特征级融合（KPCA）脚本
功能：读取融合瓦斯浓度与预处理后传感器数据 → 构建特征向量 → KPCA降维 → 保存结果
输入：data_fused.xlsx（数据级融合结果）、processed_data.xlsx（预处理后数据）
输出：data_with_pcs.xlsx、kpca_variance.xlsx、kpca主成分曲线.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
fonts = [f.name for f in fm.fontManager.ttflist]
if '宋体' in fonts:
    plt.rcParams['font.sans-serif'] = ['宋体']
elif 'SimSun' in fonts:
    plt.rcParams['font.sans-serif'] = ['SimSun']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

# ==================== 路径设置 ====================
data_dir = r'C:\Users\陈泽秋\Desktop\SQlite_db'
fused_path = os.path.join(data_dir, 'data_fused.xlsx')
processed_path = os.path.join(data_dir, 'processed_data.xlsx')
output_data_path = os.path.join(data_dir, 'data_with_pcs.xlsx')
output_variance_path = os.path.join(data_dir, 'kpca_variance.xlsx')
output_png = os.path.join(data_dir, 'kpca主成分曲线.png')

# ==================== 1. 读取数据 ====================
print("正在读取融合瓦斯浓度数据...")
df_fused = pd.read_excel(fused_path)
df_fused['timestamp'] = pd.to_datetime(df_fused['timestamp'])
df_fused = df_fused.set_index('timestamp')

print("正在读取预处理后传感器数据...")
df_processed = pd.read_excel(processed_path)
df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
df_processed = df_processed.set_index('timestamp')

# ==================== 2. 构建12维特征向量 ====================
print("\n正在构建特征向量...")

# 特征列表（融合瓦斯浓度 + 11个传感器参数，已标准化）
feature_sensors = [
    '39A14',  # 回风巷风速
    '39A15',  # 回风巷温度
    '39A16',  # 一氧化碳浓度
    '38A04',  # 氧气浓度
    '35A09',  # 抽采管路瓦斯浓度
    '35A10',  # 抽采管路瓦斯流量
    '35A11',  # 抽采管路瓦斯压力
    '35A12',  # 抽采管路瓦斯温度
    '39A07',  # 高位钻机甲烷浓度
    '38A09',  # 二氧化碳浓度
    '40D14',  # 风向
]

# 对齐时间
common_start = max(df_fused.index.min(), df_processed.index.min())
common_end = min(df_fused.index.max(), df_processed.index.max())
time_grid = pd.date_range(common_start, common_end, freq='1min')

df_features = pd.DataFrame(index=time_grid)
df_features['CH4_fused'] = df_fused['CH4_fused'].reindex(time_grid)

for sensor in feature_sensors:
    if sensor in df_processed.columns:
        df_features[sensor] = df_processed[sensor].reindex(time_grid)
    else:
        print(f"警告: {sensor} 不在预处理数据中")

# 剔除粉尘传感器（38A02，始终为0）后共12维特征
feature_cols = ['CH4_fused'] + feature_sensors
df_features = df_features[feature_cols].dropna()
print(f"特征矩阵: {df_features.shape[0]} 行 × {df_features.shape[1]} 列")
print(f"特征列表: {feature_cols}")

# ==================== 3. KPCA降维 ====================
print("\n正在进行KPCA降维...")

# 子集拟合（前2000个点），全量变换
n_fit = min(2000, len(df_features))
df_fit = df_features.iloc[:n_fit]
df_full = df_features

# KPCA模型（RBF核，gamma=0.1）
kpca = KernelPCA(n_components=5, kernel='rbf', gamma=0.1, fit_inverse_transform=False)
kpca.fit(df_fit)

# 全量变换
pcs_all = kpca.transform(df_full)

# 计算方差贡献率
eigenvalues = kpca.eigenvalues_
total_eigenvalues = np.sum(eigenvalues)
explained_var_ratio = eigenvalues / total_eigenvalues * 100
cumsum_var_ratio = np.cumsum(explained_var_ratio)

print("\n各主成分方差贡献率及累积贡献率:")
for i in range(5):
    print(f"  PC{i+1}: 贡献率={explained_var_ratio[i]:.2f}%, 累积={cumsum_var_ratio[i]:.2f}%")

# ==================== 4. 保存KPCA结果 ====================
# 保存主成分数据
df_pcs = pd.DataFrame(index=df_full.index)
for i in range(4):
    df_pcs[f'PC{i+1}'] = pcs_all[:, i]

df_output = pd.concat([df_full, df_pcs], axis=1)
df_output = df_output.reset_index().rename(columns={'index': 'timestamp'})
df_output.to_excel(output_data_path, index=False, engine='openpyxl')
print(f"\n主成分数据已保存至: {output_data_path}")

# 保存方差贡献率
df_variance = pd.DataFrame({
    '主成分': [f'PC{i+1}' for i in range(5)],
    '方差贡献率(%)': [round(v, 2) for v in explained_var_ratio],
    '累积贡献率(%)': [round(v, 2) for v in cumsum_var_ratio]
})
df_variance.to_excel(output_variance_path, index=False, engine='openpyxl')
print(f"方差贡献率已保存至: {output_variance_path}")

# ==================== 5. 可视化前4个主成分 ====================
print("\n正在生成主成分时间序列图...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

pc_names = ['PC1', 'PC2', 'PC3', 'PC4']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, ax in enumerate(axes):
    ax.plot(df_pcs.index[:n_fit], df_pcs[pc_names[i]].iloc[:n_fit],
            color=colors[i], linewidth=1.2)
    ax.set_ylabel(pc_names[i], fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('时间', fontsize=14)
x_ticks = df_pcs.index[::200][:n_fit//200+1]
x_labels = [t.strftime('%m-%d %H:%M') for t in x_ticks]
axes[-1].set_xticks(x_ticks)
axes[-1].set_xticklabels(x_labels, rotation=30, fontsize=11)

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"主成分曲线图已保存至: {output_png}")
plt.show()

print("\n特征级融合全部完成！")
