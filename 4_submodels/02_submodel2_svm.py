"""
子模型2：动力灾害预警子模型（SVM）
输入：data_with_pcs.xlsx, patrol_features.xlsx
输出：submodel2_svm.pkl, SVM_评估图.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 修复中文路径编码问题 ====================
os.environ['JOBLIB_TEMP_FOLDER'] = r'C:\temp_joblib'
if not os.path.exists(r'C:\temp_joblib'):
    os.makedirs(r'C:\temp_joblib', exist_ok=True)

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
pcs_path = os.path.join(data_dir, 'data_with_pcs.xlsx')
patrol_path = os.path.join(data_dir, 'patrol_features.xlsx')
model_output = os.path.join(data_dir, 'submodel2_svm.pkl')
curve_output = os.path.join(data_dir, 'SVM_评估图.png')

# ==================== 1. 读取数据并对齐 ====================
print("正在读取数据...")
df_pcs = pd.read_excel(pcs_path)
df_pcs['timestamp'] = pd.to_datetime(df_pcs['timestamp'])
df_pcs = df_pcs.set_index('timestamp')

df_patrol = pd.read_excel(patrol_path)
df_patrol['timestamp'] = pd.to_datetime(df_patrol['timestamp'])
df_patrol = df_patrol.set_index('timestamp')

common_start = max(df_pcs.index.min(), df_patrol.index.min())
common_end = min(df_pcs.index.max(), df_patrol.index.max())
df_pcs = df_pcs.loc[common_start:common_end]
df_patrol = df_patrol.loc[common_start:common_end]

pc_cols = ['PC1', 'PC2', 'PC3', 'PC4']
X_pcs = df_pcs[pc_cols].values
ch4_data = df_pcs['CH4_fused'].values

patrol_cols = ['I_sw', 'I_sp', 'I_hf', 'I_od', 'f_cb', 'd_hf', 'I_at']
X_patrol = df_patrol[patrol_cols].values

print(f"对齐后样本数: {len(X_pcs)}")

# ==================== 2. 构建标签 ====================
print("\n正在构建标签...")
future_steps = 12
threshold_abs = 0.8
threshold_rise = 0.3

labels = np.zeros(len(ch4_data) - future_steps, dtype=int)
for i in range(len(labels)):
    window = ch4_data[i+1 : i+1+future_steps]
    current = ch4_data[i]
    if np.any(window > threshold_abs) or np.any(window - current > threshold_rise):
        labels[i] = 1

pos_ratio = labels.mean()
print(f"阈值: 绝对>{threshold_abs}% 或 上升>{threshold_rise}% | 正样本占比: {pos_ratio*100:.2f}%")

if pos_ratio < 0.03:
    print("正样本过少，尝试调整阈值...")
    for new_th_abs in [0.7, 0.6, 0.5]:
        for new_th_rise in [0.25, 0.2, 0.15]:
            temp_labels = np.zeros(len(ch4_data) - future_steps, dtype=int)
            for i in range(len(temp_labels)):
                window = ch4_data[i+1 : i+1+future_steps]
                if np.any(window > new_th_abs) or np.any(window - ch4_data[i] > new_th_rise):
                    temp_labels[i] = 1
            ratio = temp_labels.mean()
            if 0.03 <= ratio <= 0.30:
                threshold_abs = new_th_abs
                threshold_rise = new_th_rise
                labels = temp_labels
                pos_ratio = ratio
                print(f"  新阈值: 绝对>{threshold_abs}% 或 上升>{threshold_rise}% -> 正样本占比: {pos_ratio*100:.2f}%")
                break
        if 0.03 <= pos_ratio <= 0.30:
            break

print(f"最终正样本占比: {pos_ratio*100:.2f}%")

X_pcs = X_pcs[:-future_steps]
X_patrol = X_patrol[:-future_steps]
min_len = min(len(X_pcs), len(X_patrol), len(labels))
X_pcs = X_pcs[:min_len]
X_patrol = X_patrol[:min_len]
labels = labels[:min_len]

np.random.seed(42)
v_mining = np.random.uniform(0, 0.25, size=min_len)

X = np.column_stack([X_pcs, X_patrol, v_mining])
print(f"最终样本数: {len(X)}, 特征维度: {X.shape[1]}, 正样本数: {labels.sum()}")

# ==================== 3. 标准化与划分 ====================
print("\n正在划分数据集...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, labels, test_size=0.4, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# ==================== 4. SVM训练（单线程，避免中文路径编码错误） ====================
print("\n正在训练SVM（单线程网格搜索）...")
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 0.5, 1],
    'class_weight': ['balanced']
}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc', n_jobs=1, verbose=1)
grid.fit(X_train, y_train)

print(f"\n最佳参数: {grid.best_params_}, 最佳交叉验证AUC: {grid.best_score_:.4f}")
best_model = grid.best_estimator_

# ==================== 5. 测试评估 ====================
print("\n测试评估中...")
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_auc = roc_auc_score(y_test, y_prob)

print(f"准确率: {test_acc:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}")
print(f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

joblib.dump({'model': best_model, 'scaler': scaler}, model_output)
print(f"模型已保存至: {model_output}")

# ==================== 6. 绘图 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.matshow(cm, cmap='Blues', alpha=0.7)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
ax1.set_xlabel('预测标签', fontsize=14)
ax1.set_ylabel('真实标签', fontsize=14)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_title('(a) 混淆矩阵', fontsize=15, fontweight='bold', y=-0.15)

ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={test_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlabel('假阳性率', fontsize=14)
ax2.set_ylabel('真阳性率', fontsize=14)
ax2.legend(fontsize=13)
ax2.set_title('(b) ROC曲线', fontsize=15, fontweight='bold', y=-0.15)

plt.tight_layout()
plt.savefig(curve_output, dpi=300, bbox_inches='tight')
print(f"评估图已保存至: {curve_output}")
plt.show()

print("\n子模型2全部完成！")
