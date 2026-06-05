"""
子模型1：LSTM时序预警模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
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
pcs_path = os.path.join(data_dir, 'data_with_pcs.xlsx')
model_output = os.path.join(data_dir, 'submodel1_lstm.pth')
curve_output = os.path.join(data_dir, 'LSTM训练曲线.png')

# ==================== 1. 读取数据 ====================
print("正在读取数据...")
df_fused = pd.read_excel(fused_path)
df_fused['timestamp'] = pd.to_datetime(df_fused['timestamp'])
df_fused = df_fused.set_index('timestamp')

df_pcs = pd.read_excel(pcs_path)
df_pcs['timestamp'] = pd.to_datetime(df_pcs['timestamp'])
df_pcs = df_pcs.set_index('timestamp')

common_start = max(df_fused.index.min(), df_pcs.index.min())
common_end = min(df_fused.index.max(), df_pcs.index.max())
df_fused = df_fused.loc[common_start:common_end]
df_pcs = df_pcs.loc[common_start:common_end]

raw_ch4 = df_fused['CH4_fused'].values.astype(np.float32)
pc_data = df_pcs[['PC1','PC2','PC3','PC4']].values.astype(np.float32)

print(f"对齐后样本数: {len(raw_ch4)}")

# ==================== 2. 标签定义 ====================
look_ahead = 6
threshold = 0.5

labels = np.zeros(len(raw_ch4) - look_ahead, dtype=np.float32)
for i in range(len(labels)):
    future_window = raw_ch4[i+1 : i+1+look_ahead]
    labels[i] = 1.0 if np.any(future_window > threshold) else 0.0

pos_ratio = labels.mean()
print(f"\n阈值={threshold}%, 未来{look_ahead}步, 正样本占比: {pos_ratio*100:.2f}%")

pc_data = pc_data[:-look_ahead]
min_len = min(len(pc_data), len(labels))
pc_data = pc_data[:min_len]
labels = labels[:min_len]

# ==================== 3. 特征标准化 ====================
print("\n正在进行特征标准化...")
scaler = StandardScaler()
pc_data = scaler.fit_transform(pc_data).astype(np.float32)

# ==================== 4. 构建时序样本 ====================
seq_len = 12
n_features = 4

def create_sequences(data, labels, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len + 1):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len-1])
    return np.array(X), np.array(y)

X, y = create_sequences(pc_data, labels, seq_len)
print(f"时序样本数: {len(X)}, 输入形状: {X.shape}")

# ==================== 5. 划分数据集 ====================
if len(np.unique(y)) > 1:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
else:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {len(X_train)} (正样本: {y_train.sum():.0f}), "
      f"验证集: {len(X_val)} (正样本: {y_val.sum():.0f}), "
      f"测试集: {len(X_test)} (正样本: {y_test.sum():.0f})")

# ==================== 6. 过采样处理（训练集） ====================
print("\n正在进行过采样处理...")
# 计算正负样本权重用于采样
class_counts = np.bincount(y_train.astype(int))
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train.astype(int)]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

# ==================== 7. LSTM模型（简化版，无BatchNorm） ====================
class LSTM_Classifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze(1)

model = LSTM_Classifier(input_size=n_features, hidden_size=64, num_layers=2, dropout=0.3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters())}")

# ==================== 8. 训练 ====================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

num_epochs = 150
best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
early_stop_patience = 30

train_losses, val_losses, val_accuracies, val_positive_ratios = [], [], [], []

print(f"\n开始训练 ({num_epochs}轮，含早停)...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss = 0
    all_preds, all_probs, all_trues = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_trues.extend(targets.cpu().numpy())

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_trues, all_preds)
    val_pos_ratio = np.mean(all_preds)  # 预测为正类的比例

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    val_positive_ratios.append(val_pos_ratio)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), model_output.replace('.pth', '_best.pth'))
    else:
        patience_counter += 1

    if (epoch+1) % 25 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | 训练损失: {avg_train_loss:.4f} | "
              f"验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
              f"正类预测比: {val_pos_ratio:.4f} | LR: {lr:.6f}")

    if patience_counter >= early_stop_patience:
        print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
        break

# 加载最佳模型
model.load_state_dict(torch.load(model_output.replace('.pth', '_best.pth')))
print(f"加载最佳模型（第 {best_epoch} 轮，验证准确率: {best_val_acc:.4f}）")

# ==================== 9. 测试评估 ====================
print("\n测试评估中...")
model.eval()
with torch.no_grad():
    y_prob = model(X_test_t.to(device)).cpu().numpy()
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_pred = (y_prob > 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
if len(np.unique(y_test)) > 1:
    test_auc = roc_auc_score(y_test, y_prob)
else:
    test_auc = 0.5

print(f"准确率: {test_acc:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}")
print(f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:\n{cm}")

# 保存模型
final_epoch = len(train_losses)
torch.save({
    'model_state_dict': model.state_dict(),
    'threshold': threshold,
    'seq_len': seq_len,
    'look_ahead': look_ahead,
    'best_epoch': best_epoch,
    'metrics': {'acc': test_acc, 'prec': test_precision, 'rec': test_recall,
                'f1': test_f1, 'auc': test_auc},
    'confusion_matrix': cm.tolist(),
    'scaler': scaler
}, model_output)
print(f"模型已保存至: {model_output}")

# ==================== 10. 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

epochs_range = range(1, final_epoch+1)

ax1 = axes[0]
ax1.plot(epochs_range, train_losses, 'b-', lw=1.5, label='训练损失')
ax1.plot(epochs_range, val_losses, 'r-', lw=1.5, label='验证损失')
ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'最佳轮次({best_epoch})')
ax1.set_xlabel('训练轮次', fontsize=14)
ax1.set_ylabel('损失', fontsize=14)
ax1.legend(fontsize=12)
ax1.set_title('(a) 训练损失曲线', fontsize=15, fontweight='bold', y=-0.25)

ax2 = axes[1]
ax2.plot(epochs_range, val_accuracies, 'g-', lw=1.5, label='验证准确率')
ax2.plot(epochs_range, val_positive_ratios, 'orange', lw=1.5, linestyle='--', label='正类预测比例')
ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('训练轮次', fontsize=14)
ax2.set_ylabel('比例', fontsize=14)
ax2.legend(fontsize=12)
ax2.set_title('(b) 验证准确率与正类预测比例', fontsize=15, fontweight='bold', y=-0.25)

ax3 = axes[2]
if len(np.unique(y_test)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={test_auc:.4f})')
ax3.plot([0,1],[0,1], 'k--', lw=1)
ax3.set_xlabel('假阳性率', fontsize=14)
ax3.set_ylabel('真阳性率', fontsize=14)
ax3.legend(fontsize=12)
cm_text = f'TN={cm[0,0]}  FP={cm[0,1]}\nFN={cm[1,0]}  TP={cm[1,1]}'
ax3.text(0.6, 0.25, cm_text, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
ax3.set_title('(c) ROC曲线与混淆矩阵', fontsize=15, fontweight='bold', y=-0.25)

plt.tight_layout()
plt.savefig(curve_output, dpi=300, bbox_inches='tight')
print(f"训练曲线已保存至: {curve_output}")
plt.show()

print("\n子模型1最终版全部完成！")
