"""
21512采面数据级融合与对比图生成（第1段）
功能：读取原始数据、1分钟重采样、时间对齐
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

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
db_dir = r'C:\Users\陈泽秋\Desktop\SQlite_db\cqu_SQLite'
fused_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\data_fused.xlsx'
output_dir = r'C:\Users\陈泽秋\Desktop'

# 备份已有融合文件
if os.path.exists(fused_path):
    backup_path = fused_path.replace('.xlsx', '_backup.xlsx')
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(fused_path, backup_path)
    print(f"原融合文件已备份至: {backup_path}")

# ==================== 传感器列表与权重 ====================
methane_sensors = ['38A03', '38A01', '39A13', '34A01', '40A05']

weights = {
    '38A03': 0.32,
    '38A01': 0.35,
    '39A13': 0.23,
    '34A01': 0.06,
    '40A05': 0.04
}

# ==================== 1. 读取原始数据（合并两个数据表） ====================
print("正在从SQLite数据库读取甲烷传感器数据...")
dfs = {}
for sensor in methane_sensors:
    db_path = os.path.join(db_dir, f'{sensor}.db')
    if not os.path.exists(db_path):
        print(f"警告: {sensor}.db 不存在，跳过")
        continue

    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )
    dfs_list = []
    for _, row in tables.iterrows():
        table_name = row['name']
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        dfs_list.append(df)
    conn.close()

    df_all = pd.concat(dfs_list, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date').reset_index(drop=True)
    ts = df_all.set_index('date')['value'].astype(float)
    ts = ts[~ts.index.duplicated(keep='first')]
    ts.name = sensor
    dfs[sensor] = ts
    print(f"  {sensor}: {len(ts)} 条记录")

# ==================== 2. 1分钟重采样（中值） ====================
print("\n正在进行1分钟重采样...")
dfs_resampled = {}
for sensor in methane_sensors:
    dfs_resampled[sensor] = dfs[sensor].resample('1min').median()

# ==================== 3. 时间对齐 ====================
print("正在进行时间对齐...")
common_start = max(ts.index.min() for ts in dfs_resampled.values())
common_end = min(ts.index.max() for ts in dfs_resampled.values())
time_grid = pd.date_range(common_start, common_end, freq='1min')

df_aligned = pd.DataFrame(index=time_grid)
for sensor in methane_sensors:
    df_aligned[sensor] = dfs_resampled[sensor].reindex(time_grid)

print(f"对齐后数据量: {len(df_aligned)} 行")
print(">>> 第1段完成，请运行第2段 <<<")
"""
21512采面数据级融合与对比图生成（第2段）
功能：缺失值填补、滑动平均滤波、加权融合、保存结果
前提：已运行第1段
"""

import pandas as pd
import numpy as np
import os

fused_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\data_fused.xlsx'

# ==================== 4. 缺失值填补 ====================
print("\n正在进行缺失值填补...")
for sensor in methane_sensors:
    if sensor not in df_aligned.columns:
        continue
    missing_count = df_aligned[sensor].isna().sum()
    if missing_count > 0:
        df_aligned[sensor] = df_aligned[sensor].interpolate(method='linear', limit=5)
        still_missing = df_aligned[sensor].isna().sum()
        print(f"  {sensor}: 填补前{missing_count}点, 填补后仍缺失{still_missing}点")
    else:
        print(f"  {sensor}: 无缺失")

# ==================== 5. 滑动平均滤波（窗口=7） ====================
print("\n正在进行滑动平均滤波（窗口=7）...")
for sensor in methane_sensors:
    if sensor in df_aligned.columns:
        df_aligned[sensor] = df_aligned[sensor].rolling(window=7, min_periods=1).mean()

# ==================== 6. 加权平均融合 ====================
print("\n正在进行加权平均融合...")
df_fused = pd.DataFrame(index=df_aligned.index)
df_fused['CH4_fused'] = 0.0

for sensor in methane_sensors:
    w = weights.get(sensor, 0)
    if w > 0 and sensor in df_aligned.columns:
        valid_mask = df_aligned[sensor].notna()
        df_fused.loc[valid_mask, 'CH4_fused'] += w * df_aligned.loc[valid_mask, sensor]
        print(f"  {sensor}: 权重={w}, 有效数据点={valid_mask.sum()}")

# 缺失时刻重新归一化权重
for t in df_fused.index:
    valid_sensors = [s for s in methane_sensors 
                     if s in df_aligned.columns and pd.notna(df_aligned.loc[t, s])]
    if len(valid_sensors) < 5:
        sum_w = sum(weights.get(s, 0) for s in valid_sensors)
        if sum_w > 0:
            df_fused.loc[t, 'CH4_fused'] /= sum_w

# ==================== 7. 保存融合结果 ====================
df_output = df_fused.reset_index()
df_output.rename(columns={'index': 'timestamp'}, inplace=True)
df_output.to_excel(fused_path, index=False, engine='openpyxl')

print(f"\n数据级融合完成！")
print(f"输出文件: {fused_path}")
print(f"融合序列长度: {len(df_output)}")
print(f"浓度范围: {df_output['CH4_fused'].min():.4f}% ~ {df_output['CH4_fused'].max():.4f}%")
print(">>> 第2段完成，请运行第3段 <<<")
"""
21512采面数据级融合与对比图生成（第3段）
功能：绘制融合前后对比图，图片保存到桌面
前提：已运行第1、2段
"""

import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = r'C:\Users\陈泽秋\Desktop'

# ==================== 8. 准备绘图数据 ====================
print("\n正在准备绘图数据...")

aligned = pd.DataFrame(index=df_fused.index)
aligned['fused'] = df_fused['CH4_fused']
aligned['38A01'] = df_aligned['38A01']

aligned_valid = aligned.dropna()
n_points = min(2000, len(aligned_valid))
plot_data = aligned_valid.iloc[:n_points]
print(f"绘图数据: {n_points} 个点")

# ==================== 9. 绘图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

x_labels = [plot_data.index[i].strftime('%m-%d\n%H:%M') for i in range(0, n_points, 200)]

# 左图：整体对比
ax1.plot(plot_data.index, plot_data['38A01'], '-', color='steelblue',
         linewidth=1.5, alpha=0.7, label='38A01 原始单传感器')
ax1.plot(plot_data.index, plot_data['fused'], '-', color='darkred',
         linewidth=2.0, label='融合瓦斯浓度')
ax1.set_xlabel('时间', fontsize=14)
ax1.set_ylabel('瓦斯浓度 CH₄（%）', fontsize=14)
ax1.tick_params(labelsize=12)
ax1.set_xticks(plot_data.index[::200])
ax1.set_xticklabels(x_labels, rotation=30, fontsize=11)
ax1.legend(fontsize=13, loc='upper right')
ax1.set_title('(a) 融合前后整体对比', fontsize=15, fontweight='bold', y=-0.25)

# 右图：局部放大
zoom_start = 500
zoom_end = min(800, n_points)
plot_zoom = plot_data.iloc[zoom_start:zoom_end]
zoom_labels = [plot_zoom.index[i].strftime('%H:%M') for i in range(0, len(plot_zoom), 50)]

ax2.plot(plot_zoom.index, plot_zoom['38A01'], 'o-', color='steelblue',
         linewidth=1.8, markersize=4, markerfacecolor='white', label='38A01')
ax2.plot(plot_zoom.index, plot_zoom['fused'], 's-', color='darkred',
         linewidth=2.0, markersize=4, markerfacecolor='white', label='融合值')
ax2.set_xlabel('时间', fontsize=14)
ax2.set_ylabel('瓦斯浓度 CH₄（%）', fontsize=14)
ax2.tick_params(labelsize=12)
ax2.set_xticks(plot_zoom.index[::50])
ax2.set_xticklabels(zoom_labels, rotation=30, fontsize=11)
ax2.legend(fontsize=13, loc='upper right')
ax2.set_title('(b) 局部放大（细节对比）', fontsize=15, fontweight='bold', y=-0.25)

plt.tight_layout()
output_png = os.path.join(output_dir, '图4.X_数据级融合对比.png')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"\n图片已保存至: {output_png}")
plt.show()

