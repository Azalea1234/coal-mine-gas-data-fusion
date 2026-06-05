"""
21512采面多源数据预处理脚本
功能:从SQLite读取数据、时间对齐统一重采样
"""

import sqlite3
import pandas as pd
import numpy as np
import os

# ==================== 路径设置 ====================
db_dir = r'C:\Users\陈泽秋\Desktop\SQlite_db\cqu_SQLite'
output_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\processed_data.xlsx'

if os.path.exists(output_path):
    backup_path = output_path.replace('.xlsx', '_backup.xlsx')
    os.rename(output_path, backup_path)
    print(f"原文件已备份至: {backup_path}")

# ==================== 传感器列表 ====================
sensors = [
    '38A03', '38A01', '39A13', '34A01', '40A05',
    '39A16', '38A09', '39A15', '39A14', '38A04',
    '35A12', '35A09', '35A11', '35A10',
    '39A02', '39A01', '39A04', '39A03',
    '39A07', '38A02', '40D14'
]

# ==================== 1. 从SQLite读取数据 ====================
print("正在从SQLite数据库读取传感器数据...")
dfs_original = {}

for sensor in sensors:
    db_path = os.path.join(db_dir, f'{sensor}.db')
    if not os.path.exists(db_path):
        print(f"警告: {sensor}.db 不存在，跳过")
        continue

    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )
    if tables.empty:
        print(f"警告: {sensor}.db 中无数据表，跳过")
        conn.close()
        continue

    table_name = tables.iloc[0, 0]
    df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
    conn.close()

    time_col = 'date'
    value_col = 'value'

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    ts = df.set_index(time_col)[value_col].astype(float)
    ts = ts[~ts.index.duplicated(keep='first')]
    ts.name = sensor

    dfs_original[sensor] = ts
    print(f"  {sensor}: {len(ts)} 条记录, 时间范围 {ts.index.min()} ~ {ts.index.max()}")

# ==================== 2. 独立重采样后再时间对齐 ====================
print("\n正在进行1分钟重采样并统一对齐...")

target_freq = '1min'  # 重采样到1分钟
dfs_resampled = {}

for sensor, ts in dfs_original.items():
    # 使用中位数重采样，如果1分钟内有多个值，取中值
    ts_resampled = ts.resample(target_freq).median()
    ts_resampled.name = sensor
    dfs_resampled[sensor] = ts_resampled

# 取所有重采样后数据的交集时间范围
all_starts = [ts.index.min() for ts in dfs_resampled.values()]
all_ends = [ts.index.max() for ts in dfs_resampled.values()]
common_start = max(all_starts)
common_end = min(all_ends)

time_grid = pd.date_range(common_start, common_end, freq=target_freq)
df_wide = pd.DataFrame({'timestamp': time_grid})
df_wide = df_wide.set_index('timestamp')

for sensor, ts in dfs_resampled.items():
    df_wide[sensor] = ts.reindex(time_grid, method=None)
    missing_count = df_wide[sensor].isna().sum()
    total_count = len(df_wide)
    print(f"  {sensor}: 缺失 {missing_count}/{total_count} "
          f"({missing_count/total_count*100:.2f}%)")

print(f"\n重采样后时间网格: {len(time_grid)} 个点, "
      f"时间范围 {common_start} ~ {common_end}")

import pandas as pd
import numpy as np

# ==================== 传感器分类 ====================
slow_params = [
    '38A03', '38A01', '39A13', '34A01', '40A05',
    '39A15', '35A12', '39A04',
    '39A16', '38A09', '38A04',
    '39A07'
]

pipe_params = [
    '35A09', '39A01', '35A10', '39A02', '35A11', '39A03'
]

step_params = ['39A14', '40D14', '38A02']

# ==================== 3. 缺失值填补 ====================
print("\n正在进行缺失值填补...")

missing_markers = {}

for sensor in sensors:
    if sensor not in df_wide.columns:
        continue
    
    series = df_wide[sensor].copy()
    missing_mask = series.isna()
    
    missing_groups = (missing_mask != missing_mask.shift()).cumsum()
    missing_lengths = missing_mask.groupby(missing_groups).transform('sum')
    
    short_missing = missing_mask & (missing_lengths <= 5)
    if short_missing.any():
        series = series.interpolate(method='linear', limit=5)
        print(f"  {sensor}: 短时缺失填补 {short_missing.sum()} 个点")
    
    long_missing = missing_mask & (missing_lengths > 5)
    if long_missing.any():
        print(f"  {sensor}: 长时缺失 {long_missing.sum()} 个点，保留标记")
    
    missing_markers[sensor] = {
        'total': missing_mask.sum(),
        'short_filled': short_missing.sum() if short_missing.any() else 0,
        'long_removed': long_missing.sum() if long_missing.any() else 0
    }
    
    df_wide[sensor] = series

# ==================== 4. 分类滑动平均滤波 ====================
print("\n正在进行分类滑动平均滤波...")

for sensor in sensors:
    if sensor not in df_wide.columns:
        continue
    
    if sensor in slow_params:
        window = 7
        df_wide[sensor] = df_wide[sensor].rolling(window=window, min_periods=1).mean()
        print(f"  {sensor}: 滑动平均（窗口={window}）")
    elif sensor in pipe_params:
        window = 3
        df_wide[sensor] = df_wide[sensor].rolling(window=window, min_periods=1).mean()
        print(f"  {sensor}: 滑动平均（窗口={window}）")
    elif sensor in step_params:
        print(f"  {sensor}: 不滤波（保留阶跃特征）")


import pandas as pd
import numpy as np
import os

output_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\processed_data.xlsx'

# ==================== 5. Z-score标准化 ====================
print("\n正在进行Z-score标准化...")

for sensor in sensors:
    if sensor not in df_wide.columns:
        continue
    
    series = df_wide[sensor].copy()
    valid_mask = ~series.isna()
    
    if valid_mask.sum() > 0:
        valid_data = series[valid_mask]
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        
        if std_val == 0 or pd.isna(std_val):
            std_val = 1.0
            print(f"  {sensor}: 标准差为0，跳过除标准差操作")
        
        df_wide[sensor] = (series - mean_val) / std_val
        print(f"  {sensor}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
    else:
        print(f"  {sensor}: 无有效数据，跳过标准化")

# ==================== 6. 保存 ====================
print(f"\n正在保存处理后的数据至: {output_path}")

df_output = df_wide.reset_index()
df_output.rename(columns={'index': 'timestamp'}, inplace=True)
df_output.to_excel(output_path, index=False, engine='openpyxl')

print(f"\n数据预处理完成！")
print(f"输出文件: {output_path}")
print(f"数据形状: {df_output.shape[0]} 行 × {df_output.shape[1]} 列")
print(f"时间范围: {df_output['timestamp'].min()} ~ {df_output['timestamp'].max()}")

# ==================== 7. 预处理报告 ====================
print("\n" + "="*60)
print("预处理报告")
print("="*60)

total_rows = len(df_output)
print(f"总时间点数: {total_rows}")
print(f"传感器数量: {len(sensors)}")

print(f"\n缺失值处理:")
for sensor in sensors:
    if sensor in missing_markers:
        m = missing_markers[sensor]
        print(f"  {sensor}: 总缺失{m['total']}点, "
              f"短时填补{m['short_filled']}点, "
              f"长时保留{m['long_removed']}点")

print(f"\n滤波策略:")
print(f"  缓变参数({len(slow_params)}个): 滑动平均窗口N=7")
print(f"  管路波动参数({len(pipe_params)}个): 滑动平均窗口N=3")
print(f"  含阶跃参数({len(step_params)}个): 不滤波")

print(f"\n标准化方法: Z-score (均值为0, 标准差为1)")
print("\n预处理脚本运行完毕。")
