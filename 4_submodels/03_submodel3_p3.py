# ==================== 1. 读取地质特征 ====================
print("正在读取地质特征...")
df_geo = pd.read_csv(geology_path)

# 实际列名：sensor_id, sensor_name, x_coord, y_coord, d_fault_m, r_gas_m3t, i_danger
# 选取代表传感器（工作面38A01）的地质特征
sensor_ref = '38A01'
ref_row = df_geo[df_geo['sensor_id'].astype(str) == sensor_ref]
if ref_row.empty:
    print(f"警告：地质特征表中未找到{sensor_ref}，使用第一行数据")
    ref_row = df_geo.iloc[[0]]

d_fault = ref_row['d_fault_m'].values[0]     # 断层距离 (m)
r_gas = ref_row['r_gas_m3t'].values[0]       # 瓦斯含量 (m³/t)
I_danger = ref_row['i_danger'].values[0]     # 突出危险区标志

print(f"代表传感器: {sensor_ref}, d_fault={d_fault}m, r_gas={r_gas}m³/t, I_danger={I_danger}")

# ==================== 2. 读取动态数据 ====================
print("正在读取融合瓦斯浓度...")
df_fused = pd.read_excel(fused_path)
df_fused['timestamp'] = pd.to_datetime(df_fused['timestamp'])
df_fused = df_fused.set_index('timestamp')
ch4_series = df_fused['CH4_fused']

print("正在读取巡检特征...")
df_patrol = pd.read_excel(patrol_path)
df_patrol['timestamp'] = pd.to_datetime(df_patrol['timestamp'])
df_patrol = df_patrol.set_index('timestamp')

# 取公共时间范围
common_start = max(ch4_series.index.min(), df_patrol.index.min())
common_end = min(ch4_series.index.max(), df_patrol.index.max())
time_grid = pd.date_range(common_start, common_end, freq='1min')

ch4_aligned = ch4_series.reindex(time_grid).interpolate(method='linear', limit=5)
patrol_aligned = df_patrol.reindex(time_grid).fillna(0.0)

# ==================== 3. 模糊推理函数 ====================
def compute_p3(ch4, f_cb, I_sw, I_od, d_fault, r_gas, I_danger):
    """计算地质风险增强概率 P3"""
    
    # ---------- 基础概率（基于地质静态信息 + 当前瓦斯浓度） ----------
    if d_fault < 50 and r_gas > 8:
        p3_base = 0.8
    elif d_fault > 200 and I_danger == 0 and ch4 < 0.5:
        p3_base = 0.1
    else:
        # 线性插值
        risk_distance = max(0, 1 - d_fault / 250)
        risk_gas = min(r_gas / 12, 1.0)
        risk_ch4 = min(ch4 / 1.0, 1.0)
        p3_base = 0.2 + 0.6 * (risk_distance * 0.3 + risk_gas * 0.4 + risk_ch4 * 0.3)
        p3_base = np.clip(p3_base, 0.1, 0.8)

    # ---------- 巡检特征修正 ----------
    p3 = p3_base
    
    # 煤炮频次高且出现挂汗，大幅上调
    if f_cb > 3 and I_sw == 1:
        p3 += 0.2
    # 煤炮频次较高或挂汗单独出现，适度上调
    elif f_cb > 2 or I_sw == 1:
        p3 += 0.1
    
    # 有瓦斯气味，额外上调
    if I_od == 1:
        p3 += 0.05

    return np.clip(p3, 0.0, 1.0)

# ==================== 4. 逐时刻计算P3 ====================
print("正在计算P3...")
p3_values = []

for t in time_grid:
    ch4 = ch4_aligned.loc[t]
    f_cb = patrol_aligned.loc[t, 'f_cb'] if 'f_cb' in patrol_aligned.columns else 0.0
    I_sw = patrol_aligned.loc[t, 'I_sw'] if 'I_sw' in patrol_aligned.columns else 0
    I_od = patrol_aligned.loc[t, 'I_od'] if 'I_od' in patrol_aligned.columns else 0
    
    p3 = compute_p3(ch4, f_cb, I_sw, I_od, d_fault, r_gas, I_danger)
    p3_values.append(p3)

df_p3 = pd.DataFrame({'timestamp': time_grid, 'P3': p3_values})
df_p3.set_index('timestamp', inplace=True)
