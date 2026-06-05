"""
巡检文本结构化提取脚本
功能：读取巡检记录docx文件 → 词典匹配/正则提取 → 前向填充 → 保存特征表
优化点：排除阴性描述、改进底鼓量正则、增加上下文判断
输入：21512采面巡检文本记录.docx
输出：patrol_features.xlsx
"""

import pandas as pd
import numpy as np
import re
import os
from docx import Document

# ==================== 路径设置 ====================
input_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\cqu_SQLite\21512采面巡检文本记录.docx'
output_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\patrol_features.xlsx'

# ==================== 异常特征词典（含负向词排除） ====================
# 声响类
sound_keywords = ['煤炮', '闷雷声', '噼啪声', '岩爆声', '响煤炮']
sound_negatives = ['无煤炮', '未听到', '没有煤炮', '未发现煤炮']

# 视觉类 — 挂汗
sw_keywords = ['挂汗']
sw_negatives = ['无挂汗', '没有挂汗', '未见挂汗', '挂汗消失', '挂汗基本消失', '挂汗面积缩小']

# 视觉类 — 片帮
sp_keywords = ['片帮', '煤壁脱落', '小块脱落', '煤壁破碎', '裂隙']
sp_negatives = ['无片帮', '没有片帮', '未见片帮', '无新增片帮']

# 视觉类 — 底鼓
hf_keywords = ['底鼓', '鼓起']
hf_negatives = ['无底鼓', '没有底鼓', '未见底鼓', '无新增鼓起', '无异常鼓起', '底鼓量监测正常']

# 视觉类 — 瓦斯气味
od_keywords = ['瓦斯气味', '异味', '有气味']
od_negatives = ['无瓦斯气味', '没有异味', '无异味', '无异常气味']

# 重点关注
attention_keywords = ['重点关注', '加密巡检', '加密监测', '加强监测', '密切关注', '警戒', '通知调度室', '上报']

# ==================== 辅助函数 ====================
def has_negative(text, keyword, negatives):
    """检查文本中关键词附近是否有否定词"""
    for neg in negatives:
        # 检查否定词是否在关键词前后20个字符内
        idx = text.find(keyword)
        if idx >= 0:
            context_start = max(0, idx - 20)
            context_end = min(len(text), idx + len(keyword) + 20)
            context = text[context_start:context_end]
            if neg in context:
                return True
    return False

def is_genuine_abnormal(text, keywords, negatives):
    """判断是否为真正的异常（非阴性描述）"""
    for kw in keywords:
        if kw in text and not has_negative(text, kw, negatives):
            return True
    return False

# ==================== 解析单条记录 ====================
def parse_record(text, record_date):
    features = {
        'timestamp': record_date,
        'I_sw': 0, 'I_sp': 0, 'I_hf': 0, 'I_od': 0,
        'f_cb': 0.0, 'd_hf': 0.0, 'I_at': 0
    }
    
    # --- 声响类（煤炮） ---
    for kw in sound_keywords:
        if kw in text:
            # 排除阴性描述
            neg = False
            for nkw in sound_negatives:
                if nkw in text:
                    neg = True
                    break
            if neg:
                break
            
            # 提取频次：匹配 "约X次/Y分钟" 或 "X次/10分钟" 或 "X次"
            freq_match = re.search(r'(?:约\s*)?(\d+)\s*次\s*/\s*(?:每\s*)?(\d+)\s*(?:分钟|min)', text)
            if freq_match:
                count = int(freq_match.group(1))
                minutes = int(freq_match.group(2))
                features['f_cb'] = round(count / minutes * 10, 1)
            else:
                # 匹配单独的 "约X次" 或 "X次"
                single_match = re.search(r'(?:约\s*)?(\d+)\s*次', text)
                if single_match:
                    features['f_cb'] = float(single_match.group(1))
                elif any(w in text for w in ['频繁', '增多', '增加', '持续']):
                    features['f_cb'] = 4.0
                elif any(w in text for w in ['偶有', '轻微', '偶而']):
                    features['f_cb'] = 1.0
                elif any(w in text for w in ['减少', '减弱', '变弱', '变小']):
                    features['f_cb'] = 1.0  # 频次下降但仍存在
            break
    
    # --- 挂汗 ---
    if is_genuine_abnormal(text, sw_keywords, sw_negatives):
        features['I_sw'] = 1
    
    # --- 片帮 ---
    if is_genuine_abnormal(text, sp_keywords, sp_negatives):
        features['I_sp'] = 1
    
    # --- 底鼓 ---
    if is_genuine_abnormal(text, hf_keywords, hf_negatives):
        features['I_hf'] = 1
        # 提取底鼓量（支持多种表述）
        hf_patterns = [
            r'(?:底鼓|鼓起)\s*(?:量\s*)?(?:约|大约|达)?\s*(\d+(?:\.\d+)?)\s*cm',
            r'鼓起\s*(?:约|大约)?\s*(\d+(?:\.\d+)?)\s*cm',
            r'底鼓\s*(?:约|大约)?\s*(\d+(?:\.\d+)?)\s*cm',
            r'(?:底板|底鼓量)\s*(?:鼓起|鼓起量)?\s*(?:约|大约)?\s*(\d+(?:\.\d+)?)\s*cm',
        ]
        for pattern in hf_patterns:
            match = re.search(pattern, text)
            if match:
                features['d_hf'] = float(match.group(1))
                break
    
    # --- 瓦斯气味 ---
    if is_genuine_abnormal(text, od_keywords, od_negatives):
        features['I_od'] = 1
    
    # --- 重点关注 ---
    for kw in attention_keywords:
        if kw in text:
            features['I_at'] = 1
            break
    
    return features

# ==================== 读取并解析docx ====================
print("正在读取巡检记录文本...")
doc = Document(input_path)

records = []
current_lines = []
in_record = False

for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue
    
    # 判断是否是新记录的起始行
    if re.search(r'记录\s*\d+', text) and '巡查人' in text:
        if in_record and current_lines:
            full_text = ' '.join(current_lines)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})', current_lines[0])
            if date_match:
                record_date = pd.to_datetime(f"{date_match.group(1)} {date_match.group(2)}")
                features = parse_record(full_text, record_date)
                records.append(features)
        current_lines = [text]
        in_record = True
    elif in_record:
        if text.startswith('===') or text.startswith('###') or text.startswith('数据使用'):
            if current_lines:
                full_text = ' '.join(current_lines)
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})', current_lines[0])
                if date_match:
                    record_date = pd.to_datetime(f"{date_match.group(1)} {date_match.group(2)}")
                    features = parse_record(full_text, record_date)
                    records.append(features)
                current_lines = []
            in_record = False
        else:
            current_lines.append(text)

# 处理最后一条记录
if in_record and current_lines:
    full_text = ' '.join(current_lines)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})', current_lines[0])
    if date_match:
        record_date = pd.to_datetime(f"{date_match.group(1)} {date_match.group(2)}")
        features = parse_record(full_text, record_date)
        records.append(features)

print(f"成功解析 {len(records)} 条巡检记录")

# ==================== 创建特征DataFrame ====================
df_records = pd.DataFrame(records)
df_records = df_records.sort_values('timestamp').reset_index(drop=True)

# ==================== 前向填充 ====================
print("正在进行前向填充...")
time_start = df_records['timestamp'].min().floor('D')
time_end = df_records['timestamp'].max().ceil('D')
time_grid = pd.date_range(time_start, time_end, freq='1min')

df_full = pd.DataFrame({'timestamp': time_grid})
for col in ['I_sw', 'I_sp', 'I_hf', 'I_od', 'f_cb', 'd_hf', 'I_at']:
    df_full[col] = 0.0

for i in range(len(df_records)):
    row = df_records.iloc[i]
    mask = df_full['timestamp'] >= row['timestamp']
    if i < len(df_records) - 1:
        next_time = df_records.iloc[i + 1]['timestamp']
        mask &= df_full['timestamp'] < next_time
    for col in ['I_sw', 'I_sp', 'I_hf', 'I_od', 'f_cb', 'd_hf', 'I_at']:
        df_full.loc[mask, col] = row[col]

# ==================== 保存结果 ====================
df_full.to_excel(output_path, index=False, engine='openpyxl')
print(f"\n特征表已保存至: {output_path}")
print(f"时间范围: {df_full['timestamp'].min()} ~ {df_full['timestamp'].max()}")
print(f"数据行数: {len(df_full)}")

# ==================== 统计报告 ====================
print("\n" + "="*60)
print("巡检特征统计报告")
print("="*60)
print(f"总巡检记录数: {len(df_records)}")
abnormal_count = (df_records[['I_sw','I_sp','I_hf','I_od','f_cb','I_at']].sum(axis=1) > 0).sum()
print(f"含异常记录数: {abnormal_count}")

for col, name in zip(
    ['I_sw', 'I_sp', 'I_hf', 'I_od', 'I_at'],
    ['挂汗', '片帮', '底鼓', '瓦斯气味', '重点关注']
):
    count = df_records[col].sum()
    print(f"  {name}: 出现 {int(count)} 次")

if df_records['f_cb'].max() > 0:
    cb = df_records[df_records['f_cb'] > 0]
    print(f"  煤炮声: 出现 {len(cb)} 次, 频次 {cb['f_cb'].min():.1f} ~ {cb['f_cb'].max():.1f} 次/10min")

if df_records['d_hf'].max() > 0:
    hf = df_records[df_records['d_hf'] > 0]
    print(f"  底鼓量: 出现 {len(hf)} 次, 范围 {hf['d_hf'].min():.1f} ~ {hf['d_hf'].max():.1f} cm")

print("\n脚本运行完毕。")
