"""
地质图探查与解析脚本
功能：
  1. 读取瓦斯地质图.dxf文件，遍历图层，统计各类实体数量
  2. 提取断层线、瓦斯含量等值线、突出危险区多边形及文本标注
  3. 提取图纸中的控制点坐标，计算仿射变换参数
  4. 为后续地质特征生成提供基础数据
"""

import ezdxf
import numpy as np
import os

# ==================== 路径设置 ====================
dxf_path = r'C:\Users\陈泽秋\Desktop\SQlite_db\cqu_SQLite\瓦斯地质图.dxf'

# ==================== 第一部分：图层探查与实体统计 ====================
print("=" * 60)
print("第一部分：图层探查与实体统计")
print("=" * 60)

# 读取DXF文件
doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()

# 统计各类实体数量
entity_count = {
    'LINE': 0,
    'LWPOLYLINE': 0,
    'POLYLINE': 0,
    'SPLINE': 0,
    'TEXT': 0,
    'MTEXT': 0,
    'INSERT': 0,
    'CIRCLE': 0,
    'ARC': 0,
    'POINT': 0
}

for entity in msp:
    entity_count[entity.dxftype()] = entity_count.get(entity.dxftype(), 0) + 1

print("图纸实体统计：")
for etype, count in entity_count.items():
    if count > 0:
        print(f"  {etype}: {count} 个")

# 遍历图层
print("\n图层信息：")
layers = doc.layers
layer_info = {}
for layer in layers:
    layer_name = layer.dxf.name
    # 统计该图层中的实体数量
    count = len([e for e in msp if e.dxf.layer == layer_name])
    if count > 0:
        layer_info[layer_name] = count
        print(f"  图层 '{layer_name}': {count} 个实体")

print(f"\n共有 {len(layer_info)} 个有效图层")

# ==================== 第二部分：关键地质要素提取 ====================
print("\n" + "=" * 60)
print("第二部分：关键地质要素提取")
print("=" * 60)

# 断层线：从"3煤断层"图层提取（颜色1，红色）
fault_lines = []
fault_layer_name = "3煤断层"
for entity in msp:
    if entity.dxf.layer == fault_layer_name:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            fault_lines.append({
                'type': 'LINE',
                'start': (start.x, start.y),
                'end': (end.x, end.y)
            })
        elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            points = list(entity.vertices())
            fault_lines.append({
                'type': entity.dxftype(),
                'vertices': [(p.x, p.y) for p in points]
            })

print(f"提取断层线: {len(fault_lines)} 条")

# 瓦斯含量等值线：从"3煤瓦斯含量等值线"图层提取
contour_lines = []
contour_layer_name = "3煤瓦斯含量等值线"
for entity in msp:
    if entity.dxf.layer == contour_layer_name:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            contour_lines.append({
                'type': 'LINE',
                'start': (start.x, start.y),
                'end': (end.x, end.y)
            })
        elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
            points = list(entity.vertices())
            contour_lines.append({
                'type': entity.dxftype(),
                'vertices': [(p.x, p.y) for p in points]
            })

print(f"提取等值线: {len(contour_lines)} 条")

# 瓦斯含量文本标注
gas_texts = []
for entity in msp:
    if entity.dxf.layer == contour_layer_name and entity.dxftype() == 'TEXT':
        text_content = entity.dxf.text
        insert_point = (entity.dxf.insert.x, entity.dxf.insert.y)
        gas_texts.append({
            'text': text_content,
            'position': insert_point
        })

print(f"提取瓦斯含量文本标注: {len(gas_texts)} 个")

# 提取文本中的数值
gas_values = []
for item in gas_texts:
    try:
        value = float(item['text'].replace(' ', ''))
        gas_values.append(value)
    except ValueError:
        pass

if gas_values:
    print(f"  等值线标注值范围: {min(gas_values):.1f} ~ {max(gas_values):.1f} m³/t")
    print(f"  等值线标注均值: {np.mean(gas_values):.2f} m³/t")

# 突出危险区：从"突出危险区"图层提取
danger_zones = []
danger_layer_name = "突出危险区"
for entity in msp:
    if entity.dxf.layer == danger_layer_name:
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE', 'SPLINE'):
            points = list(entity.vertices())
            if len(points) >= 3:  # 至少三个点才能构成多边形
                danger_zones.append({
                    'type': entity.dxftype(),
                    'vertices': [(p.x, p.y) for p in points]
                })

print(f"提取突出危险区多边形: {len(danger_zones)} 个")

# ==================== 第三部分：控制点提取与坐标转换 ====================
print("\n" + "=" * 60)
print("第三部分：控制点提取与坐标转换")
print("=" * 60)

# 从"！！！3煤巷道"图层提取控制点
control_layer_name = "！！！3煤巷道"
control_points = []

# 已知矿井坐标的控制点（根据矿方资料）
# 点1：图面坐标(9052.30, 5019.36)，矿井坐标 X=3958008.000
# 点2：图面坐标(9164.27, 5020.62)，矿井坐标 X=3958029.000
known_control_points = [
    {'drawing_x': 9052.30, 'drawing_y': 5019.36, 'mine_x': 3958008.000, 'mine_y': None},
    {'drawing_x': 9164.27, 'drawing_y': 5020.62, 'mine_x': 3958029.000, 'mine_y': None}
]

print(f"已知矿井坐标控制点: {len(known_control_points)} 个")

# 在巷道图层中搜索这些坐标附近的TEXT实体，获取完整矿井坐标
for entity in msp:
    if entity.dxf.layer == control_layer_name and entity.dxftype() == 'TEXT':
        text_content = entity.dxf.text
        insert_point = (entity.dxf.insert.x, entity.dxf.insert.y)
        control_points.append({
            'text': text_content,
            'position': insert_point
        })

print(f"巷道图层文本标注: {len(control_points)} 个")

# 计算仿射变换参数
# 由于图纸中缺少Y方向矿井坐标，此处基于已知信息进行推算
# 实际应用中应补充完整的控制点坐标
print("\n仿射变换参数计算：")
print("  控制点1: 图面(9052.30, 5019.36) → 矿井 X=3958008.000")
print("  控制点2: 图面(9164.27, 5020.62) → 矿井 X=3958029.000")

dx = 9164.27 - 9052.30  # 图面X差值
dX = 3958029.000 - 3958008.000  # 矿井X差值
scale_x = dX / dx  # X方向比例尺

print(f"\n  X方向比例尺: {scale_x:.6f} (矿井米/图面单位)")
print(f"  注：图纸坐标单位为毫米，矿井坐标单位为米")
print(f"  实际比例约为 1:{1000/scale_x:.0f}")

# 保存提取结果供后续使用
print("\n" + "=" * 60)
print("探查与解析完成。提取结果将用于后续地质特征生成。")
print("=" * 60)

# 输出关键统计信息
print(f"\n汇总:")
print(f"  图层数量: {len(layer_info)}")
print(f"  断层线: {len(fault_lines)} 条")
print(f"  等值线: {len(contour_lines)} 条")
print(f"  瓦斯含量标注: {len(gas_texts)} 个")
print(f"  突出危险区: {len(danger_zones)} 个")
print(f"  控制点: {len(known_control_points)} 个")
