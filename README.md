# coal-mine-gas-data-fusion
Code for undergraduate thesis:Research on Characteristics and Fusion Methods of Multi-source Heterogeneous Data for Coal Mine  Disasters

本仓库为本科毕业论文《煤矿瓦斯灾害多源异构数据特征及融合方法研究》的配套 Python 代码。

## 代码结构

| 文件夹 | 内容说明 |
|:---|:---|
| 1_data_pretreatment/ | 数据预处理：缺失值填补、分类滑动平均滤波、Z-score 标准化、巡检文本结构化提取、地质特征生成 |
| 2_data_level_fusion/ | 数据级融合：5 台激光甲烷传感器加权平均融合 |
| 3_feature_level_fusion/ | 特征级融合：核主成分分析（KPCA）降维 |
| 4_submodels/ | 子模型训练与评估：LSTM 时序预警、SVM 动力灾害预警、模糊推理地质增强预警 |
| 5_decision_fusion/ | 决策级融合：贝叶斯估计综合三级子模型输出，生成预警等级 |

## 运行环境

- Python 3.12
- 依赖库：pandas, numpy, torch, scikit-learn, matplotlib, openpyxl, joblib, scipy, python-docx
