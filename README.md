# 基于机器学习的个人信用风险评估与逾期预测系统
## 项目简介
本项目基于Kaggle公开信贷数据集GiveMeSomeCredit，搭建完整信贷风控建模全流程，实现数据清洗、特征工程、多模型对比训练、网格搜索超参优化、消融实验、AUC/KS指标评估，配套Streamlit交互式Web预测平台，可输入用户原始特征实时判断信贷逾期风险等级。
项目覆盖5类主流机器学习模型：逻辑回归、决策树、随机森林、XGBoost、LightGBM，采用5折分层交叉验证保证实验结果稳定可复现。

## 环境依赖
Python >= 3.9
一键安装全部依赖包：
pip install -r requirements.txt

## 项目目录结构
Credit-Risk-Assessment-System/</br>
├── data/                # 数据集存放目录（数据集自行前往Kaggle下载）</br>
├── models/              # 训练输出：指标表格、ROC曲线图、实验结果图 </br>
├── src/</br>
│   ├── preprocessing.py # 数据清洗、异常处理、特征衍生、缺失值填充工具</br>
│   ├── train.py         # 多模型训练、网格搜索调参、消融实验、指标计算</br>
│   └── app.py           # Streamlit可视化预测前端页面</br>
├── requirements.txt     # 项目依赖清单</br>
└── README.md            # 项目说明文档</br>

## 数据集说明
数据集名称：GiveMeSomeCredit</br>
下载地址：Kaggle 平台公开数据集</br>
包含10项用户原始信贷特征，用于二分类逾期预测任务；</br>
项目代码内置完整特征衍生逻辑，自动生成逾期频次、人均收入、对数负债比等衍生特征。
## 运行步骤
### 1. 数据准备
将下载后的 cs-training.csv、cs-test.csv 放入 data 文件夹
### 2. 执行模型训练
python src/train.py
运行完成后，models 文件夹自动生成多模型指标对比表、ROC 曲线图、消融实验结果。
### 3. 启动 Web 预测平台
streamlit run src/app.py
## 页面功能：
1.原始数据分布、特征相关性可视化展示</br>
2.多模型 AUC、KS、混淆矩阵指标对比</br>
3.单用户手动输入特征，实时输出逾期概率与高 / 中 / 低风险分级
## 模型与实验说明
### 防过拟合策略
所有树模型通过限制 max_depth、min_samples_leaf/min_child_samples 从结构约束抑制过拟合；XGBoost、LightGBM 额外搭配 L1、L2 正则，形成多层防过拟合机制。
### 评价指标
AUC：全局样本区分能力</br>
KS：风控核心指标，衡量好坏客户最大区分差值，公式：
KS=max(TPR−FPR)
### 实验结论
集成树模型（XGBoost、LightGBM、随机森林）ROC 曲线、AUC 数值差距较小，原因：数据集风险区分能力存在固定上限，集成树拟合非线性规律能力相近，全部经过网格搜索最优调参；综合精度与推理速度，LightGBM 为本项目最优模型。
## 核心创新点
1.训练与推理复用同一套预处理逻辑，规避线上线下数据分布不一致、数据泄露问题；</br>
2.消融实验验证各特征工程模块对模型性能的提升效果；</br>
3.轻量化 Web 可视化平台，无需部署服务即可本地完成风险预测演示，适配毕业设计答辩展示。
