import gc
import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 基础绘图设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  #黑体
plt.rcParams["axes.unicode_minus"] = False    #显示负号

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    warnings.warn("未安装xgboost，无法使用XGBClassifier模型")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    warnings.warn("未安装lightgbm，无法使用LGBMClassifier模型")

from preprocessing import (
    TARGET,
    RANDOM_STATE,
    project_root,
    load_credit_data,
    preprocess_dataframe,
    get_model_matrix,
)

warnings.filterwarnings("ignore")

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
}

MODEL_DISPLAY_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]


def ks_score(y_true, y_prob):
    """计算 KS：最大化累计好坏样本分布差，风控评分模型常用。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

# 包依赖校验
def require_boosting_packages():
    missing = []
    if XGBClassifier is None:
        missing.append("xgboost")
    if LGBMClassifier is None:
        missing.append("lightgbm")
    if missing:
        raise ImportError("缺少依赖包：" + ", ".join(missing) + "。请先执行 pip install " + " ".join(missing))


def as_float32(X):
    """降低矩阵内存占用，避免训练时占用过高。"""
    if isinstance(X, pd.DataFrame):
        return X.astype(np.float32)
    return X.astype(np.float32, copy=False)


def build_models(scale_pos_weight):
    """固定 5 个基线模型；降低并行数量，避免电脑卡死。"""
    require_boosting_packages()
    # 每个key是模型名称，对应一个三元组 (模型实例, 网格搜索参数字典, 是否需要标准化)
    return {
        "Logistic Regression": (
            # 逻辑回归基础固定初始化参数（固定配置，不参与网格搜索调优）
            LogisticRegression(
                max_iter=1200,                # 最大迭代次数，L1正则收敛慢，放大迭代防止训练不收敛告警
                class_weight="balanced",      # 线性模型自带方案，自动反向加权平衡逾期少数样本
                solver="liblinear",           # 唯一同时支持L1、L2两种正则的求解器，适配下方网格penalty搜索
                random_state=RANDOM_STATE    # 固定随机种子，保证实验可复现，每次训练结果一致
            ),
            # 网格搜索候选超参：GridSearchCV会自动遍历全部组合，5折交叉验证择优
            {
                "C": [0.05, 0.1, 0.5, 1.0, 3.0],  # 正则强度倒数，数值越小正则越强，防止过拟合
                "penalty": ["l1", "l2"]          # 两种正则方式：L1特征稀疏筛选 / L2权重平滑收缩
            },
            True,  # 标记：逻辑回归对量纲敏感，训练与预测时必须使用标准化后的特征
        ),
         "Decision Tree": (
            # 单棵决策树基础固定参数
            DecisionTreeClassifier(
                class_weight="balanced",    # 自动平衡正负样本权重，缓解逾期样本少的问题
                random_state=RANDOM_STATE   # 固定随机分裂种子，实验可复现
            ),
            # 网格搜索超参：控制树复杂度，抑制过拟合
            {
                "max_depth": [3, 4, 5, 6],          # 树最大深度，深度越高模型越复杂、越容易过拟合
                "min_samples_leaf": [80, 150, 260]  # 叶子节点最少样本数，数值越大分裂约束越强
            },
            False, # 标记：树模型基于分位数分裂，不受特征量纲影响，无需标准化
        ),
        "Random Forest": (
            # 随机森林集成树基础固定参数
            RandomForestClassifier(
                class_weight="balanced_subsample", # 每棵子树内部单独做样本平衡，泛化效果优于全局balanced
                random_state=RANDOM_STATE,
                n_jobs=1 # 限制单线程运行，避免网格搜索+多树并行双重占用内存，防止电脑卡顿
            ),
            # 网格搜索超参
            {
                "n_estimators": [160, 240],         # 森林内决策树总数量，数量越大拟合效果越好、训练更慢
                "max_depth": [5, 7, 9],             # 单棵树最大深度，相比单棵树可适当放宽深度
                "min_samples_leaf": [40, 80]        # 叶子节点最小样本约束，防止细分到噪声样本
            },
            False,
        ),
         "XGBoost": (
            # XGBoost梯度提升树基础固定参数
            XGBClassifier(
                objective="binary:logistic",    # 任务目标：二分类，输出0~1概率，适配逾期预测
                eval_metric="auc",              # 训练过程监控指标选用风控核心AUC
                tree_method="hist",             # 直方图加速树分裂，降低内存消耗，低配设备友好
                random_state=RANDOM_STATE,
                n_jobs=1,                       # 单线程限制，降低硬件资源占用
                scale_pos_weight=scale_pos_weight # 外部传入正负样本比值，损失函数加权，平衡逾期样本
            ),
            # 网格搜索超参：控制迭代、学习率、正则、行列采样，平衡精度与过拟合
            {
                "n_estimators": [260, 380],    # 梯度提升迭代轮数
                "max_depth": [3, 4],            # 提升树单树深度，浅层树规避过拟合
                "learning_rate": [0.03, 0.06],  # 学习率，小步长更新参数，收敛更稳定
                "min_child_weight": [1, 5],     # 叶子节点样本权重和下限，越大正则越强
                "subsample": [0.85],            # 行采样：每轮随机抽取85%样本训练，固定值减少搜索量
                "colsample_bytree": [0.85],     # 列采样：每棵树随机使用85%特征
                "reg_alpha": [0.0, 0.1],        # L1正则系数，0=不稀疏，0.1轻微筛选弱特征
                "reg_lambda": [1.0, 3.0]        # L2正则系数，收缩特征权重，抑制极端贡献
            },
            False,
        ),
         "LightGBM": (
            # LightGBM轻量梯度提升树基础固定参数
            LGBMClassifier(
                objective="binary",         # 二分类任务
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,                 # 关闭训练冗余日志输出，控制台打印更简洁
                scale_pos_weight=scale_pos_weight # 同XGBoost，少数逾期样本损失加权
            ),
            # 网格搜索超参，和XGBoost逻辑一致，适配LightGBM专属叶子参数num_leaves
            {
                "n_estimators": [260, 380],
                "learning_rate": [0.03, 0.06],
                "num_leaves": [15, 31],     # LightGBM核心参数，限制单树叶子总数，直接控制模型复杂度
                "max_depth": [4, 6],
                "min_child_samples": [30, 80], # 叶子节点最少样本数，替代XGBoost的min_child_weight
                "subsample": [0.85],
                "colsample_bytree": [0.85],
                "reg_alpha": [0.0, 0.1],
                "reg_lambda": [1.0, 3.0],
            },
            False,
        ),
    }


# 网格搜索
def fit_grid_model(name, estimator, params, X_train, y_train, cv):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        pre_dispatch=1,
        verbose=0,
    )
    print(f"开始训练：{name}")
    grid.fit(X_train, y_train)
    return grid

"""XGB/LGBM在分裂节点时会单独把NaN归为一个分支，自动学习缺失样本的分裂方向；
而sklearn的传统模型遇到NaN会直接训练报错，所以需要这个函数做分支区分。"""
def model_supports_nan(model_name):
    return model_name in ["XGBoost", "LightGBM"]


# 消融实验数据预处理
def prepare_ablation_matrix(raw_train, train_index, valid_index, use_fe, use_anomaly, use_missing):
    train_part = raw_train.iloc[train_index].copy()
    valid_part = raw_train.iloc[valid_index].copy()

    train_processed, medians, _, feature_columns, _ = preprocess_dataframe(
        train_part,
        enable_feature_engineering=use_fe,
        enable_anomaly=use_anomaly,
        enable_missing=use_missing,
        enable_scaling=False,
    )
    valid_processed, _, _, _, _ = preprocess_dataframe(
        valid_part,
        medians=medians,
        enable_feature_engineering=use_fe,
        enable_anomaly=use_anomaly,
        enable_missing=use_missing,
        enable_scaling=False,
    )

    allow_nan = not use_missing
    X_train = get_model_matrix(train_processed, feature_columns, allow_nan=allow_nan)
    X_valid = get_model_matrix(valid_processed, feature_columns, allow_nan=allow_nan)
    return X_train, X_valid, feature_columns


# 消融实验
def run_ablation_experiments(raw_train, train_index, valid_index, y_train, y_valid, best_model_name, best_model, models_dir):
    """使用验证集 AUC 最高的模型执行 5 组消融实验。"""
    groups = [
        ("对照组：完整预处理", True, True, True, True),
        ("实验组1：仅移除特征衍生", False, True, True, True),
        ("实验组2：移除异常值处理", True, False, True, True),
        ("实验组3：移除缺失值填充", True, True, False, True),
        ("实验组4：移除标准化", True, True, True, False),
    ]
    rows = []

    for group_name, use_fe, use_anomaly, use_missing, use_scaling in groups:
        X_train, X_valid, feature_columns = prepare_ablation_matrix(
            raw_train, train_index, valid_index, use_fe, use_anomaly, use_missing
        )

        # sklearn 传统模型不能直接接收 NaN；仅在算法兼容层使用固定哨兵值，不改变预处理开关含义。
        if X_train.isnull().sum().sum() > 0 and not model_supports_nan(best_model_name):
            X_train = X_train.fillna(-999)
            X_valid = X_valid.fillna(-999)

        if use_scaling:
            local_scaler = StandardScaler()
            X_train = local_scaler.fit_transform(as_float32(X_train))
            X_valid = local_scaler.transform(as_float32(X_valid))
        else:
            X_train = as_float32(X_train)
            X_valid = as_float32(X_valid)

        ablation_model = clone(best_model)
        if hasattr(ablation_model, "n_jobs"):
            ablation_model.set_params(n_jobs=1)
        ablation_model.fit(X_train, y_train)
        y_prob = ablation_model.predict_proba(X_valid)[:, 1]

        rows.append({
            "实验组": group_name,
            "使用模型": best_model_name,
            "特征衍生": use_fe,
            "异常值处理": use_anomaly,
            "缺失值填充": use_missing,
            "标准化": use_scaling,
            "AUC": roc_auc_score(y_valid, y_prob),
            "KS": ks_score(y_valid, y_prob),
        })
        del X_train, X_valid, ablation_model, y_prob
        gc.collect()

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(os.path.join(models_dir, "ablation_results.csv"), index=False, encoding="utf-8-sig")
    return ablation_df

# 模型训练
# 完整模型训练主函数：数据加载、划分、预处理、网格调参、模型保存、指标汇总、绘图、消融实验、元数据落地
def train_models():
    # 获取项目根目录路径
    root = project_root()
    # 拼接模型保存文件夹路径
    models_dir = os.path.join(root, "models")
    # 创建models文件夹，已存在也不报错
    os.makedirs(models_dir, exist_ok=True)

    # 读取原始信贷数据集
    raw_train, _ = load_credit_data(os.path.join(root, "data"))
    # 提取标签列并转为整数0/1
    y = raw_train[TARGET].astype(int)

    # 1.数据集分层划分 8:2 训练/验证集
    train_index, valid_index = train_test_split(
        np.arange(len(raw_train)),
        test_size=0.2,   # 20%作为验证集，80%训练
        random_state=RANDOM_STATE,  # 固定随机种子，实验可复现
        stratify=y,   # stratify=y 分层抽样，保证训练/验证集中逾期用户占比和原数据集一致，避免分布偏移

    )
    # 按索引切分训练集、验证集DataFrame
    train_part = raw_train.iloc[train_index].copy()
    valid_part = raw_train.iloc[valid_index].copy()
    # 切分对应标签
    y_train = y.iloc[train_index]
    y_valid = y.iloc[valid_index]

    # 2.训练集预处理：计算缺失填充中位数、衍生风控特征、异常截断
    # medians会保存训练集各特征中位数，后续验证集、预测页面统一复用
    train_df, medians, _, feature_columns, _ = preprocess_dataframe(train_part, enable_scaling=False)
    # 验证集预处理：复用训练集的中位数填充，不重新统计分布，模拟线上真实推理流程
    valid_df, _, _, _, _ = preprocess_dataframe(valid_part, medians=medians, enable_scaling=False)
    # 提取最终建模特征矩阵，统一转为float32降低内存占用
    X_train = as_float32(get_model_matrix(train_df, feature_columns))
    X_valid = as_float32(get_model_matrix(valid_df, feature_columns))

    # 3.标准化器训练
    scaler = StandardScaler()
    # 仅用训练集拟合均值方差，避免验证集信息泄露
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    # 验证集仅做transform，不重新拟合
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
    # 保存标准化器文件，实时预测页面加载复用
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # 4.计算类别不平衡权重 scale_pos_weight
    # 统计训练集中正常样本(0)、逾期样本(1)数量
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    # 公式：负样本数 / 正样本数，max兜底防止无逾期样本时分母为0报错
    scale_pos_weight = negative / max(positive, 1)

    # 5.5折分层交叉验证配置
    # StratifiedKFold分层折，每折内正负样本比例和原训练集保持一致
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # 加载5类模型定义、超参网格、是否需要标准化标记
    model_defs = build_models(scale_pos_weight)
    # 存储各模型指标、ROC曲线数据、最优模型记录
    results = []
    roc_data = {}
    best_auc = -1.0
    best_model_name = None
    best_model = None

    # 按预设展示顺序循环训练每一个模型
    for name in MODEL_DISPLAY_ORDER:
        # 取出模型初始化实例、网格搜索超参、是否使用标准化特征
        estimator, params, use_scaled = model_defs[name]
        # 判断模型是否需要标准化特征：逻辑回归用缩放后数据，树模型用原始尺度
        fit_X = X_train_scaled if use_scaled else X_train
        valid_X = X_valid_scaled if use_scaled else X_valid
        # 执行网格搜索调参，返回最优超参组合的GridSearchCV对象
        grid = fit_grid_model(name, estimator, params, fit_X, y_train, cv)
        # 获取网格搜索后的最优模型
        model = grid.best_estimator_
        # 在验证集预测逾期概率
        y_prob = model.predict_proba(valid_X)[:, 1]
        # 计算验证集核心风控指标AUC、KS
        auc = roc_auc_score(y_valid, y_prob)
        ks = ks_score(y_valid, y_prob)
        # 计算ROC曲线坐标点，用于绘图
        fpr, tpr, _ = roc_curve(y_valid, y_prob)

        # 保存训练完成的模型文件到models目录
        model_path = os.path.join(models_dir, MODEL_FILE_MAP[name])
        joblib.dump(model, model_path)

        # 记录当前模型ROC坐标、AUC、KS，用于统一绘图
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc, "ks": ks}
        # 存入模型指标结果列表
        results.append({
            "模型": name,
            "验证集AUC": auc,
            "验证集KS": ks,
            "交叉验证最优AUC": grid.best_score_,
            "最优参数": grid.best_params_,
            "模型文件": MODEL_FILE_MAP[name],
        })

        # 更新全局最优模型（按验证集AUC判断）
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = clone(model)

        # 控制台打印当前模型训练详情
        print(f"\n{name}")
        print(f"模型文件: {model_path}")
        print(f"最优参数: {grid.best_params_}")
        print(f"5折交叉验证最优AUC: {grid.best_score_:.4f}")
        print(f"验证集AUC: {auc:.4f}  验证集KS: {ks:.4f}")

        # 打印分类报告与混淆矩阵，阈值默认0.5
        print(classification_report(y_valid, (y_prob >= 0.5).astype(int), digits=4))
        print(confusion_matrix(y_valid, (y_prob >= 0.5).astype(int)))

        # 释放内存，避免多模型循环训练内存溢出
        del grid, model, y_prob, fit_X, valid_X
        gc.collect()

    # 6.汇总所有模型指标，按验证集AUC、KS降序排序
    result_df = pd.DataFrame(results).sort_values(["验证集AUC", "验证集KS"], ascending=False)
    # 执行消融实验：控制变量验证各衍生特征对模型效果的贡献，使用全局最优模型
    ablation_df = run_ablation_experiments(
        raw_train, train_index, valid_index, y_train, y_valid, best_model_name, best_model, models_dir
    )

    # 7.打包全部训练元数据，供Streamlit前端页面读取
    metadata = {
        "best_model_name": best_model_name,        # 最优模型名称
        "model_files": MODEL_FILE_MAP,             # 模型名称-文件映射字典
        "model_display_order": MODEL_DISPLAY_ORDER,# 页面展示模型顺序
        "medians": medians,                        # 训练集缺失填充中位数（预测页面复用）
        "feature_columns": feature_columns,        # 最终建模特征列表
        "results": result_df,                      # 多模型指标对比表
        "ablation_results": ablation_df,           # 消融实验指标
        "roc_data": roc_data,                      # 所有模型ROC曲线坐标数据
        "scale_pos_weight": scale_pos_weight,      # 类别不平衡权重
    }
    # 持久化保存元数据、ROC原始数据、模型指标csv
    joblib.dump(metadata, os.path.join(models_dir, "metadata.pkl"))
    joblib.dump(roc_data, os.path.join(models_dir, "roc_data.pkl"))
    result_df.to_csv(os.path.join(models_dir, "model_results.csv"), index=False, encoding="utf-8-sig")

    # 8.绘制多模型对比ROC曲线图并保存图片
    plt.figure(figsize=(8, 6))
    for name, item in roc_data.items():
        plt.plot(item["fpr"], item["tpr"], linewidth=2, label=f"{name} AUC={item['auc']:.4f} KS={item['ks']:.4f}")
    # 绘制随机猜测基准虚线AUC=0.5
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("假正例率")
    plt.ylabel("真正例率")
    plt.title("模型 ROC 曲线对比")
    plt.legend()
    plt.tight_layout()  # 自动适配布局防止文字重叠
    plt.savefig(os.path.join(models_dir, "roc_curve.png"), dpi=160)
    plt.close() # 关闭画布释放内存

    # 控制台输出训练完成总结信息
    print("\n模型训练完成，5个模型已分别保存到 models 文件夹。")
    print(result_df.to_string(index=False))
    print("\n消融实验使用模型：" + str(best_model_name))
    print(ablation_df.to_string(index=False))
    # 返回完整训练元数据，供页面缓存读取
    return metadata


if __name__ == "__main__":
    train_models()
