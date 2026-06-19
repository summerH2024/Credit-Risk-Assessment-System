import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42

# 数据集原生10个基础特征
RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# 单独拎出三段逾期次数字段
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# 所有计数型字段，统一做分位数缩尾处理
COUNT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]

# 三个衍生特征：总逾期次数、人均收入、对数负债比率
ENGINEERED_FEATURES = ["TotalPastDue", "IncomePerPerson", "DebtRatio_Log"]

# 风控表达增强特征：不改变原业务规则，只增强逾期强度、收入负债与信贷暴露表达
RISK_FEATURES = [
    "PastDueWeighted",
    "HasPastDue",
    "CreditLinePerAge",
    "DebtIncomePressure",
    "RevolvingDebtInteraction",
]
#全特征
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES + RISK_FEATURES
#仅原始特征，配合后续功能开关，做消融实验
BASELINE_FEATURE_COLUMNS = RAW_FEATURES.copy()

#字段描述字典
COLUMN_DESCRIPTIONS = {
    TARGET: "两年内是否发生 90 天以上严重逾期，1 表示逾期，0 表示正常",
    "RevolvingUtilizationOfUnsecuredLines": "无担保循环额度使用率",
    "age": "借款人年龄",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 天逾期次数",
    "DebtRatio": "负债比率",
    "MonthlyIncome": "月收入",
    "NumberOfOpenCreditLinesAndLoans": "开放式信贷和贷款数量",
    "NumberOfTimes90DaysLate": "90 天以上逾期次数",
    "NumberRealEstateLoansOrLines": "房地产贷款或额度数量",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 天逾期次数",
    "NumberOfDependents": "家属人数",
    "TotalPastDue": "总逾期次数",
    "IncomePerPerson": "人均月收入",
    "DebtRatio_Log": "负债比率对数",
    "PastDueWeighted": "按逾期严重程度加权后的逾期次数",
    "HasPastDue": "是否出现过逾期",
    "CreditLinePerAge": "单位年龄信贷账户数量",
    "DebtIncomePressure": "收入负债压力近似指标",
    "RevolvingDebtInteraction": "额度使用率与负债比率交互项",
}


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


#数据集加载
def load_credit_data(data_dir=None):
    data_dir = data_dir or os.path.join(project_root(), "data")
    train_path = os.path.join(data_dir, "cs-training.csv")
    test_path = os.path.join(data_dir, "cs-test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("未找到 data/cs-training.csv，请先将GiveMeSomeCredit数据集放入data文件夹。")
    #训练集
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    for df in [train, test]:
        if df is not None and "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return train, test

#特征列选择函数:根据是否开启特征工程，返回两套不同的特征字段列表
def get_feature_columns(enable_feature_engineering=True):
    return FEATURE_COLUMNS if enable_feature_engineering else BASELINE_FEATURE_COLUMNS


#异常检测
def detect_anomalies(df):
    """
    按照信贷业务规则定义异常判定条件：
    1.无担保额度使用率 > 1：判定异常；
    2.年龄 <18 或>100：判定异常；
    3.逾期次数为 96、98：属于系统特殊编码，判定为异常；
    4.负债比率、收入、各类计数字段：采用分位数判定极值异常（99.5/99.9/99 分位数
    """
    rules = {
        "RevolvingUtilizationOfUnsecuredLines": df["RevolvingUtilizationOfUnsecuredLines"] > 1,
        "age": (df["age"] < 18) | (df["age"] > 100),

        "NumberOfTime30-59DaysPastDueNotWorse": df["NumberOfTime30-59DaysPastDueNotWorse"].isin([96, 98]),
        "NumberOfTime60-89DaysPastDueNotWorse": df["NumberOfTime60-89DaysPastDueNotWorse"].isin([96, 98]),
        "NumberOfTimes90DaysLate": df["NumberOfTimes90DaysLate"].isin([96, 98]),

        "DebtRatio": df["DebtRatio"] > df["DebtRatio"].quantile(0.995),
        "MonthlyIncome": df["MonthlyIncome"] > df["MonthlyIncome"].quantile(0.999),
        "NumberOfOpenCreditLinesAndLoans": df["NumberOfOpenCreditLinesAndLoans"] > df["NumberOfOpenCreditLinesAndLoans"].quantile(0.99),
        "NumberRealEstateLoansOrLines": df["NumberRealEstateLoansOrLines"] > df["NumberRealEstateLoansOrLines"].quantile(0.99),
        "NumberOfDependents": df["NumberOfDependents"] > df["NumberOfDependents"].quantile(0.99),
    }
    rows = []
    total = max(len(df), 1)
    for col, mask in rules.items():
        count = int(mask.fillna(False).sum())
        rows.append({"字段": col,
                     "字段含义": COLUMN_DESCRIPTIONS.get(col, col),
                     "异常数量": count,
                     "异常比例": count / total})
    return pd.DataFrame(rows)


#异常数据清洗
def clean_anomalies(df):
    cleaned = df.copy()  #不修改原始数据集，所有清洗操作都作用在新副本 cleaned 上，避免原始数据被破坏
    # 额度使用率上限：1.0。
    cleaned.loc[cleaned["RevolvingUtilizationOfUnsecuredLines"] > 1, "RevolvingUtilizationOfUnsecuredLines"] = 1.0
    # 年龄范围：18~100。
    cleaned.loc[(cleaned["age"] < 18) | (cleaned["age"] > 100), "age"] = np.nan
    # 逾期次数 96、98 视为特殊编码，转为缺失值。
    for col in PAST_DUE_COLS:
        cleaned.loc[cleaned[col].isin([96, 98]), col] = np.nan
    # DebtRatio：99.5% 分位数缩尾。
    cleaned["DebtRatio"] = cleaned["DebtRatio"].clip(upper=cleaned["DebtRatio"].quantile(0.995))
    # MonthlyIncome：99.9% 分位数缩尾。
    cleaned["MonthlyIncome"] = cleaned["MonthlyIncome"].clip(upper=cleaned["MonthlyIncome"].quantile(0.999))
    cleaned.loc[cleaned["MonthlyIncome"] <= 0, "MonthlyIncome"] = np.nan
    # 计数类特征：99% 分位数缩尾。
    for col in COUNT_COLS:
        cleaned[col] = cleaned[col].clip(upper=cleaned[col].quantile(0.99))
    return cleaned


#特征工程
def add_features(df):
    featured = df.copy()
    dependents = featured["NumberOfDependents"].fillna(0).clip(lower=0)
    income = featured["MonthlyIncome"].replace([np.inf, -np.inf], np.nan)
    debt_ratio = featured["DebtRatio"].clip(lower=0)
    age = featured["age"].replace(0, np.nan)

    featured["TotalPastDue"] = featured[PAST_DUE_COLS].sum(axis=1)
    featured["IncomePerPerson"] = income / (dependents + 1)
    featured["DebtRatio_Log"] = np.log1p(debt_ratio)
    # 加权逾期特征
    featured["PastDueWeighted"] = (
        featured["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
        + 2 * featured["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
        + 3 * featured["NumberOfTimes90DaysLate"].fillna(0)
    )
    # 是否存在逾期（0/1 二值特征）
    featured["HasPastDue"] = (featured["TotalPastDue"] > 0).astype(int)
    # 年均信贷账户数
    featured["CreditLinePerAge"] = featured["NumberOfOpenCreditLinesAndLoans"] / age
    # 债务收入压力指数
    featured["DebtIncomePressure"] = debt_ratio / np.log1p(income.clip(lower=0))
    # 使用率与负债对数交互特征
    featured["RevolvingDebtInteraction"] = featured["RevolvingUtilizationOfUnsecuredLines"] * featured["DebtRatio_Log"]
    return featured

#缺失值填充(中位数)
def fill_missing_values(df, medians=None, feature_columns=None):
    filled = df.copy()
    columns = feature_columns or [col for col in FEATURE_COLUMNS if col in filled.columns]
    if medians is None:
        medians = filled[columns].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    filled[columns] = filled[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    return filled, medians


# 统一预处理函数 消融实验
def preprocess_dataframe(
    df,
    medians=None,
    scaler=None,
    enable_feature_engineering=True,
    enable_anomaly=True,
    enable_missing=True,
    enable_scaling=False,
    fit_scaler=False,
):
    """统一预处理入口，四个开关用于固定消融实验分组。"""
    processed = df.copy()
    # 是否清洗异常值
    if enable_anomaly:
        processed = clean_anomalies(processed)

    # 是否生成衍生特征
    if enable_feature_engineering:
        processed = add_features(processed)

    # 获取本次建模要用的全部特征名
    feature_columns = get_feature_columns(enable_feature_engineering)

    # 中位数填充缺失值
    if enable_missing:
        processed, medians = fill_missing_values(processed, medians, feature_columns)

    # 标准化缩放
    X = None
    if enable_scaling:
        # 1.提取特征，无穷值转为缺失
        temp = processed[feature_columns].replace([np.inf, -np.inf], np.nan)
        # 2.若仍存在缺失，用中位数补上
        if temp.isnull().sum().sum() > 0:
            if medians is None:
                medians = temp.median(numeric_only=True)
            temp = temp.fillna(medians)
        if fit_scaler or scaler is None:
            # 训练集：新建标准化器，拟合数据
            scaler = StandardScaler()
            scaled = scaler.fit_transform(temp)
        else:
            # 测试集：仅转换，不重新拟合
            scaled = scaler.transform(temp)
        # 缩放后数值写回总表
        processed.loc[:, feature_columns] = scaled
        # 单独输出纯特征矩阵X，直接送入模型
        X = pd.DataFrame(scaled, columns=feature_columns, index=processed.index)

    return processed, medians, scaler, feature_columns, X

# 辅助工具函数
def get_model_matrix(df, feature_columns=None, allow_nan=False):
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"预测特征缺失: {missing}")
    X = df[feature_columns].copy().replace([np.inf, -np.inf], np.nan)
    if not allow_nan and X.isnull().sum().sum() > 0:
        raise ValueError("Input X contains NaN，请先执行 preprocess_dataframe 并启用缺失值填充。")
    return X


# 特征分箱
def make_feature_bins(df):
    work = df.copy()
    work["年龄区间"] = pd.cut(work["age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60以上"])
    work["收入区间"] = pd.qcut(work["MonthlyIncome"].rank(method="first"), q=5, labels=["低", "较低", "中", "较高", "高"])
    work["额度使用率区间"] = pd.cut(work["RevolvingUtilizationOfUnsecuredLines"], bins=[-0.01, 0.2, 0.5, 0.8, 1.0], labels=["0-20%", "20%-50%", "50%-80%", "80%-100%"])
    return work