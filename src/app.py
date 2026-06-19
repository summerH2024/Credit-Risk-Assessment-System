import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.set_loglevel("error")

from preprocessing import (
    TARGET,
    FEATURE_COLUMNS,
    COLUMN_DESCRIPTIONS,
    project_root,
    load_credit_data,
    detect_anomalies,
    preprocess_dataframe,
    get_model_matrix,
    make_feature_bins,
)

st.set_page_config(page_title="个人信用风险评估与逾期预测", layout="wide")

PAGES = ["项目概览", "数据读取与探索", "数据清洗", "特征工程", "模型训练与评估", "实时预测"]

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
}

MODEL_LABELS = {
    "逻辑回归": "Logistic Regression",
    "决策树": "Decision Tree",
    "随机森林": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}

#@st.cache_data:缓存表格数据
#@st.cache_resource:缓存模型、大对象、训练资源


# 缓存原始数据集
@st.cache_data(show_spinner=False, persist=False)
def cached_train_data():
    train_df, test_df = load_credit_data()
    return train_df, test_df


# 缓存清洗后完整数据集
@st.cache_data(show_spinner=False, persist=False)
def cached_clean_data():
    # 读取原始训练、测试全量数据
    train_df, test_df = load_credit_data()
    # 只用训练集做全套预处理，不开启标准化
    clean_df, medians, scaler, feature_columns, _ = preprocess_dataframe(train_df, enable_scaling=False)
    return train_df, clean_df, medians, feature_columns


# 缓存全部实验元数据
# 网页不用每次切换页面重复读取 csv、pkl 文件，一次性加载后全程复用。
@st.cache_resource(show_spinner=False)
def cached_metadata():
    models_dir = os.path.join(project_root(), "models")
    metadata_path = os.path.join(models_dir, "metadata.pkl")
    # 如果训练脚本跑完已经导出 metadata.pkl，直接读取并返回，跳过后面读取 csv、计算最优模型逻辑
    if os.path.exists(metadata_path):
        return joblib.load(metadata_path)

    # 无预存 pkl时，现场读取两份结果表格
    result_path = os.path.join(models_dir, "model_results.csv")
    ablation_path = os.path.join(models_dir, "ablation_results.csv")

    if not os.path.exists(result_path):
        return None
    results = pd.read_csv(result_path)
    # 选出全局最优模型
    best_model_name = results.sort_values(["验证集AUC", "验证集KS"], ascending=False).iloc[0]["模型"]
    # 读取消融实验结果
    ablation = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None
    return {
        "best_model_name": best_model_name,
        "model_files": MODEL_FILE_MAP,
        "feature_columns": FEATURE_COLUMNS,
        "results": results,
        "ablation_results": ablation,
        "roc_data": {},
    }


# 按需缓存单个模型
# 前端下拉框选择哪个模型，只加载对应单个.pkl文件，不会一次性加载 5 个模型，大幅降低内存占用
@st.cache_resource(show_spinner=False)
def cached_model(model_name):
    models_dir = os.path.join(project_root(), "models")
    file_name = MODEL_FILE_MAP.get(model_name)
    if not file_name:
        return None
    model_path = os.path.join(models_dir, file_name)
    return joblib.load(model_path) if os.path.exists(model_path) else None

# 缓存标准化器
# 逻辑回归预测必须标准化，单独缓存标准化器，实时预测页面复用
@st.cache_resource(show_spinner=False)
def cached_scaler():
    scaler_path = os.path.join(project_root(), "models", "scaler.pkl")
    return joblib.load(scaler_path) if os.path.exists(scaler_path) else None


def compact_header(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def show_metric_row(items):
    # 根据传入指标数量，创建同等数量并排列
    cols = st.columns(len(items))
    # 遍历每一列、每一组(标签,数值)
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# 缺失值横向条形图
def plot_missing(df):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(0.5, 0.5, "无缺失值", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.barh(missing.index, missing.values, color="#4c78a8")
        ax.set_xlabel("缺失数量")
    st.pyplot(fig, use_container_width=True) # 图表自适应页面宽度
    plt.close(fig)

# 样本不平衡饼图
def plot_class_pie(df):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    values = df[TARGET].value_counts().sort_index()
    ax.pie(values.values, labels=["正常用户", "逾期用户"], autopct="%1.2f%%",
           colors=["#5b8ff9", "#e8684a"], explode=[0, 0.08], startangle=90)
    ax.set_title("样本不平衡分布")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# 特征分布双图:直方图 + 横向箱线图
def plot_feature_distribution(df, feature):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ser = df[feature].dropna()
    axes[0].hist(ser, bins=20, color="#4c78a8", alpha=0.7) # 统计数值区间内样本数量，判断分布形态（正态 / 右偏 / 多峰）将数值范围切分为 20 个区间
    axes[0].set_title(f"{feature} 直方图")
    axes[1].boxplot(ser, vert=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#72b7b2"))
    axes[1].set_title(f"{feature} 箱线图")
    plt.tight_layout() # 自动调整子图间距，防止标题、坐标轴文字重叠拥挤
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# 项目概览
def page_overview():
    train_df, _ = cached_train_data()
    compact_header("基于数据挖掘的用户信用风险评估与逾期预测")
    show_metric_row([
        ("训练样本量", f"{len(train_df):,}"),
        ("字段数量", f"{train_df.shape[1]}"),
        ("逾期样本占比", f"{train_df[TARGET].mean() * 100:.2f}%"),
        ("建模任务", "二分类"),
    ])
    st.markdown("#### 项目流程")
    st.write("数据读取、探索分析、异常值检测、业务规则清洗、特征工程、标准化、8:2分层抽样、5模型训练、5折交叉验证、AUC/KS、消融实验、实时预测。")
    desc = pd.DataFrame([{"字段": c, "含义": COLUMN_DESCRIPTIONS.get(c, "")} for c in train_df.columns if c in COLUMN_DESCRIPTIONS])
    st.dataframe(desc, use_container_width=True, height=420)

# 数据读取与探索
def page_explore():
    train_df, _ = cached_train_data()
    compact_header("数据读取与探索")
    show_metric_row([
        ("样本量", f"{train_df.shape[0]:,}"),
        ("变量数", train_df.shape[1]),
        ("缺失单元格", f"{int(train_df.isnull().sum().sum()):,}"),
        ("目标变量", TARGET),
    ])
    # 页面切分为左右两栏，宽度比例 1.15 : 0.85
    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        plot_missing(train_df) # 绘制横向条形图，展示各特征缺失数量
    with c2:
        plot_class_pie(train_df) # 绘制饼图，展示正常 / 逾期样本占比、样本不平衡情况
    st.markdown("#### 核心特征分布")
    # 取前10个预设特征，过滤掉不存在的列
    raw_features = [c for c in FEATURE_COLUMNS[:10] if c in train_df.columns]
    # 下拉框选择特征，默认选中第一个
    feature = st.selectbox("选择特征", raw_features, index=0)
    # 绘制直方图+箱线图双图
    plot_feature_distribution(train_df, feature)
    # 展示原始数据集前 30 行预览
    st.dataframe(train_df.head(30), use_container_width=True, height=480)

# 数据清洗
def page_cleaning():
    train_df, clean_df, _, feature_columns = cached_clean_data()
    compact_header("数据清洗")
    # 异常值检测结果表格
    st.dataframe(detect_anomalies(train_df), use_container_width=True, height=386)
    show_metric_row([
        ("清洗前缺失值", f"{int(train_df.isnull().sum().sum()):,}"),
        ("清洗后建模缺失值", f"{int(clean_df[feature_columns].isnull().sum().sum()):,}"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        plot_feature_distribution(train_df, "RevolvingUtilizationOfUnsecuredLines")
    with c2:
        plot_feature_distribution(clean_df, "RevolvingUtilizationOfUnsecuredLines")

# 特征工程
def page_features():
    _, clean_df, _, feature_columns = cached_clean_data()
    compact_header("特征工程")
    show_metric_row([
        ("保留特征", "TotalPastDue"),
        ("保留特征", "IncomePerPerson"),
        ("保留特征", "DebtRatio_Log"),
        ("最终特征数", len(feature_columns)),
    ])
    st.markdown("#### 特征-标签交叉分析")
    binned = make_feature_bins(clean_df)
    tabs = st.tabs(["年龄区间", "收入区间", "额度使用率区间"])
    for tab, col in zip(tabs, ["年龄区间", "收入区间", "额度使用率区间"]):
        with tab:
            # 分组统计：每组样本总量、每组逾期均值（逾期率）
            cross = binned.groupby(col, observed=True)[TARGET].agg(["count", "mean"]).reset_index()
            cross.columns = [col, "样本数", "逾期率"]
            # 展示分箱统计表
            st.dataframe(cross, use_container_width=True)
            # 绘制柱状图：X=区间，Y=对应区间逾期率
            fig, ax = plt.subplots(figsize=(7, 3.3))
            ax.bar(cross[col], cross["逾期率"], color="#e8684a")
            # 限制Y轴上限，图表更美观，避免极端值压缩视图
            ax.set_ylim(0, min(0.15, cross["逾期率"].max() * 1.3))
            ax.set_title(f"{col} 逾期率")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    st.markdown("#### 相关性与多重共线性分析")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = clean_df[feature_columns].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1) #热力图配色 RdBu_r：红色 = 强正相关，蓝色 = 强负相关，白色 = 无相关
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # 右侧附加颜色条，直观对应颜色与相关强度
    ax.set_xticks(range(len(feature_columns)))
    ax.set_yticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_columns, fontsize=9)
    ax.set_title("特征相关性热力图")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# 模型训练与评估
def page_model():
    compact_header("模型训练与评估")
    metadata = cached_metadata()
    if not metadata:
        st.warning("请先运行 python src/train.py")
        return
    st.markdown("#### 5个模型 AUC/KS 对比")
    # 渲染模型指标表格，铺满页面宽度
    st.dataframe(metadata["results"], use_container_width=True)
    # 表格按验证集AUC降序排列，第一行即为效果最优模型
    best = metadata["results"].iloc[0]
    # 横向卡片展示最优模型核心评估指标
    show_metric_row([
        ("最优模型", metadata["best_model_name"]),
        ("最高AUC", f"{best['验证集AUC']:.4f}"),
        ("最高KS", f"{best['验证集KS']:.4f}"),
    ])
    st.markdown("#### 消融实验结果")
    # 判断是否存在消融实验数据
    if metadata.get("ablation_results") is not None:
        # 展示消融对比表格，查看剔除某类特征后指标下降幅度
        st.dataframe(metadata["ablation_results"], use_container_width=True)
    else:
        st.info("无消融结果")
    st.markdown("#### ROC 曲线")
    # 拼接ROC图片完整存储路径：项目根目录/models/roc_curve.png
    roc_png = os.path.join(project_root(), "models", "roc_curve.png")
    if os.path.exists(roc_png):
        # 加载并自适应宽度展示ROC曲线图，直观对比各模型分类性能
        st.image(roc_png, use_container_width=True)

# 实时预测
def page_predict():
    compact_header("实时预测")
    metadata = cached_metadata()
    if not metadata:
        st.warning("请先训练模型")
        return
    label = st.selectbox("选择预测模型", list(MODEL_LABELS.keys()), index=0)
    # 映射下拉中文名称到模型真实文件名
    model_name = MODEL_LABELS[label]
    # 读取缓存的训练好的模型文件
    model = cached_model(model_name)
    if not model:
        st.warning("模型文件不存在")
        return

    # 将页面分为3列，摆放所有特征输入框
    c1, c2, c3 = st.columns(3)
    with c1:
        # 无担保循环授信使用率，范围0~1，步长0.01，默认0.35
        revolving = st.number_input("无担保额度使用率", 0.0, 1.0, 0.35, 0.01)
        # 用户年龄，合法区间18~100
        age = st.number_input("年龄", 18, 100, 45, 1)
        # 30-59天逾期历史次数
        past_30 = st.number_input("30-59天逾期次数", 0, 20, 0, 1)
        # 负债比率
        debt_ratio = st.number_input("负债比率", 0.0, 10000.0, 0.35, 0.01)
    with c2:
        # 用户月收入，支持大额数值，步长500
        income = st.number_input("月收入", 0.0, 1000000.0, 6500.0, 500.0)
        # 持有的信贷、贷款总条数
        open_credit = st.number_input("开放式信贷和贷款数量", 0, 100, 8, 1)
        # 90天及以上重度逾期次数
        late_90 = st.number_input("90天以上逾期次数", 0, 20, 0, 1)
    with c3:
        # 房贷、不动产类贷款数量
        real_estate = st.number_input("房地产贷款或额度数量", 0, 50, 1, 1)
        # 60-89天中度逾期次数
        past_60 = st.number_input("60-89天逾期次数", 0, 20, 0, 1)
        # 家庭赡养家属人数
        dependents = st.number_input("家属人数", 0, 20, 0, 1)

    # 将页面输入的所有特征组装为单行DataFrame，和训练集字段对齐
    raw = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_credit,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": past_60,
        "NumberOfDependents": dependents,
    }])

    if st.button("开始预测", type="primary", use_container_width=True):
        feat_cols = metadata.get("feature_columns", FEATURE_COLUMNS)
        # 执行数据预处理：缺失填充、异常截断、衍生特征，不做标准化缩放
        clean, _, _, _, _ = preprocess_dataframe(raw, medians=metadata.get("medians"), enable_scaling=False)
        # 提取建模用特征矩阵X
        X = get_model_matrix(clean, feat_cols)

        # 逻辑回归模型需要标准化，树模型无需缩放
        if model_name == "Logistic Regression":
            # 读取训练阶段保存的标准化器
            scaler = cached_scaler()
            if not scaler:
                st.error("请生成 scaler.pkl")
                return
            # 对特征矩阵执行标准化变换
            X = scaler.transform(X)

        # 模型预测：取样本属于逾期(标签1)的概率
        prob = float(model.predict_proba(X)[:, 1][0])
        # 进度条可视化逾期概率
        st.progress(prob)
        # 卡片展示百分比逾期概率，保留2位小数
        st.metric("预测逾期概率", f"{prob*100:.2f}%")
        # 底部小字标注当前使用的模型
        st.caption(f"模型：{label}")

        # 分三段阈值划分风险等级
        if prob >= 0.5:
            st.error("风险等级：高风险")
        elif prob >= 0.25:
            st.warning("风险等级：中风险")
        else:
            st.success("风险等级：低风险")

# 主入口与页面路由
def main():
    st.sidebar.title("信用风险评估")
    page = st.sidebar.radio("页面导航", PAGES, index=0)
    if page == "项目概览":
        page_overview()
    elif page == "数据读取与探索":
        page_explore()
    elif page == "数据清洗":
        page_cleaning()
    elif page == "特征工程":
        page_features()
    elif page == "模型训练与评估":
        page_model()
    elif page == "实时预测":
        page_predict()


if __name__ == "__main__":
    main()