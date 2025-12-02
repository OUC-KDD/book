# -*- coding: utf-8 -*-
"""
基于浮标数据(46221 / 46251)的 SVR 时间序列预测示例
数据格式：
    DateTime, u10, v10, WVHT       # 或 DateTime, WVHT, u10, v10

思想：
    1. 读取并按时间排序，按 1 小时重采样，缺测线性插值；
    2. 以过去 history_len 小时的 (WVHT, u10, v10) 作为输入特征，
       预测下一小时(或若干小时后)的 WVHT；
    3. 前 80% 样本用于训练，后 20% 用于测试；
    4. 使用 标准化 + SVR（RBF核） 进行回归，并输出 RMSE、R² 等指标。
"""

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics


# =========================
# 1. 读取与基础预处理
# =========================
def load_buoy_csv(path):
    """
    读取单个浮标站点的 CSV，并做基础预处理：
    - 解析时间
    - 按时间排序
    - 以 1 小时为间隔重采样
    - 对 WVHT, u10, v10 做线性插值填补
    """
    df = pd.read_csv(path)

    # 统一列名顺序
    # 有的文件是 [DateTime, u10, v10, WVHT]，有的是 [DateTime, WVHT, u10, v10]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').set_index('DateTime')

    # 只保留数值列
    cols = ['WVHT', 'u10', 'v10']
    # 有的文件 WVHT 不在第一列，重新取列
    df = df[cols]

    # 以 1 小时为频率重采样（补齐缺失时间点）
    full_index = pd.date_range(start=df.index.min(),
                               end=df.index.max(),
                               freq='1H')
    df = df.reindex(full_index)

    # 线性插值填补缺测
    df[cols] = df[cols].interpolate(method='time')

    # 若开头/结尾还有 NaN（无法插值），再用前向/后向填补一下
    df[cols] = df[cols].ffill().bfill()

    return df


# =========================
# 2. 构造时间序列样本（滑动窗口）
# =========================
def build_time_series_samples(df, history_len=24, horizon=1):
    """
    输入：
        df: 经过预处理的 DataFrame，包含列 [WVHT, u10, v10]
        history_len: 作为输入的历史长度（单位：小时）
        horizon: 预测步长，=1 表示预测 history_len 之后的下一小时

    输出：
        X: [N, history_len * 3] 的特征矩阵
        y: [N,] 的目标向量（未来的 WVHT）
    """
    values = df[['WVHT', 'u10', 'v10']].values
    N = len(values)

    X_list, y_list = [], []

    # 构造滑动窗口
    # 窗口 [t, t+history_len-1] → 目标为 t+history_len-1+horizon 对应的 WVHT
    for start in range(0, N - history_len - horizon + 1):
        end = start + history_len
        target_index = end + horizon - 1

        window = values[start:end, :]          # [history_len, 3]
        target = values[target_index, 0]       # WVHT 作为预测目标

        X_list.append(window.reshape(-1))      # 拉平成 1 维向量
        y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


# =========================
# 3. SVR 基线模型（与论文风格一致）
# =========================
def build_svr_model():
    """
    线性核 SVR（LinearSVR）+ 标准化
    """
    svr = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=5000,     # 必要时可以再调大一些
            random_state=0
        ))
    ])
    return svr


# =========================
# 4. 训练与测试
# =========================
def train_and_eval_svr(X, y, train_ratio=0.8):
    """
    前 train_ratio 的样本作为训练集，后面的作为测试集
    """
    N = len(y)
    split = int(N * train_ratio)

    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    model = build_svr_model()
    model.fit(X_train, y_train)

    # 训练集 & 测试集预测
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # 指标
    def print_metrics(name, y_true, y_pred):
        mae  = metrics.mean_absolute_error(y_true, y_pred)
        mse  = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = metrics.r2_score(y_true, y_pred)
        print(f"=== {name} ===")
        print("R^2 :", r2)
        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print()

    print_metrics("Train", y_train, y_pred_train)
    print_metrics("Test",  y_test,  y_pred_test)

    return model


# =========================
# 5. 主程序
# =========================
if __name__ == "__main__":
    # 以 46221 这一个浮标为例，如需 46251 只需换文件名
    path_46221 = "46221.csv"
    path_46251 = "46251.csv"

    # 读取一个站点的数据
    df_21 = load_buoy_csv(path_46221)
    print("46221 预处理后数据形状:", df_21.shape)

    # 构造时间序列样本（例如：用前 24 小时预测下一小时）
    history_len = 24   
    horizon = 1        # 预测 1 小时后
    X, y = build_time_series_samples(df_21, history_len=history_len,
                                     horizon=horizon)
    print("样本特征矩阵形状:", X.shape)
    print("样本标签向量形状:", y.shape)

    # 训练 + 测试
    model = train_and_eval_svr(X, y, train_ratio=0.8)

