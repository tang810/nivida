"""
LSTMRegressor：长短期记忆神经网络，一种深度学习回归预测的方法，通常能够适应复杂的数据分布和任务

"""

import os
from contextlib import redirect_stdout
# arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# LSTM
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

#XGB
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
import math
# def arima(data=None):
#         date_average_value = data
#         df =data

#     model = ARIMA(date_average_value['value'], order=(1, 1, 0))  # order=(p, d, q)
#     results = model.fit()

#     plt.figure(figsize=(10, 6))
#     # 确保将'Date'列转换为datetime类型以便正确处理时间序列数据
#     df['date'] = pd.to_datetime(df['date'])
#     plt.plot(df['date'], date_average_value['value'], label='Original')
#     plt.plot(df['date'], results.fittedvalues, color='red', label='Fitted Values')
#     # 设置x轴的刻度标签为月份形式
#     ax = plt.gca()  # 获取当前坐标轴对象
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 设置时间格式化器为年-月

#     # 自动调整子图参数,使得图形适配整个图像区域。
#     plt.autoscale(enable=True, axis='both', tight=True)

#     # 优化x轴刻度显示（可选），特别是当有大量数据时避免过于拥挤。
#     plt.xticks(rotation=45) 

#     plt.legend()
#     plt.savefig('./figs/arima.png')

#     return results

def LSTMRegressor(data=None, taskid=""):
    df = data
    results_str=""
    image_path=[]
    

    #判断数据究竟有几列，如果除去不止两列的话，不用做resahpe处理，否则需要用reshape
    except_date = df.loc[:, ~df.columns.isin(['date'])]
    feat_num = except_date.shape[1]
    #归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(except_date.values)

    # 创建LSTM模型
    model = Sequential()
    model.add(Input(shape=(12,feat_num)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=128,return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dense(feat_num))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 准备数据
    X_train, y_train = [], []
    for i in range(12, len(scaled_data)):
        X_train.append(scaled_data[i-12:i])
        y_train.append(scaled_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # 如果只有一列，确保数据形状符合LSTM的输入要求
    if feat_num == 1:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)


    # 递归预测未来的12个时间节点的数值
    last_sequence = X_train[-1]  # 获取训练集的最后一个序列
    predictions = []

    for _ in range(12):  # 预测12个时间步
        pred = model.predict(last_sequence[np.newaxis, :, :])
        predictions.append(pred[0])  # 存储预测结果
        # 将预测结果追加到输入序列，并删除最早的一个时间步
        last_sequence = np.append(last_sequence[1:], pred, axis=0)

    # 逆归一化预测值
    predictions = scaler.inverse_transform(predictions)

    # # 输出预测结果
    # print(predictions)


    fig, axes = plt.subplots(
        nrows=math.ceil(feat_num / 2),
        ncols=2 if feat_num > 1 else 1,
        figsize=(12, 6),
        dpi=100,
        sharex=True
    )

    # 统一处理单子图的情况
    if feat_num == 1:
        axes = np.array([axes])

    # 展平 axes 数组，以便统一索引
    axes = axes.flatten()

    #计算样本间隔
    recent_intervals = df['date'].diff().dropna().tail(5)
    avg_interval = recent_intervals.mean()
    results_str+=f"平均日期间隔: {avg_interval}"

    # 获取最后一个日期
    last_date = df['date'].iloc[-1]
    # 使用这个平均间隔来生成未来的日期
    future_dates = [last_date + avg_interval * i for i in range(1, 13)]
    # 将日期转换为 pd.DatetimeIndex
    future_dates = pd.DatetimeIndex(future_dates)

    # 绘制每一列的数据
    for i, col in enumerate(except_date.columns):
        axes[i].plot(future_dates, predictions[:, i], color='red', label='Predicted Future Values')
        axes[i].plot(df['date'], df[col], color='blue', label='Original Data')
        axes[i].set_title(f"{col}")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if i == 0:
            axes[i].legend()

    # 如果子图多余，隐藏
    for i in range(feat_num, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.autoscale(enable=True, axis='both', tight=True)
    tmp_img_path = "src/tmp_img_save/LSTM.png"
    image_path.append(tmp_img_path)
    plt.savefig(tmp_img_path,bbox_inches='tight',pad_inches=0.1)
    return results_str,image_path


from pgmpy.estimators import HillClimbSearch
from dowhy import CausalModel
from pgmpy.estimators import K2Score
import networkx as nx
import matplotlib.pyplot as plt

def causal_inference(data=None, taskid=""):
    #预测完之后需要跟一个因果推断
    #先对数据集进行因果推断，看看数据整体的关联关系
    results_str=""
    image_path=[]

    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=K2Score(data))
    results_str += f"结构学习到的模型边为：{best_model.egdes()}\n"
    G = nx.DiGraph(best_model.edges())
    dot_str = nx.nx_pydot.to_pydot(G).to_string()
    #绘制贝叶斯网络
    plt.figure(figsize=(12, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold', edge_color='gray', arrows=True, arrowsize=20)
    plt.title("Bayesian Network Structure")

    tmp_img_path = "src/tmp_img_save/DAG.png"
    image_path.append(tmp_img_path)
    plt.savefig(tmp_img_path,bbox_inches='tight',pad_inches=0.1)
    
    try:
        model = CausalModel(
            data=data,
            treatment=str(data.columns[1]),  # 选择你认为可能的干预变量
            outcome=str(data.columns[2]),  # 神经网络的预测结果
            graph=dot_str
        )
        identified_estimand = model.identify_effect()
        # 估计因果效应
        causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

        results_str+=f"Causal Estimate: {causal_estimate.value}\n"
    except Exception as e:
        results_str+=f"错误为：{e},构建因果模型失败\n"

    return results_str,image_path





# def XGB_Regressor(data=None):


#     # 判断数据究竟有几列，如果除去不止两列的话，不用做reshape处理，否则需要用reshape
#     except_date = data.loc[:, ~data.columns.isin(['date'])]
#     feat_num = except_date.shape[1]

#     # 归一化处理
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(except_date.values)

#     # 准备数据
#     time_window = 12
#     X_train, y_train = [], []
#     for i in range(time_window, len(scaled_data)):
#         X_train.append(scaled_data[i-time_window:i, :].flatten())  # 展平时间窗口
#         y_train.append(scaled_data[i, 0])
#     X_train, y_train = np.array(X_train), np.array(y_train)

#     # 创建和训练XGBoost模型
#     model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
#     model.fit(X_train, y_train)

#     # 预测
#     predictions = model.predict(X_train[-12:])
#     predictions = scaler.inverse_transform(predictions)

#     # 打印预测值和实际值
#     with open("./output/output.txt", "a") as f:
#         with redirect_stdout(f):
#             print("通过XGBRegressor预测时间点的数值为：", predictions)

#     return predictions
