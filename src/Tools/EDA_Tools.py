"""
total_trend: 检查整体趋势
fft_periodic: 使用傅里叶变换分析周期性
acf_and_pacf: 相关系数和自相关系数的计算
z_score：z值的计算，统计
adf_detection: adf检测平稳性

CBLOF：基于聚类的异常检测方法
"""

import sys
import os
import time 
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

import numpy as np
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as pltimport 

import time
from pprint import pprint

import georinex as gr
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from contextlib import redirect_stdout
import math
import matplotlib
from collections import Counter
import seaborn as sns
import matplotlib.font_manager as fm

# 统一图像风格
title_font_size = 12
label_font_size = 12
tick_font_size = 10
legend_font_size = 10

# 获取字体文件的绝对路径
#current_dir = os.path.dirname(os.path.abspath(__file__))
#font_path = os.getenv('FONT_PATH', None)
font_path = os.path.join('fonts', 'SimHei.ttf')
# 添加字体文件
fm.fontManager.addfont(font_path)

def fft_periodic(data=None, taskid="", idx=0, date_name=None):
    """
    使用傅里叶变换分析周期性
    
    Args:
        data: DataFrame数据
        taskid: 任务ID
        idx: 数据集索引
        date_name: 日期列名，如果为None则使用索引
    """

    df = data
    # 获取数值列
    if date_name:
        df_drop_date = df.drop(date_name, axis=1)
        numeric_columns = df_drop_date.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
     # 处理日期列和x轴数据
    if date_name and date_name in df.columns:
        if pd.api.types.is_integer_dtype(df[date_name]):
            x_values = df[date_name]
            x_label = date_name
            use_date = False
        else:
            try:
                df[date_name] = pd.to_datetime(df[date_name])
                x_values = df[date_name]
                x_label = 'Year-Month'
                use_date = True
            except:
                x_values = df[date_name]
                x_label = date_name
                use_date = False
    else:
        # 如果date_name不存在或为None
        x_values = np.arange(len(df))
        x_label = 'Index'
        use_date = False
    
    # 确定数值列的数量和子图布局
    num_columns = len(numeric_columns)
    nrows = math.ceil(num_columns / 2)
    ncols = 2 if num_columns > 1 else 1
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(9, 6),
        dpi=100,
        sharex=True
    )

    # 处理axes的形状
    if num_columns == 1:
        axes = [axes]
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    results = ""
    # 遍历每个数值列并在单独的子图中绘制
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        fft_result = np.fft.fft(df[column])
        frequencies = np.fft.fftfreq(len(df[column]), d=1)
        magnitude = np.abs(fft_result)
        #results += "Column:{} magnitude:{} \n".format(column, magnitude)
        ax.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2], label=column)
        ax.set_title(f'{column} Frequency')
        ax.set_ylabel('Magnitude')
        
        if use_date:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        ax.grid(True)

    # 如果是奇数个子图，删除最后一个空白子图
    if num_columns % 2 == 1 and num_columns > 1:
        fig.delaxes(axes[-1])

    # 设置x轴标签
    if num_columns == 1:
        # 单图情况
        axes[-1].set_xlabel(x_label)
        if use_date:
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    else:
        # 多图情况
        last_row_start = (nrows - 1) * ncols - num_columns % 2  # 计算最后两个子图的索引
        last_row_axes = axes[last_row_start:]  # 获取最后两个子图
        for ax in last_row_axes:
            ax.set_xlabel(x_label)
            # 只在日期类型时旋转标签
            if use_date:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 调整布局
    plt.tight_layout()
    plt.autoscale(enable=True, axis='both', tight=True)

    # 保存图像
    task_plot_path = 'images/{}'.format(taskid)
    if not os.path.exists(task_plot_path):
        os.mkdir(task_plot_path)
    
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    trend_path = os.path.join(task_plot_path, 'trend_{}.png'.format(current_time))
    plt.savefig(trend_path)
    plt.close()

    return [trend_path], None

def total_trend(data=None, taskid="", idx=0, date_name=None):
    """
    绘制总体趋势图，只对数值型特征绘制子图
    
    Args:
        data: DataFrame数据
        taskid: 任务ID
        idx: 数据集索引
        date_name: 日期列名，如果为None则使用索引
    """
    df = data
    # 获取数值列
    if date_name:
        df_drop_date = df.drop(date_name, axis=1)
        numeric_columns = df_drop_date.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
     # 处理日期列和x轴数据
    if date_name and date_name in df.columns:
        if pd.api.types.is_integer_dtype(df[date_name]):
            x_values = df[date_name]
            x_label = date_name
            use_date = False
        else:
            try:
                df[date_name] = pd.to_datetime(df[date_name])
                x_values = df[date_name]
                x_label = 'Year-Month'
                use_date = True
            except:
                x_values = df[date_name]
                x_label = date_name
                use_date = False
    else:
        # 如果date_name不存在或为None
        x_values = np.arange(len(df))
        x_label = 'Index'
        use_date = False
    
    # 确定数值列的数量和子图布局
    num_columns = len(numeric_columns)
    nrows = math.ceil(num_columns / 2)
    ncols = 2 if num_columns > 1 else 1
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        # figsize=(9, 6),
        figsize=(9, nrows * 2),
        dpi=100,
        sharex=False,
    )
    # 在函数开始时设置主题和字体
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置备选字体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 处理axes的形状
    if num_columns == 1:
        axes = [axes]
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    sns.set_theme(style="whitegrid",
                  font=['sans-serif', 'SimHei'],
                  rc={'axes.unicode_minus': False})
 
    # 使用 Seaborn 提供的 "deep" 配色方案，提供柔和且区分度高的颜色
    colors = sns.color_palette("deep", n_colors=num_columns)  # "deep" 配色方案

    # 遍历每个数值列并在单独的子图中绘制
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        ax.plot(x_values, df[column], label=column, linewidth=2, marker='o', markersize=4, color=colors[i])
        ax.set_title(f'{column} Trend', fontsize=title_font_size, weight='bold')
        ax.set_ylabel('Variation', fontsize=label_font_size)
        
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)  # 设置网格线样式和透明度
        ax.legend(fontsize=legend_font_size)  # 设置图例字体大小
        ax.tick_params(axis='both', labelsize=tick_font_size)  # 设置坐标轴刻度的字体大小

    # (奇数个子图最后一列问题)如果是奇数个子图，删除最后一个空白子图
    if num_columns % 2 == 1 and num_columns > 1:
        fig.delaxes(axes[-1])
        axes = axes[:-1]

    # 设置x轴标签
    if num_columns == 1:
        # 单图情况
        axes[-1].set_xlabel(x_label, fontsize=label_font_size)
        if use_date:
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    else:
        # 多图情况
        # last_row_start = (nrows - 1) * ncols - num_columns % 2  # 最后2个子图起始索引
        last_row_start = (nrows - 1) * ncols   # 最后2个子图起始索引
        for i in range(num_columns):
            ax = axes[i]
            # 只为2个子图显示x轴
            if i >= last_row_start:
                ax.set_xlabel(x_label, fontsize=label_font_size)
                if use_date:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            else:
                # 隐藏x轴并调整子图位置
                ax.set_xticklabels([])
                ax.set_xlabel('')

    # 调整布局
    plt.tight_layout()
    plt.autoscale(enable=True, axis='both', tight=True)

    # 保存图像
    task_plot_path = 'images/{}'.format(taskid)
    if not os.path.exists(task_plot_path):
        os.mkdir(task_plot_path)
    
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    trend_path = os.path.join(task_plot_path, 'trend_{}.png'.format(current_time))
    plt.savefig(trend_path)
    plt.close()

    return [trend_path], None

def acf_and_pacf(data=None,taskid="",idx=0, date_name=None):
    df = data
    column_names = df.columns.tolist()
    if date_name:
        df = df.drop(date_name, axis=1)
    col_exp_date = df.columns.tolist()

    # 使用value列数据进行ACF和PACF计算
    lags = 12
    acf_values = []
    pacf_values = []
    for i in col_exp_date:
        acf_value = acf(df[i], nlags=lags, fft=True)
        pacf_value = pacf(df[i], nlags=lags)
        acf_values.append('acf value of {} is {}\n'.format(i,acf_value))
        pacf_values.append('pacf value of {} is {}\n'.format(i,pacf_value))
        
    result = "\nacf:{} \n pacf:{}\n".format(acf_values, pacf_values)
    return [], result
    

def z_score(data=None,taskid="",idx=0, date_name=None):
    results_str = ""
    df = data
    column_names = df.columns.tolist()
    # 获取数值列
    if date_name:
        df_drop_date = df.drop(date_name, axis=1)
        numeric_columns = df_drop_date.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
    col_exp_date = df.columns.tolist()
    num_columns = len(numeric_columns)
    
    fig, axes = plt.subplots(
        nrows=math.ceil(num_columns / 2),
        ncols=2 if num_columns > 1 else 1,
        figsize=(12, 6),
        dpi=100,
        sharex=True
    )

    if num_columns == 1:
        axes = [axes]

    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    global_min = np.min(df[col_exp_date].min())
    global_max = np.max(df[col_exp_date].max())

    # 遍历每一列（不包括日期列）并在单独的子图中绘制
    for i, column in enumerate(col_exp_date):
        mean = np.mean(df[column])
        std_dev = np.std(df[column])
        z_scores = (df[column] - mean) / std_dev
        threshold = 2.7 
        outliers_indices = np.abs(z_scores) > threshold
        indices = np.where(outliers_indices)[0]
        # print("Potential Outliers' Indices:", indices)
        
        ax = axes[i]
        
        # 绘制Boxplot
        ax.boxplot(df[column], vert=True, patch_artist=True, showmeans=True)
        
        # 设置标题和Y轴标签
        ax.set_title(f'{column} Boxplot')

        # 设置所有子图的纵坐标范围相同
        # ax.set_ylim(global_min, global_max)
        ax.set_xticks([])
        
        # 显示网格线
        ax.grid(True)
        
        #每个column特征变量的文字结果
        results_str += f"{column} z_score值为： {z_scores}\n"
        results_str += f"根据z score的值，在{column}中，超过阈值视为异常点的时间和取值为：\n"
        results_str += f"date:{df[date_name][indices].tolist()}\nvalue:{df[column][indices].tolist()}\n\n"

    # 仅在最后一个子图上设置x轴标签
    if num_columns == 1:
        axes[-1].set_xlabel('Year-Month')
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[-2].set_xlabel('Year-Month')
        plt.setp(axes[-2].xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Year-Month')
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
    plt.tight_layout()  # 可以调大pad来增加子图之间的间距
    plt.autoscale(enable=True, axis='both', tight=True)
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    task_plot_path = 'static/images/MultiAgents/{}'.format(taskid)
    if not os.path.exists(task_plot_path):
        os.mkdir(task_plot_path)
    trend_path = os.path.join(task_plot_path, 'trend_plot_{}_{}.png'.format(str(idx),current_time))
    plt.savefig(trend_path)

    return trend_path, "z_score的结果为：\n{}".format(results_str)
    
def adf_detection(data=None, taskid="",idx=0, date_name=None):
    df = data
    column_names = df.columns.tolist()
    # 检查是否包含“日期”列
    if '日期' in column_names:
        date_name='日期'
    elif 'date' in column_names:
        # 将'date'列转换为日期格式
        date_name='date'
    num_columns = df.shape[1] - 1
    col_exp_date = df.columns.tolist()
    col_exp_date.remove(date_name)
    
    results = []
    for i in col_exp_date:
        result = "\nadf result of column {} is{}:\n".format(i, adfuller(df[i]))
        results.append(result)
    return [], "\nadf检测的结果为：{}\n".format(results)

def CBLOF(data=None, taskid="", idx=0, date_name=None):
    """
    基于聚类的异常检测方法
    
    Args:
        data: DataFrame数据
        taskid: 任务ID
        idx: 数据集索引
        date_name: 日期列名，如果为None则使用索引
    """
    def find_large_small_clusters(cluster_sizes, alpha=0.9, beta=5):
        # 统计每个簇的大小
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

        total_size = sum(cluster_sizes.values())

        # 绝对多数原则
        large_clusters = []
        cumulative_size = 0
        for cluster, size in sorted_clusters:
            cumulative_size += size
            large_clusters.append(cluster)
            if cumulative_size / total_size >= alpha:
                break

        # 突降原则
        for i in range(1, len(sorted_clusters)):
            prev_size = sorted_clusters[i - 1][1]
            curr_size = sorted_clusters[i][1]
            if prev_size / curr_size >= beta:
                large_clusters = [cluster for cluster, _ in sorted_clusters[:i]]
                break

        small_clusters = [cluster for cluster in cluster_sizes.keys() if cluster not in large_clusters]

        return large_clusters, small_clusters

    # 获取数值列
    df = data
    if date_name:
        df_drop_date = df.drop(date_name, axis=1)
        numeric_columns = df_drop_date.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_values_2d = np.atleast_2d(df[numeric_columns].values)

     # 处理日期列和x轴数据
    if date_name and date_name in df.columns:
        if pd.api.types.is_integer_dtype(df[date_name]):
            x_values = df[date_name]
            x_label = date_name
            use_date = False
        else:
            try:
                df[date_name] = pd.to_datetime(df[date_name])
                x_values = df[date_name]
                x_label = 'Year-Month'
                use_date = True
            except:
                x_values = df[date_name]
                x_label = date_name
                use_date = False
    else:
        # 如果date_name不存在或为None
        x_values = np.arange(len(df))
        x_label = 'Index'
        use_date = False

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(df_values_2d)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = Counter(labels)

    # 筛选大小簇
    large_clusters, small_clusters = find_large_small_clusters(cluster_sizes)

    # 计算CBLOF值
    cblof_scores = np.zeros(len(df_values_2d))
    results_str = ""

    for i, (x, label) in enumerate(zip(df_values_2d, labels)):
        if label in large_clusters:
            cblof_scores[i] = cluster_sizes[label] * np.linalg.norm(x - cluster_centers[label])
        else:
            if not np.any(large_clusters):
                cblof_scores[i] = 0
                results_str += "没有大簇，无法计算异常值"
                break
            else:
                distances_to_large_clusters = euclidean_distances([x], cluster_centers[large_clusters]).min()
                cblof_scores[i] = cluster_sizes[label] * distances_to_large_clusters

    # 设置CBLOF阈值
    cblof_threshold = np.percentile(cblof_scores, 95)
    outliers = cblof_scores > cblof_threshold
    # 设置CBLOF阈值并获取前10个最显著的异常点
    outliers = np.zeros(len(cblof_scores), dtype=bool)
    # 获取按分数降序排序的索引
    top_indices = np.argsort(cblof_scores)[::-1][:10]  # 取分数最高的前10个点
    outliers[top_indices] = True
        
    sns.set_theme(style="whitegrid",
                  font=['sans-serif', 'SimHei'],
                  rc={'axes.unicode_minus': False})
    # 使用 Seaborn 提供的 "deep" 配色方案，提供柔和且区分度高的颜色
    colors = sns.color_palette("deep", n_colors=df.shape[1])  # "deep" 配色方案

    # 创建图形
    plt.figure(figsize=(9, 6))
    
    # 在函数开始时设置主题和字体
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置备选字体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 绘制CBLOF分数
    plt.plot(x_values, cblof_scores, label='CBLOF Scores', color=colors[0])
    # 标记异常点
    plt.scatter(x_values[outliers], cblof_scores[outliers], 
               color='red', marker='x', s=50, label='Outliers')

    plt.title('CBLOF Scores Over Time', fontsize=title_font_size, weight='bold')
    plt.ylabel('CBLOF Score', fontsize=label_font_size)

    # 设置x轴格式
    if use_date:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    plt.xlabel(x_label, fontsize=label_font_size)
    plt.legend(fontsize=legend_font_size)  # 设置图例字体大小
    plt.tick_params(axis='both', labelsize=tick_font_size)  # 设置坐标轴刻度的字体大小
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)  # 设置网格线样式和透明度
    plt.tight_layout()
    plt.autoscale(enable=True, axis='both', tight=True)

    # 调整y轴范围，在最大值上方留出20%的空间
    y_max = max(cblof_scores)
    plt.ylim(0, y_max * 1.1)  # 上限设为最大值的1.2倍

    # 保存图像
    task_plot_path = 'images/{}'.format(taskid)
    if not os.path.exists(task_plot_path):
        os.mkdir(task_plot_path)
    
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    trend_path = os.path.join(task_plot_path, 'CBLOF_{}.png'.format(current_time))
    plt.savefig(trend_path)
    plt.close()

    # 添加异常点的信息到结果字符串
    if date_name:
        results_str += f"经过检测，存在异常的日期为：{df[date_name][outliers].tolist()}"
    else:
        results_str += f"经过检测，存在异常的索引位置为：{np.where(outliers)[0].tolist()}"

    return [trend_path], results_str
    

# if __name__=="__main__":
#     arima(data_path="./Tools/Finance_EDA/date_average_value.csv")
