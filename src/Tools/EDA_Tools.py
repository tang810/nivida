"""
EDA 工具模块 - BytesIO 优化版本
所有函数统一返回 (List[io.BytesIO], str) 格式
"""

import sys
import os
import time 
import io
import math
from collections import Counter
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf, adfuller
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# 统一图像风格配置
title_font_size = 12
label_font_size = 12
tick_font_size = 10
legend_font_size = 10

# 字体配置
font_path = os.path.join('fonts', 'SimHei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)


def _save_figure_to_bytesio() -> io.BytesIO:
    """将当前matplotlib图形保存到BytesIO对象"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()  # 关闭当前图形释放内存
    return buf


def fft_periodic(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    使用傅里叶变换分析周期性
    
    Returns:
        (List[BytesIO], str): 图片列表和分析结果文本
    """
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行傅里叶分析"
    
    # 确定子图布局
    num_columns = len(numeric_columns)
    nrows = math.ceil(num_columns / 2)
    ncols = 2 if num_columns > 1 else 1
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 6), dpi=100, sharex=True)
    
    # 处理axes形状
    if num_columns == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    # 绘制每个数值列的频域分析
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        fft_result = np.fft.fft(df[column])
        frequencies = np.fft.fftfreq(len(df[column]), d=1)
        magnitude = np.abs(fft_result)
        
        ax.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2], label=column)
        ax.set_title(f'{column} Frequency Domain')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.legend()
    
    # 删除多余的空白子图
    if num_columns % 2 == 1 and num_columns > 1:
        fig.delaxes(axes[-1])
    
    # 设置x轴标签
    if num_columns == 1:
        axes[0].set_xlabel('Frequency')
    else:
        last_row_start = (nrows - 1) * ncols
        for i in range(max(0, last_row_start), min(len(axes), num_columns)):
            axes[i].set_xlabel('Frequency')
    
    plt.tight_layout()
    
    return [_save_figure_to_bytesio()], f"完成了{len(numeric_columns)}个数值列的频域分析"


def total_trend(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    绘制总体趋势图，只对数值型特征绘制子图
    
    Returns:
        (List[BytesIO], str): 图片列表和分析结果文本
    """
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行趋势分析"
    
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
        x_values = np.arange(len(df))
        x_label = 'Index'
        use_date = False
    
    # 确定子图布局
    num_columns = len(numeric_columns)
    nrows = math.ceil(num_columns / 2)
    ncols = 2 if num_columns > 1 else 1
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, nrows * 2), dpi=100)
    
    # 设置主题和字体
    sns.set_theme(style="whitegrid", font=['sans-serif', 'SimHei'], rc={'axes.unicode_minus': False})
    colors = sns.color_palette("deep", n_colors=num_columns)
    
    # 处理axes形状
    if num_columns == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    # 绘制每个数值列的趋势
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        ax.plot(x_values, df[column], label=column, linewidth=2, marker='o', markersize=4, color=colors[i])
        ax.set_title(f'{column} Trend', fontsize=title_font_size, weight='bold')
        ax.set_ylabel('Values', fontsize=label_font_size)
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=legend_font_size)
        ax.tick_params(axis='both', labelsize=tick_font_size)
    
    # 删除多余的空白子图
    if num_columns % 2 == 1 and num_columns > 1:
        fig.delaxes(axes[-1])
        axes = axes[:-1]
    
    # 设置x轴标签
    if num_columns == 1:
        axes[0].set_xlabel(x_label, fontsize=label_font_size)
        if use_date:
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    else:
        last_row_start = (nrows - 1) * ncols
        for i in range(num_columns):
            if i >= last_row_start:
                axes[i].set_xlabel(x_label, fontsize=label_font_size)
                if use_date:
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[i].set_xticklabels([])
                axes[i].set_xlabel('')
    
    plt.tight_layout()
    
    return [_save_figure_to_bytesio()], f"完成了{len(numeric_columns)}个数值列的趋势分析"


def acf_and_pacf(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    计算ACF和PACF值
    
    Returns:
        (List[BytesIO], str): 空图片列表和分析结果文本
    """
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行ACF/PACF分析"
    
    lags = min(12, len(df) // 4)  # 防止lags过大
    results = []
    
    for column in numeric_columns:
        try:
            acf_values = acf(df[column], nlags=lags, fft=True)
            pacf_values = pacf(df[column], nlags=lags)
            results.append(f"{column}列的ACF值: {acf_values[:5]}...")  # 只显示前5个值
            results.append(f"{column}列的PACF值: {pacf_values[:5]}...")
        except Exception as e:
            results.append(f"{column}列的ACF/PACF计算失败: {e}")
    
    return [], "\n".join(results)


def z_score(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    Z-score异常检测和箱线图分析
    
    Returns:
        (List[BytesIO], str): 图片列表和分析结果文本
    """
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行Z-score分析"
    
    num_columns = len(numeric_columns)
    fig, axes = plt.subplots(nrows=math.ceil(num_columns / 2), ncols=2 if num_columns > 1 else 1, 
                            figsize=(12, 6), dpi=100)
    
    if num_columns == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    results_text = []
    threshold = 2.7
    
    # 绘制每个数值列的箱线图和异常检测
    for i, column in enumerate(numeric_columns):
        mean_val = np.mean(df[column])
        std_val = np.std(df[column])
        z_scores = (df[column] - mean_val) / std_val
        outliers_indices = np.abs(z_scores) > threshold
        outlier_count = np.sum(outliers_indices)
        
        ax = axes[i]
        ax.boxplot(df[column], vert=True, patch_artist=True, showmeans=True)
        ax.set_title(f'{column} 箱线图 (异常点: {outlier_count}个)')
        ax.set_xticks([])
        ax.grid(True)
        
        results_text.append(f"{column}: 发现{outlier_count}个异常点 (|Z-score| > {threshold})")
        
        if date_name and date_name in df.columns and outlier_count > 0:
            outlier_indices = np.where(outliers_indices)[0]
            if len(outlier_indices) <= 5:  # 只显示前5个异常点
                outlier_dates = df[date_name].iloc[outlier_indices].tolist()
                outlier_values = df[column].iloc[outlier_indices].tolist()
                results_text.append(f"  异常点详情: {list(zip(outlier_dates, outlier_values))}")
    
    # 删除多余子图
    if num_columns % 2 == 1 and num_columns > 1:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    return [_save_figure_to_bytesio()], "\n".join(results_text)


def adf_detection(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    ADF平稳性检测
    
    Returns:
        (List[BytesIO], str): 空图片列表和检测结果文本
    """
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行ADF检测"
    
    results = []
    for column in numeric_columns:
        try:
            adf_result = adfuller(df[column])
            p_value = adf_result[1]
            is_stationary = p_value < 0.05
            status = "平稳" if is_stationary else "非平稳"
            results.append(f"{column}: {status} (p-value: {p_value:.4f})")
        except Exception as e:
            results.append(f"{column}: ADF检测失败 - {e}")
    
    return [], "ADF平稳性检测结果:\n" + "\n".join(results)


def CBLOF(data: pd.DataFrame, date_name: Optional[str] = None) -> Tuple[List[io.BytesIO], str]:
    """
    基于聚类的异常检测方法
    
    Returns:
        (List[BytesIO], str): 图片列表和分析结果文本
    """
    def find_large_small_clusters(cluster_sizes, alpha=0.9, beta=5):
        """区分大小簇"""
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        total_size = sum(cluster_sizes.values())
        
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
    
    df = data
    
    # 获取数值列
    if date_name and date_name in df.columns:
        numeric_columns = df.drop(date_name, axis=1).select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        return [], "没有找到数值列进行CBLOF分析"
    
    df_values_2d = df[numeric_columns].values
    if df_values_2d.shape[0] < 20:
        return [], "数据量太少，无法进行有效的聚类分析"
    
    # 处理日期列和x轴数据
    if date_name and date_name in df.columns:
        if pd.api.types.is_integer_dtype(df[date_name]):
            x_values = df[date_name]
            x_label = date_name
            use_date = False
        else:
            try:
                df_copy = df.copy()
                df_copy[date_name] = pd.to_datetime(df_copy[date_name])
                x_values = df_copy[date_name]
                x_label = 'Year-Month'
                use_date = True
            except:
                x_values = df[date_name]
                x_label = date_name
                use_date = False
    else:
        x_values = np.arange(len(df))
        x_label = 'Index'
        use_date = False
    
    # 聚类分析
    n_clusters = min(20, len(df_values_2d) // 3)  # 动态调整簇数量
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_values_2d)
    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = Counter(labels)
    
    # 筛选大小簇
    large_clusters, small_clusters = find_large_small_clusters(cluster_sizes)
    
    # 计算CBLOF分数
    cblof_scores = np.zeros(len(df_values_2d))
    
    for i, (x, label) in enumerate(zip(df_values_2d, labels)):
        if label in large_clusters:
            cblof_scores[i] = cluster_sizes[label] * np.linalg.norm(x - cluster_centers[label])
        else:
            if large_clusters:
                distances = euclidean_distances([x], cluster_centers[large_clusters])
                cblof_scores[i] = cluster_sizes[label] * distances.min()
            else:
                cblof_scores[i] = 0
    
    # 获取前10个最异常的点
    top_indices = np.argsort(cblof_scores)[::-1][:10]
    outliers = np.zeros(len(cblof_scores), dtype=bool)
    outliers[top_indices] = True
    
    # 绘图
    sns.set_theme(style="whitegrid", font=['sans-serif', 'SimHei'], rc={'axes.unicode_minus': False})
    colors = sns.color_palette("deep", n_colors=2)
    
    plt.figure(figsize=(9, 6))
    plt.plot(x_values, cblof_scores, label='CBLOF Scores', color=colors[0])
    plt.scatter(x_values[outliers], cblof_scores[outliers], 
               color='red', marker='x', s=50, label='Top 10 Outliers')
    
    plt.title('CBLOF 异常检测分析', fontsize=title_font_size, weight='bold')
    plt.ylabel('CBLOF Score', fontsize=label_font_size)
    plt.xlabel(x_label, fontsize=label_font_size)
    
    if use_date:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    
    plt.legend(fontsize=legend_font_size)
    plt.tick_params(axis='both', labelsize=tick_font_size)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # 生成结果文本
    results_text = f"CBLOF聚类异常检测完成，共发现{len(top_indices)}个显著异常点"
    if date_name and date_name in df.columns:
        outlier_info = []
        for idx in top_indices[:5]:  # 只显示前5个
            date_val = x_values.iloc[idx] if hasattr(x_values, 'iloc') else x_values[idx]
            score = cblof_scores[idx]
            outlier_info.append(f"  {date_val}: CBLOF={score:.2f}")
        if outlier_info:
            results_text += "\n前5个异常点:\n" + "\n".join(outlier_info)
    
    return [_save_figure_to_bytesio()], results_text