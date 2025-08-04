import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 获取字体文件的绝对路径
#current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join('fonts', 'SimHei.ttf')

# 添加字体文件
fm.fontManager.addfont(font_path)
# 设置显示负号的字体
matplotlib.rcParams['axes.unicode_minus'] = False
# 如果需要显示中文，可以同时设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

# 统一图像风格
title_font_size = 12
label_font_size = 10
tick_font_size = 10


def boxplot(data,output_path):

    # 获取所有数值列的名字
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # 创建一个箱形图
    plt.figure(figsize=(9, 6))
    # 在函数开始时设置主题和字体
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置备选字体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    sns.set_theme(style="whitegrid",
                  font=['sans-serif', 'SimHei'],
                  rc={'axes.unicode_minus': False})

    sns.boxplot(data=data[numeric_columns], palette="Set3")  # 设置颜色主题
    
    plt.title('Box Plots of All Numeric Columns', fontsize=title_font_size, pad=10, weight='bold')
    plt.xlabel('Data Columns', fontsize=label_font_size)  # X轴标签
    plt.ylabel('Values', fontsize=label_font_size)  # Y轴标签

    plt.xticks(rotation=45, ha='right') # 调整布局以防标签重叠
    plt.tick_params(axis='both', labelsize=tick_font_size)  # 设置坐标轴刻度的字体大小
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)  # 设置网格线样式和透明度
    plt.tight_layout()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()



def heatmap_plot(data, output_path):
    # 获取所有数值列的名字
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 计算相关系数矩阵
    correlation_matrix = data[numeric_columns].corr()
    
    # 设置图形大小
    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(9, 9))
    # 在函数开始时设置主题和字体
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置备选字体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    sns.set_theme(style="whitegrid",
                  font=['sans-serif', 'SimHei'],
                  rc={'axes.unicode_minus': False})
    
    # 绘制热力图
    sns.heatmap(
        correlation_matrix,
        annot=True,          # 显示数值
        cmap='coolwarm',     # 色彩方案
        fmt=".2f",           # 数值格式为两位小数
        square=True,         # 保持方形
        linewidths=0.5,      # 网格线宽度
        cbar_kws={"shrink": .8},  # 调整颜色条大小
        annot_kws={"size": tick_font_size}  # 相关系数字体大小
    )
    
    # 自定义标题
    plt.title('Correlation Heatmap', fontsize=title_font_size, pad=10, weight='bold')
    
    # 设置轴标签字体大小
    plt.xlabel('Features', fontsize=label_font_size)
    plt.ylabel('Features', fontsize=label_font_size)
    
    # 设置刻度标签字体大小和旋转角度
    plt.xticks(rotation=45, ha='right', fontsize=tick_font_size)
    plt.yticks(rotation=0, fontsize=tick_font_size)
    
    # 调整子图参数，确保所有元素都能显示
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    
    # 关闭图形
    plt.close()


def density_plot(data, output_path):
    # 选择所有数值类型的列
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 设置图形风格
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.edgecolor': '.8',
        'axes.linewidth': 1.5
    })
    # 在函数开始时设置主题和字体
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置备选字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    sns.set_theme(style="whitegrid",
                  font=['sans-serif', 'SimHei'],
                  rc={'axes.unicode_minus': False})

    if len(numeric_columns) > 1:
        # 计算子图的行数和列数
        num_plots = len(numeric_columns)
        ncols = min(num_plots, 3)  # 每行最多放置3个子图
        nrows = (num_plots + ncols - 1) // ncols  # 向上取整得到行数

        # 创建一个新的figure对象
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols, 
            # figsize=(ncols * 4, nrows * 3),
            figsize=(9, nrows * 2),
            dpi=100
        )

        # 如果axes不是一个数组，则将其转换为数组
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # 使用 Seaborn 的调色板
        colors = sns.color_palette("Blues", n_colors=num_plots)

        # 绘制密度图
        for i, (column, color) in enumerate(zip(numeric_columns, colors)):
            ax = axes[i]
            # 先绘制填充
            sns.kdeplot(
                data=data,
                x=column,
                ax=ax,
                color=color,
                fill=True,
                alpha=0.5,
                linewidth=0
            )
            # 再绘制黑色线条
            sns.kdeplot(
                data=data,
                x=column,
                ax=ax,
                color='black',  # 使用黑色线条
                fill=False,
                alpha=1,
                linewidth=1    # 设置线条宽度
            )
            
            # 设置每个子图的样式
            ax.set_title(f'Density Distribution of {column}', 
                        fontsize=title_font_size, 
                        pad=10, 
                        weight='bold')
            ax.set_xlabel('Values', fontsize=label_font_size, labelpad=10)
            ax.set_ylabel('Density', fontsize=label_font_size, labelpad=10)
            
            # 设置网格和刻度
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(
                axis='both',
                labelsize=tick_font_size,
                length=5,
                width=1,
                direction='out'
            )

        # 隐藏多余的子图
        for ax in axes[num_plots:]:
            ax.axis('off')

    else:  # 只有一个数值列的情况
        plt.figure(figsize=(9, 6))
        # 先绘制填充
        sns.kdeplot(
            data=data,
            x=numeric_columns[0],
            fill=True,
            color=sns.color_palette("crest")[0],
            alpha=0.5,
            linewidth=0  # 设置为0移除原始线条
        )
        
        # 再绘制黑色线条
        sns.kdeplot(
            data=data,
            x=numeric_columns[0],
            fill=False,
            color='black',  # 使用黑色线条
            alpha=1,
            linewidth=2    # 设置线条宽度
        )

        # 设置图表标题和标签
        plt.title(f'Density Distribution of {numeric_columns[0]}',
                 fontsize=title_font_size,
                 pad=10,
                 weight='bold')
        plt.xlabel('Values', fontsize=label_font_size, labelpad=10)
        plt.ylabel('Density', fontsize=label_font_size, labelpad=10)
        
        # 设置网格和刻度
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tick_params(
            axis='both',
            labelsize=tick_font_size,
            length=5,
            width=1,
            direction='out'
        )

    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(
        output_path,
        dpi=100,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    # 关闭图形
    plt.close()


def violin_plot(data,output_path):
    # 选择所有数值类型的列
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 计算子图的行数和列数
    num_plots = len(numeric_columns)
    ncols = min(num_plots, 3)  # 每行最多放置3个子图
    nrows = (num_plots + ncols - 1) // ncols  # 向上取整得到行数

    # 创建一个新的figure对象
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

    # 如果axes不是一个数组，则将其转换为数组
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    # 绘制小提琴图
    for i, column in enumerate(numeric_columns):
        ax = axes.flat[i]
        sns.violinplot(y=data[column], ax=ax)
        ax.set_title(f'Violin plot of {column}')

    # 隐藏多余的子图
    for ax in axes.flat[num_plots:]:
        ax.axis('off')

    # 调整子图间距
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path)
    plt.close()
