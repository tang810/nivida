import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

def sde_simulation_3d():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示

    # 解决绘图中负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 固定种子
    np.random.seed(11)

    # 1. 数据生成
    # 初始化参数
    n = 365
    # 生成日期
    dates = pd.date_range(start='1/1/2023', periods=n)

    # 选择参数 r 和 p
    r_true = 1
    p_true = 0.00004  # 这个概率很小，以产生较大的值

    # 生成系统状态，确保为正整数
    S = np.random.negative_binomial(r_true, p_true, size=n) + 1  # 加1确保最小值为1
    print(np.min(S), np.max(S))
    # 生成批次号，假设批次为 1 到 5 之间的随机整数
    batch = np.random.randint(1, 150, size=n)

    # 创建 DataFrame
    data = pd.DataFrame({
        '日期': dates,
        '系统状态': S,
        '批次': batch
    })

    # 将日期转换为 datetime 类型
    data['日期'] = pd.to_datetime(data['日期'])

    # ========================
    # 第二步：参数估计（布朗运动）
    # ========================

    # 选择第一条实际系统状态数据进行参数估计
    S = data['系统状态'].values
    log_S = np.log(S)

    # 计算对数增长率
    dt = 1  # 时间步长为1天
    log_return = np.diff(log_S)
    time = np.arange(len(log_return))

    # 估计漂移率（mu）和波动率（sigma）
    mu_hat = np.mean(log_return) / dt
    sigma_hat = np.std(log_return) / np.sqrt(dt)

    print(f"估计的漂移率 mu = {mu_hat:.5f}")
    print(f"估计的波动率 sigma = {sigma_hat:.5f}")

    # ========================
    # 第三步：模拟未来系统状态
    # ========================

    # 设置模拟参数
    num_simulations = 1000  # 模拟路径数量
    num_steps = 30  # 模拟未来30天
    S0 = S[-1]  # 当前系统状态的最后一个值

    # 初始化模拟结果数组
    simulated_S = np.zeros((num_steps, num_simulations))
    dt_sim = dt

    # 进行模拟
    for i in range(num_simulations):
        # 生成标准正态随机数
        Z = np.random.normal(size=num_steps)
        # 初始化数组
        S_t = np.zeros(num_steps)
        S_t[0] = S0
        for t_step in range(1, num_steps):
            # 布朗运动的离散化公式
            S_t[t_step] = S_t[t_step - 1] * np.exp(
                (mu_hat - 0.5 * sigma_hat ** 2) * dt_sim + sigma_hat * np.sqrt(dt_sim) * Z[t_step])
        simulated_S[:, i] = S_t

    simulated_S_mean = np.mean(simulated_S, axis=1)

    # 创建时间轴
    future_time = pd.date_range(start='1/1/2024', periods=num_steps)
    # 生成批次号，假设批次为 1 到 5 之间的随机整数
    batch_temp = np.random.randint(1, 150, size=num_steps)
    data_temp = pd.DataFrame({
        '日期': future_time,
        '批次': batch_temp
    })

    # 将日期转换为 datetime 类型
    data_temp['日期'] = pd.to_datetime(data_temp['日期'])

    # 将日期转换为数值格式（ordinal）
    time_past = pd.to_datetime(data['日期']).astype(np.int64) / 10 ** 9
    time_future = pd.to_datetime(data_temp['日期']).astype(np.int64) / 10 ** 9

    # 创建网格
    xi = np.linspace(time_past.min(), time_past.max(), 100)
    xi_future = np.linspace(time_future.min(), time_future.max(), 100)
    yi = np.linspace(data['批次'].min(), data['批次'].max(), 5)
    yi_future = np.linspace(data_temp['批次'].min(), data_temp['批次'].max(), 5)
    X, Y = np.meshgrid(xi, yi)
    X_f, Y_f = np.meshgrid(xi_future, yi_future)

    # 准备插值数据
    points_past = np.array([time_past, data['批次']]).T
    points_future = np.array([time_future, data_temp['批次']]).T
    values_real = data['系统状态']
    values_sim = simulated_S_mean

    # 对真实数据进行插值
    Z_real = griddata(points_past, values_real, (X, Y), method='nearest')
    # 对仿真数据进行插值
    Z_sim = griddata(points_future, values_sim, (X_f, Y_f), method='nearest')

    # 创建时间轴
    time_total_temp = pd.date_range(start='1/1/2023', periods=num_steps + 365)

    # 将日期转换为数值格式（ordinal）
    time_total = pd.to_datetime(time_total_temp).astype(np.int64) / 10 ** 9

    # 绘制真实数据的三维等高线图
    fig = plt.figure(figsize=(18, 6))

    ax = fig.add_subplot(131, projection='3d')
    ax.contour3D(X, Y, Z_real, 50, cmap='viridis')
    ax.set_xlabel('日期', labelpad=30)
    ax.set_ylabel('批次')
    ax.set_zlabel('系统状态', labelpad=12)
    ax.set_title('真实数据的三维等高线图')

    # 设置日期刻度
    xticks_past = np.linspace(time_past.min(), time_past.max(), 5)
    xticks_future = np.linspace(time_future.min(), time_future.max(), 5)
    xticks_total = np.linspace(time_total.min(), time_total.max(), 5)
    ax.set_xticks(xticks_past)
    ax.set_xticklabels([pd.to_datetime(d, unit='s').date() for d in xticks_past], rotation=45, ha='right')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.contour3D(X_f, Y_f, Z_sim, 50, cmap='plasma')
    ax2.set_xlabel('日期', labelpad=30)
    ax2.set_ylabel('批次')
    ax2.set_zlabel('系统状态', labelpad=12)
    ax2.set_title('根据实际数据预测的仿真数据三维等高线图')

    # 设置日期刻度
    ax2.set_xticks(xticks_future)
    ax2.set_xticklabels([pd.to_datetime(d, unit='s').date() for d in xticks_future], rotation=45, ha='right')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.contour3D(X, Y, Z_real, 50, cmap='viridis')
    ax3.contour3D(X_f, Y_f, Z_sim, 50, cmap='plasma')
    ax3.set_xlabel('日期', labelpad=30)
    ax3.set_ylabel('批次')
    ax3.set_zlabel('系统状态', labelpad=12)
    ax3.set_title('实际+仿真数据三维等高线图')

    # 设置日期刻度
    ax3.set_xticks(xticks_total)
    ax3.set_xticklabels([pd.to_datetime(d, unit='s').date() for d in xticks_total], rotation=45, ha='right')
    plt.savefig('./fig/SDE_3D.png', dpi=600)
    plt.show()
