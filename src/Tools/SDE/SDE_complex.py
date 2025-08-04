import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

def sde_complex():
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

    # 计算模拟均值和置信区间
    mean_simulated_S = np.mean(simulated_S, axis=1)
    std_simulated_S = np.std(simulated_S, axis=1)
    # upper_bound = mean_simulated_S + 1.96 * std_simulated_S
    # lower_bound = mean_simulated_S - 1.96 * std_simulated_S
    upper_bound = mean_simulated_S + 0.2 * std_simulated_S
    lower_bound = mean_simulated_S - 0.2 * std_simulated_S
    lower_bound[lower_bound < 0] = 1
    # ========================
    # 第四步：结果可视化
    # ========================

    # 创建时间轴
    historical_time = pd.to_datetime(data['日期'])
    future_time = historical_time.iloc[-1] + pd.to_timedelta(np.arange(1, num_steps + 1), unit='D')

    # 绘制实际数据
    plt.figure(figsize=(12, 6))

    S_actual = data['系统状态'].values
    plt.plot(historical_time, S_actual, label=f'实际系统状态')

    # 绘制模拟均值
    plt.plot(future_time, mean_simulated_S, label='模拟均值', linestyle='--', color='black')

    # 绘制置信区间
    plt.fill_between(future_time, lower_bound, upper_bound, color='orange', alpha=0.3, label='置信区间')

    # 添加SDE公式
    sde_formula = r'$dS_t = \mu S_t dt + \sigma S_t dW_t$'
    plt.text(0.05, 0.95, sde_formula, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

    # 图形设置
    plt.xlabel('日期')
    plt.ylabel('系统状态')
    plt.title('系统状态的实际数据与SDE模拟')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('./fig/SDE_complex.png', dpi=600)
    plt.show()
