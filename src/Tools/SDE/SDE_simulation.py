import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

def sde_simulation():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示

    # ========================
    # 第一步：生成假数据
    # ========================

    # 生成日期列表
    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    num_days = 365  # 一年的天数
    date_list = [start_date + timedelta(days=x) for x in range(num_days)]
    date_strings = [date.strftime('%Y-%m-%d') for date in date_list]

    # 设置随机种子
    np.random.seed(42)

    # 定义实际系统状态的数量
    num_actual_states = 3  # 生成3条实际系统状态曲线

    # 初始化数据框列表
    data_list = []

    for i in range(num_actual_states):
        # 生成系统状态（假设随时间线性增长）
        state_mean_start = 50 + i * 5  # 为每条曲线引入微小差异
        state_mean_end = 100 + i * 5
        state_mean = np.linspace(state_mean_start, state_mean_end, num_days)
        state_std = 5  # 标准差
        system_state = np.random.normal(state_mean, state_std)
        system_state[system_state < 1] = 1  # 确保系统状态为正

        # 创建数据框
        data = pd.DataFrame({
            'date': date_strings,
            'system_state': system_state
        })

        data_list.append(data)

    # 显示第一条数据的前5行
    print(data_list[0].head())

    # ========================
    # 第二步：参数估计（布朗运动）
    # ========================

    # 选择第一条实际系统状态数据进行参数估计
    S = data_list[0]['system_state'].values
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
    num_simulations = 100  # 模拟路径数量
    num_steps = 30         # 模拟未来30天
    S0 = S[-1]             # 当前系统状态的最后一个值

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
            S_t[t_step] = S_t[t_step-1] * np.exp((mu_hat - 0.5 * sigma_hat**2) * dt_sim + sigma_hat * np.sqrt(dt_sim) * Z[t_step])
        simulated_S[:, i] = S_t

    # 计算模拟均值和置信区间
    mean_simulated_S = np.mean(simulated_S, axis=1)
    std_simulated_S = np.std(simulated_S, axis=1)
    # upper_bound = mean_simulated_S + 1.96 * std_simulated_S
    # lower_bound = mean_simulated_S - 1.96 * std_simulated_S
    upper_bound = mean_simulated_S + 0.2 * std_simulated_S
    lower_bound = mean_simulated_S - 0.2 * std_simulated_S

    # ========================
    # 第四步：结果可视化
    # ========================

    # 创建时间轴
    historical_time = pd.to_datetime(data_list[0]['date'])
    future_time = historical_time.iloc[-1] + pd.to_timedelta(np.arange(1, num_steps + 1), unit='D')

    # 绘制实际数据
    plt.figure(figsize=(12, 6))

    for idx, data in enumerate(data_list):
        S_actual = data['system_state'].values
        plt.plot(historical_time, S_actual, label=f'实际系统状态 {idx+1}')

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
    plt.legend(loc='center left')
    plt.grid(True)
    plt.savefig('./fig/SDE_linear.png', dpi=600)
    plt.show()
