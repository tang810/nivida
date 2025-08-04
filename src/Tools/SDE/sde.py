import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from matplotlib import cm
from scipy.stats import norm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties, findSystemFonts, fontManager

font_path = os.path.join('fonts', 'SimHei.ttf')

prop = FontProperties(fname=font_path)
fontManager.addfont(fontpath)
plt.rc('font', family=prop.get_name())

def sbm():
    # 定义参数
    num_paths = 10  # 状态曲线数目
    num_steps = 1000  # 时间步数
    T = 1.0  # 时间区间（年）
    dt = T / num_steps  # 时间间隔
    t = np.linspace(0.0, T, num_steps)  # 时间序列

    # 生成随机数
    np.random.seed(0)  # 设置随机种子，确保结果可复现
    dw = np.random.standard_normal(size=(num_paths, num_steps)) * np.sqrt(dt)  # 标准布朗运动的增量

    # 构建布朗运动路径
    w = np.cumsum(dw, axis=1)  # 沿时间轴累积求和

    # 绘制路径
    plt.figure(figsize=(10, 6))
    for i in range(num_paths):
        plt.plot(t, w[i], label=f'Path {i + 1}')
    plt.title('Standard Brownian Motion Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./fig/SBM.png", dpi=600)
    plt.show()

def sde(data=None,taskid="",idx=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
    # 解决绘图中负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 生成随机数
    np.random.seed(0)  # 设置随机种子，确保结果可复现

    # 参数设置
    mu = 0.5  # 漂移系数
    sigma = 0.2  # 扩散系数
    T = 1.0  # 时间区间（年）
    N = 1000  # 时间步数
    dt = T / N  # 时间步长
    X0 = 1000.0  # 初始条件
    num_paths = 10  # 状态曲线数目


    # 欧拉-马尔科夫方法
    def euler_maruyama(mu, sigma, X0, dt, N):
        X = np.zeros((num_paths, N + 1))
        X[:, 0] = X0
        W = np.zeros((num_paths, N + 1))
        dW = np.random.standard_normal(size=(num_paths, N)) * np.sqrt(dt)  # 标准布朗运动的增量
        W[:, 1:] = np.cumsum(dW, axis=1)  # 累积求和得到布朗运动

        for i in range(N):
            X[:, i + 1] = X[:, i] + mu * X[:, i] * dt + sigma * X[:, i] * dW[:, i]

        return X, W


    # 模拟SDE
    X, W = euler_maruyama(mu, sigma, X0, dt, N)
    # 绘制结果
    plt.figure(figsize=(18, 7))
    plt.subplot(121)
    for i in range(num_paths):
        plt.plot(np.linspace(0, T, N + 1), W[i], label=f'W_{i + 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('随机过程')
    plt.subplot(122)
    for i in range(num_paths):
        plt.plot(np.linspace(0, T, N + 1), X[i], label=f'X_{i + 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('System State')
    plt.title('状态变化')
    plt.tight_layout()

    sde_path = 'images/{}'.format(taskid)
    if not os.path.exists(sde_path):
        os.mkdir(sde_path)
    # 获取当前时间，并格式化为字符串
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    sde_path = os.path.join(sde_path, 'SDE_{}_{}.png'.format(str(idx),current_time))
    plt.savefig(sde_path)
    return "\n\n ### SDE模拟结果：\n", sde_path

def sde_linear(data=None,taskid="",idx=0):
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
    # print(data_list[0].head())

    # ========================
    # 第二步：参数估计（布朗运动）
    # ========================

    # 选择第一条实际系统状态数据进行参数估计
    S = data_list[0]['system_state'].values
    log_S = np.log(S)

    # 计算对数增长率
    dt = 1  # 时间步长为1天
    log_return = np.diff(log_S)
    time_ = np.arange(len(log_return))

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

    sde_path = 'images/{}'.format(taskid)
    if not os.path.exists(sde_path):
        os.mkdir(sde_path)
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    sde_path = os.path.join(sde_path, 'SDE_linear_{}_{}.png'.format(str(idx),current_time))
    plt.savefig(sde_path, dpi=600)

    return "\n\n ### SDE Linear 模拟结果：\n", sde_path

def sde_simulation_3d(data=None,taskid="",idx=0):
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
    df = data
    column_names = df.columns.tolist()
    # 检查是否包含“日期”列
    if '日期' in column_names:
        date_name='日期'
    elif 'date' in column_names:
        # 将'date'列转换为日期格式
        date_name='date'
    data[date_name] = pd.to_datetime(data[date_name])

    # ========================
    # 第二步：参数估计（布朗运动）
    # ========================

    # 选择第一条实际系统状态数据进行参数估计
    S = data['系统状态'].values
    log_S = np.log(S)

    # 计算对数增长率
    dt = 1  # 时间步长为1天
    log_return = np.diff(log_S)
    time_ = np.arange(len(log_return))

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
        date_name: future_time,
        '批次': batch_temp
    })

    # 将日期转换为 datetime 类型
    data_temp[date_name] = pd.to_datetime(data_temp[date_name])

    # 将日期转换为数值格式（ordinal）
    time_past = pd.to_datetime(data[date_name]).astype(np.int64) / 10 ** 9
    time_future = pd.to_datetime(data_temp[date_name]).astype(np.int64) / 10 ** 9

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

    sde_path = 'images/{}'.format(taskid)
    if not os.path.exists(sde_path):
        os.mkdir(sde_path)
    # 获取当前时间，并格式化为字符串
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    sde_path = os.path.join(sde_path, 'SDE_3D_{}_{}.png'.format(str(idx),current_time))
    plt.savefig(sde_path, dpi=600)
    return "\n\n ### SDE 3D 模拟结果：\n", sde_path

def sde_complex(data=None,taskid="",idx=0):
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
    time_ = np.arange(len(log_return))

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
    #plt.savefig('./fig/SDE_complex.png', dpi=600)
    #plt.show()
    sde_path = 'images/{}'.format(taskid)
    if not os.path.exists(sde_path):
        os.mkdir(sde_path)
    # 获取当前时间，并格式化为字符串
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    sde_path = os.path.join(sde_path, 'SDE_complex_{}_{}.png'.format(str(idx),current_time))
    plt.savefig(sde_path, dpi=600)
    return "\n\n ### SDE Complex 模拟结果：\n", sde_path
