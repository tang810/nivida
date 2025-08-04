import numpy as np
import matplotlib.pyplot as plt

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
