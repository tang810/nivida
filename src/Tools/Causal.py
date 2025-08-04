"""
CausalInference
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import CausalModel
from pgmpy.estimators import K2Score
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

# 设置随机种子以保证结果可重复
np.random.seed(42)  # For reproducibility

def CausalInference(data=None,taskid="",idx=0):
    results_str = ""
    image_path=[]

    df = data

    # 使用 HillClimbSearch 和 BicScore 来学习贝叶斯网络结构
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=K2Score(df))
    model = BayesianNetwork(best_model.edges())
    # 打印识别的网络结构
    print(best_model.edges())

    G = nx.DiGraph(best_model.edges())
    dot_str = nx.nx_pydot.to_pydot(G).to_string()

    #绘制贝叶斯网络
    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=3000, font_size=12, font_weight='bold', edge_color='gray', arrows=True, arrowsize=20)
    plt.title("Bayesian Network Structure")
    plt.legend()
    
    tmp_img_path = "src/tmp_img_save/Causal_Graph.png"
    image_path.append(tmp_img_path)
    plt.savefig(tmp_img_path,bbox_inches='tight',pad_inches=0.1)
    plt.close()

    # 假设你已经通过结构学习得到了一个因果图 best_model
    for i in range(len(df.columns)):
        for j in range(df.columns):
            if i!=j:
                model = CausalModel(
                        data=df,
                        treatment=df.columns[i],  # 选择你认为可能的干预变量
                        outcome=df.columns[j],  # 神经网络的预测结果
                        graph=dot_str
                        )

                # 识别因果效应
                identified_estimand = model.identify_effect()

                # 估计因果效应
                causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                results_str = f"\n\nCausal Estimate for {df.columns[i]} to {df.columns[j]} is:{causal_estimate.value}\n"
                
    return image_path,results_str 