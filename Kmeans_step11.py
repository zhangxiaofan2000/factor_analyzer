# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/19 9:46
# File : Kmeans_step11.py
# 获取所有需要处理的Excel文件
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体或者楷体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def cluster_companies(data, factors, n_clusters):
    """
    对公司进行聚类

    参数:
    data (DataFrame): 包含公司数据和因子得分的DataFrame
    factors (list): 要用于聚类的因子得分列名列表
    n_clusters (int): 聚类数量

    返回:
    DataFrame: 包含公司名称、因子得分和聚类结果的DataFrame
    """

    # 提取选择的因子得分列作为聚类输入
    X = data[factors].values

    # 使用K-means算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # 将聚类结果添加到数据中
    data['聚类结果'] = kmeans.labels_

    # 可视化聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.xlabel(factors[0])
    plt.ylabel(factors[1])
    plt.title('公司聚类结果')

    # 保存图表
    plt.savefig('cluster_plot.png')

    # 返回包含公司名称、因子得分和聚类结果的DataFrame
    return data[['公司名称'] + factors + ['聚类结果']]

def plot_elbow_curve(data, factors, max_clusters):
    """
    绘制肘部图

    参数:
    data (DataFrame): 包含公司数据和因子得分的DataFrame
    factors (list): 要用于聚类的因子得分列名列表
    max_clusters (int): 最大聚类数量

    返回:
    None
    """

    # 提取选择的因子得分列作为聚类输入
    X = data[factors].values

    # 存储不同聚类数量下的评估指标
    inertia = []

    # 计算不同聚类数量下的SSE（平方和误差）
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # 绘制肘部图
    plt.figure()
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('聚类数量')
    plt.ylabel('SSE')
    plt.title(f'{year}区块链上市公司因子得分肘部图')
    plt.savefig(f'./data/肘部图/{year}区块链上市公司因子得分肘部图.png')


data_folder = "./data/加入因子得分与排名后的最终数据/"
files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".xlsx")]

for file in files:
    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]

    # 假设你已经完成了因子分析并得到了每个公司的因子得分，存储在名为ranked_df的DataFrame中
    # 选择用于聚类的因子得分列
    factors = ['Factor 1', 'Factor 2', 'Factor 3']  # 根据你的实际因子列名进行替换
    # 设置聚类数量（你可以根据需求自行调整）
    n_clusters = 3


    # # 设置最大聚类数量
    # max_clusters = 15
    #
    # # 绘制肘部图
    # plot_elbow_curve(data, factors, max_clusters)

    # 对公司进行聚类
    clustered_df = cluster_companies(data, factors, n_clusters)
    # 输出聚类结果
    print(clustered_df)

