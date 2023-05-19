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
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_)
    ax.set_xlabel(factors[0])
    ax.set_ylabel(factors[1])
    ax.set_zlabel(factors[2])
    ax.set_title('公司聚类结果')

    # 保存图表
    plt.savefig('cluster_plot.png')
    plt.savefig(f'./data/聚类结果/{year}Kmeans聚类结果.png')

    # 返回包含公司名称、因子得分和聚类结果的DataFrame
    return data


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
    plt.savefig(f'./data/肘部图/{year}区块链上市公司因子得分肘部图.png',bbox_inches='tight')

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def cluster_companies2(data, factors):
    """
    对公司进行聚类

    参数:
    data (DataFrame): 包含公司数据和因子得分的DataFrame
    factors (list): 要用于聚类的因子得分列名列表

    返回:
    DataFrame: 包含公司名称、因子得分和聚类结果的DataFrame
    """

    # 提取选择的因子得分列作为聚类输入
    X = data[factors].values

    # 使用系统聚类法进行聚类
    Z = linkage(X, method='ward')

    # 绘制聚类树状图（谱系图）
    plt.figure(figsize=(10, 6))
    dendrogram(Z)

    # 创建2列3行的图表
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    plt.xlabel('公司')
    plt.ylabel('聚类距离')
    plt.title(f'{year}公司聚类谱系图')
    plt.xticks(rotation=90)
    # 保存图表
    plt.savefig(f'./data/聚类结果/{year}系统聚类法聚类结果.png',bbox_inches='tight')
    # 返回包含公司名称、因子得分和聚类结果的DataFrame
    return data





data_folder = "./data/加入因子得分与排名后的最终数据/"
files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".xlsx")]
# 创建2列3行的图表
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, file in enumerate(files):
    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]

    # 假设你已经完成了因子分析并得到了每个公司的因子得分，存储在名为ranked_df的DataFrame中
    # 选择用于聚类的因子得分列
    factors = ['Factor 1', 'Factor 2', 'Factor 3']  # 根据你的实际因子列名进行替换
    # 设置聚类数量（你可以根据需求自行调整）
    n_clusters = 5


    # # 设置最大聚类数量
    # max_clusters = 15
    #
    # # 绘制肘部图
    # plot_elbow_curve(data, factors, max_clusters)

    # 对公司进行聚类
    clustered_df = cluster_companies(data, factors, n_clusters)
    clustered_df = clustered_df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)

    file= file.split('/')[-1]
    clustered_df.to_excel(f'G:\\PythonProject\\factor_analyzer\\data\\聚类结果\\kmeans{file}',index=False)

    # 对公司进行聚类
    clustered_df = cluster_companies2(data, factors)
    clustered_df = clustered_df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
    # 在指定的子图位置绘制聚类图
    ax = axes[i // 3, i % 3]
    plt.sca(ax)  # 设置当前子图
    cluster_companies(data, factors)
    # 设置子图标题
    ax.set_title(f'{year}公司聚类谱系图')



    file= file.split('/')[-1]
    clustered_df.to_excel(f'G:\\PythonProject\\factor_analyzer\\data\\聚类结果\\系统聚类{file}',index=False)

# 调整子图之间的间距
plt.tight_layout()

# 保存图表
plt.savefig('./data/聚类结果/系统聚类法聚类结果.png')