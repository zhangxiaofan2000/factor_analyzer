# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/19 13:37
# File : cluster_result_step12.py
import os

import numpy as np
import pandas as pd
from scipy import stats

import pandas as pd
from scipy import stats


def analyze_cluster_results(cluster_data, cluster_column, metric_columns):
    """
    对聚类结果进行统计分析和验证，并将结果存入Excel文件

    参数:
        cluster_data (DataFrame): 包含聚类结果的数据框
        cluster_column (str): 聚类簇别所在的列名
        metric_columns (list): 需要分析的指标列名列表

    返回:
        results (DataFrame): 包含每个簇的平均指标值、方差、F-value和p-value的结果表格
    """
    results = []

    for column in metric_columns:
        # 计算每个簇的平均指标值
        mean_values = cluster_data.groupby(cluster_column)[column].mean()

        # 计算每个簇的方差
        variance_values = cluster_data.groupby(cluster_column)[column].var()
        clusters2 = [cluster_data[cluster_data[cluster_column] == cluster][column] for cluster in cluster_data[cluster_column].unique()]
        f_value, p_value = stats.f_oneway(*clusters2)

        # 进行方差分析
        clusters = cluster_data[cluster_column].unique()
        for cluster in clusters:
            cluster_data_subset = cluster_data[cluster_data[cluster_column] == cluster]

        # 将每个类别的结果添加到结果列表中
        for i in range(len(clusters)):
            results.append([column,clusters[i], mean_values[i], variance_values[i], f_value, p_value])

    # 创建结果数据框
    results_df = pd.DataFrame(results, columns=['指标列名','聚类类别', '平均指标值', '方差', 'F-value', 'p-value'])

    # 存入Excel文件

    return results_df


data_folder = "./data/聚类结果/"
files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".xlsx")]
res = []
for i, file in enumerate(files):

    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]

    # 假设你已经完成了因子分析并得到了每个公司的因子得分，存储在名为ranked_df的DataFrame中
    # 选择用于聚类的因子得分列
    metric_columns = ['Factor 1', 'Factor 2', 'Factor 3']  # 根据你的实际因子列名进行替换
    # 指定聚类结果列名、需要分析的指标列列表
    cluster_column = '聚类结果'

    # 调用函数进行统计分析和保存结果至Excel文件
    results = analyze_cluster_results(data, cluster_column, metric_columns)
    results = results.sort_values(by=['指标列名','聚类类别'], ascending=True).reset_index(drop=True)

    results.to_excel(f'./data/聚类结果分析/{year}_{i}聚类结果统计分析.xlsx', index=False)

