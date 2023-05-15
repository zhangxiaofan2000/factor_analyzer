# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/10 22:12
# File : factor_analyzer.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体或者楷体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 定义计算公司评分的函数
def calculate_company_scores(data, n_factors=3):
    """
    Calculate 基于factor analysis的每家公司的得分 .
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 3:])
    # 使用因子分析确定因子数量
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_data)

    # 计算每个公司在每个因子上的得分
    factor_scores = fa.transform(scaled_data)
    factor_scores_df = pd.DataFrame(factor_scores, columns=[f"Factor {i}" for i in range(1, n_factors+1)])
    result_df = pd.concat([data[['公司名称']], factor_scores_df], axis=1)

    # 对每个公司的分数进行标准化并计算总分
    normalized_scores = result_df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    normalized_scores["Total Score"] = normalized_scores.sum(axis=1)
    ranked_df = pd.concat([result_df[['公司名称']], normalized_scores[['Total Score']]], axis=1)

    # 按总分降序排序
    ranked_df = ranked_df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)

    return ranked_df

def calculate_factor_contributions(data, n_factors=3):
    """
    计算每个因子对数据的贡献度

    参数：
    data: 包含数据的 pandas.DataFrame 对象
    n_factors: int，指定使用的因子数量，默认为 3

    返回：
    pandas.DataFrame，包含每个因子对数据的贡献度
    """
    # 对数据进行标准化处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 3:])
    # 使用因子分析确定因子数量
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_data)
    # 计算每个因子的贡献度
    factor_loadings = fa.loadings_
    factor_contributions = factor_loadings ** 2  # 计算因子载荷的平方
    factor_contributions = factor_contributions / factor_contributions.sum(axis=0, keepdims=True)  # 对每列求和并归一化
    factor_contributions = pd.DataFrame(factor_contributions,index=data.columns[3:],
                                        columns=[f"因子 {i}" for i in range(1, n_factors + 1)])
    # 按总分降序排序
    factor_contributions['总贡献'] = factor_contributions.sum(axis=1)  # 计算每个因子的总分
    factor_contributions.sort_values(by='总贡献', ascending=False, inplace=True)  # 按总分降序排序
    return factor_contributions


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import pandas as pd

def factor_analysis_fit(df,index):
    # 选择分析变量并进行因子分析
    # ...
    # 对数据进行标准化处理
    scaler = StandardScaler()
    df = scaler.fit_transform(df.iloc[:, 3:])
    # 适度性检验
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    kmo_value, kmo_model = calculate_kmo(df)


    result = {'year':index,'Chi-Square Value': chi_square_value, 'P-value': p_value, 'KMO Value': np.mean(kmo_value)}
    return result
    # 返回检验数据





# 获取所有需要处理的Excel文件
data_folder = "./data/最终数据/"
files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".xlsx")]
results = []

# 遍历每个Excel文件进行处理并可视化
for file in files:
    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]

    # 进行因子分析适度性检验
    result = factor_analysis_fit(data, year)
    results.append(result)

# 合并所有结果
merged_result = {}
for key in results[0].keys():
    merged_result[key] = np.array([result[key] for result in results]).flatten()
df = pd.DataFrame(merged_result)
df.to_excel('./data/检验.xlsx')


# 统计每年大于0和小于0的个数
positive_counts = []
negative_counts = []
mean_scores = []
years = []
for file in files:
    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]
    # 计算公司评分
    years.append(year)
    ranked_df = calculate_company_scores(data)
    positive_count = (ranked_df["Total Score"] > 0).sum().sum()
    negative_count = (ranked_df["Total Score"] < 0).sum().sum()
    positive_counts.append(positive_count)
    negative_counts.append(negative_count)
    mean_scores.append(ranked_df["Total Score"].mean())


    # 可视化
    plt.figure(figsize=(20, 9),dpi=400)
    # plt.yticks(fontsize=8)
    sns.barplot(data=ranked_df, y='Total Score', x='公司名称', palette='Blues_d')
    plt.title(f"{year}区块链概念上市公司经营绩效评估分数")
    plt.xlabel("公司名称")
    plt.ylabel("总分数")
    plt.tight_layout()
    plt.xticks(rotation=90, ha='right', rotation_mode='anchor')

    plt.savefig(f'./data/绩效评分图/{year}区块链概念上市公司经营绩效评估分数.png',bbox_inches='tight')

# 绘制折线图
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(years,mean_scores,linewidth=3.5)
ax1.set_xlabel('年份')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

ax1.set_ylabel('平均分数',)
ax1.tick_params(axis='y')
plt.show()

fig2, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
ax2.plot(years,positive_counts, color='green', label='综合评分大于0的公司个数')
ax2.plot(years,negative_counts, color='red', label='综合评分小于0的公司个数')
ax2.set_ylabel('数量', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend()
plt.show()
print(positive_counts)
print(negative_counts)

for file in files:
    # 读取数据
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]
    # 计算因子贡献
    ranked_df = calculate_factor_contributions(data)
    ranked_df.to_excel(f'./data/因子贡献度/{year}_因子贡献度.xlsx')
