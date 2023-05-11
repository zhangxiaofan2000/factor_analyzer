# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/10 22:12
# File : factor_analyzer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

# 生成假数据
np.random.seed(123)
data = pd.DataFrame(np.random.rand(100, 6),
                    columns=['company', 'revenue', 'profit', 'market_share',
                             'customer_satisfaction', 'employee_satisfaction'])

# 对除公司名称外的数据进行标准化处理
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, 1:])

# 使用因子分析确定因子数量
n_factors = 3
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(scaled_data)

# 计算每个公司在每个因子上的得分
factor_scores = fa.transform(scaled_data)
factor_scores_df = pd.DataFrame(factor_scores, columns=[f"Factor {i}" for i in range(1, n_factors+1)])
result_df = pd.concat([data[['company']], factor_scores_df], axis=1)

# 对每个因子得分进行归一化，使得不同因子对排名的影响权重相同
normalized_scores = result_df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
normalized_scores["Total Score"] = normalized_scores.sum(axis=1)
ranked_df = pd.concat([result_df[['company']], normalized_scores[['Total Score']]], axis=1)
ranked_df = ranked_df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)

print(ranked_df)
