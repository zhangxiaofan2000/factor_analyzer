# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/19 22:42
# File : get_model_step9.5.py
import os
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler


def calculate_company_scores(data, n_factors=3):
    """
    Calculate 基于factor analysis的每家公司的得分 .
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 3:])
    # 使用因子分析确定因子数量
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_data)
    factor_loadings = fa.loadings_
    n_factors = factor_loadings.shape[1]
    equation = ""
    for i in range(n_factors):
        factor_equation = ""
        for j, loading in enumerate(factor_loadings[:, i]):
            if loading != 0:
                if factor_equation != "":
                    factor_equation += " + "
                factor_equation += f"{loading:.3f}X_{{{j + 1}}}"
        equation += f"Y_{{{i + 1}}} = {factor_equation}"
        if i < n_factors - 1:
            equation += "+ "

        print(equation)
def calculate_company_scores2(data, n_factors=3):
    """
    Calculate 基于factor analysis的每家公司的得分 .
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 3:])
    # 使用因子分析确定因子数量
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_data)

    eigenvalues, _ = fa.get_eigenvalues()
    total_variance = sum(eigenvalues)

    contribution_ratios = eigenvalues / total_variance



    return contribution_ratios

def convert_to_formula(composite_scores):
    formula = "F = "
    for i, score in enumerate(composite_scores):
        formula += f"{score:.5f}F_{{{i + 1}}}"
        if i < len(composite_scores) - 1:
            formula += " + "
    return formula
# 获取所有需要处理的Excel文件
data_folder = "./data/最终数据/"
files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".xlsx")]
results = []
for file in files:
    data = pd.read_excel(file)
    year = os.path.splitext(file)[0][-4:]
    # 计算公司评分
    print('#########')
    print(convert_to_formula(calculate_company_scores2(data, n_factors=3)))


