# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/11 22:47
# File : Get_Data.py
import os

import pandas as pd
from tqdm import tqdm

columns = ['股票代码', '公司名称', '时间', '营业收入', '归属于上市公司股东的净利润', '归属于上市公司股东的扣除非经常性损益的净利润',
           '经营活动产生的现金流量净额', '基本每股收益', '稀释每股收益', '加权平均净资产收益率', '总资产', '归属于上市公司股东的净资产',
           '销售费用', '管理费用', '财务费用', '研发费用', '货币资金', '应收账款']

df = pd.DataFrame(columns=columns)


path = r"G:\PythonProject\factor_analyzer\data\word"
files_1 = [f for f in os.listdir(path) if f.endswith('.docx')]

# 遍历所有文件
for file in tqdm(files_1):
    # 提取文件名中的股票代码、时间和公司名称
    file_parts = file.split('_')
    stock_code = file_parts[0]
    year = file_parts[1]
    company_name = file_parts[2].split('.')[0]

    # 将信息添加到DataFrame中
    df = df.append({'股票代码': stock_code,
                    '时间': year,
                    '公司名称': company_name}, ignore_index=True)
df.to_excel("./data/data.xlsx")
