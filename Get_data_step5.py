# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/15 0:25
# File : Get_data_step5.py
import os

import pandas as pd
from fuzzywuzzy import process
from tqdm import tqdm

# 全局变量，存储所有文件合并后的数据
df = pd.read_excel(r'G:\PythonProject\factor_analyzer\data\data.xlsx')

path = r"G:\PythonProject\factor_analyzer\data\主要会计数据"
def merge_file(file):
    # 提取股票代码、时间和公司名称
    stock_code, time, company_name = file.split('_')[:3]
    # 读取文件
    df_file = pd.read_excel(file, skiprows=1)
    # 匹配df的行
    row_index = df.index[(df['股票代码'] == stock_code) & (df['时间'] == time) & (df['公司名称'] == company_name)]
    # 列名模糊匹配阈值
    threshold = 80
    # 匹配列名并存储数据
    for column in df_file.columns:
        matched_column, score = process.extractOne(column, df.columns)
        if score >= threshold:
            print(matched_column,column)
            print(df_file[column].iloc[0])
            # df.at[row_index, matched_column] = df_file[column].iloc[0]

files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
for file in tqdm(files):
    merge_file(file)
