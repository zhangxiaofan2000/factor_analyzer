# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/14 23:42
# File : get_Data_step2.py

import os

import pandas as pd
from tqdm import tqdm



df = pd.read_excel(r'G:\PythonProject\factor_analyzer\data\data.xlsx')

path = r"G:\PythonProject\factor_analyzer\data\主要会计数据"
files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
column_names = set()

# 遍历所有文件
for file in tqdm(files):
    # 提取文件名中的股票代码、时间和公司名称
    stock_code, time = file.split('_')[:2]
    file_path = os.path.join(path, file)
    file_df = pd.read_excel(file_path, skiprows=1)
    column_names.update(file_df.columns)
with open("column_names.txt", "w") as f:
    for name in column_names:
        f.write(name + "\n")


    # print(stock_code,time)
    # row_idx =  (df['股票代码'] == stock_code) & (df['时间'] == time)
    # row_idx = row_idx[row_idx == True].index[0]
    #
    # print(row_idx)

