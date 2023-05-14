# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/15 0:58
# File : Get_data_step7.py

import os

import pandas as pd
from fuzzywuzzy import process
from tqdm import tqdm
import concurrent.futures

# 全局变量，存储所有文件合并后的数据
df = pd.read_excel(r'G:\PythonProject\factor_analyzer\data\data3.xlsx', dtype={'股票代码': str,'时间': str})

def merge_file(file):
    file_path = os.path.join(path, file)
    # 提取股票代码、时间和公司名称
    stock_code, time, company_name = file.split('_')[:3]
    # 读取文件
    df_file = pd.read_excel(file_path, skiprows=1)
    # 匹配df的行
    row_index = df.index[(df['股票代码'] == stock_code) & (df['时间'] == time) ]
    # 列名模糊匹配阈值
    threshold = 80
    # 匹配列名并存储数据
    for column in df_file.columns:
        matched_column, score = process.extractOne(column, df.columns)
        if score >= threshold:
            if not df_file[column].empty:
                value = df_file[column].iloc[0]
                if not pd.isna(value) and value != '':
                    df.at[row_index, matched_column] = value

path = r"G:\PythonProject\factor_analyzer\data\资产excel"
files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 并发执行任务
    results = list(tqdm(executor.map(merge_file, files), total=len(files)))

df.to_excel('company_data.xlsx')