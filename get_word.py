# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/11 21:13
# File : get_word.py
import concurrent.futures
import os
import pandas as pd
import shutil

from tqdm import tqdm

# 读取Excel文件
df = pd.read_excel(r".\data\区块链公司.xlsx")
stock_code  = list(df["股票代码"].str.split('.').str[0])

# 文件夹路径 年报数据 4w个word
folder_path = r"G:\PythonProject\proprocess\data\repo"

def process_file(filename):
    try:
        file_code = filename[:6]
        # 如果文件名前6个字符满足条件，则将该Word文件复制到新的文件夹中
        if file_code in stock_code:
            # 复制Word文件
            file_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("-","_")
            new_path = os.path.join(r"G:\PythonProject\factor_analyzer\data\word", new_filename)
            shutil.copyfile(file_path, new_path)
    except:
        print(filename)
    return filename
# 遍历文件夹中的所有Word文档
# 获取所有文件名
files = os.listdir(folder_path)

with concurrent.futures.ThreadPoolExecutor() as executor:
    # 并发执行任务
    results = list(tqdm(executor.map(process_file, files), total=len(files)))