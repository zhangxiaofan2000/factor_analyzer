# -*- coding: utf-8 -*-
# Auther : ZhangYiLong
# Mail : 503302425@qq.com
# Date : 2023/5/11 21:42
# File : rename_word.py
import os
import concurrent.futures
from tqdm import tqdm

import shutil
import pandas as pd

# 文件夹路径
folder_path = r'.\data\word'
df = pd.read_excel(r".\data\区块链公司.xlsx")

def process_file(filename):
    try:
        # 获取文件名
        name, ext = os.path.splitext(filename)
        # 判断第一个减号后面是否是数字
        parts = name.split('_')
        if len(parts) > 1 and not parts[1].isdigit():
            # 修改文件名
            new_name = parts[0] + '_2022' + ext
            # 复制Word文件并修改文件名
            file_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            shutil.copyfile(file_path, new_path)
            os.remove(file_path)
    except:
        print("出现错误："+filename)
    return filename
def process_file2(filename):
    try:
        # 获取股票代码
        stock_code = filename[:6]
        # 获取公司名称
        company_name = df[df['股票代码'].str.contains(stock_code)]['股票简称'].iloc[0].replace("*","")
        # 组合新文件名
        new_name = filename[:-5] + '_' + company_name + '.docx'
        # 重命名Word文件
        file_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        shutil.copyfile(file_path, new_path)
        os.remove(file_path)
    except:
        print("出现错误："+filename)
    return filename

# 遍历文件夹中的所有Word文档
# 获取所有文件名
files = os.listdir(folder_path)

with concurrent.futures.ThreadPoolExecutor() as executor:
    # 并发执行任务
    results = list(tqdm(executor.map(process_file, files), total=len(files)))

with concurrent.futures.ThreadPoolExecutor() as executor:
    # 并发执行任务
    results2 = list(tqdm(executor.map(process_file2, files), total=len(files)))