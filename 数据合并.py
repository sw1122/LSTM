import os
import pandas as pd
import numpy as np


Folder_Path = r'D:\Desktop\2018年5MW运行数据\2018年\2018运行数据'  # 原始数据所在的文件夹
SaveFile_Path = r'D:\Desktop\2018年5MW运行数据\2018年\各月数据集合'  # 要保存的文件路径
MergeFile_Name = r'201803-12.csv' #要保存的文件名

os.chdir(Folder_Path)  # 修改当前工作目录
file_list = os.listdir()  # 将该文件夹下的所有文件名存入一个列表

def merge_data():  # 合并数据，并提取特征量保存文件
    dfs = []
    for i in range(0, len(file_list)):  # 循环遍历列表中各个excel文件名，并追加到合并后的文件
        print('合并进度:{}/{}'.format(i + 1, len(file_list)))
        df = pd.read_excel(Folder_Path + '\\' + file_list[i])
        df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)  # 将空值替换成NAL
        df.fillna(method='ffill', inplace=True)  # 向下填充将NAL值填充为上一个值
        if i > 0:
            df.drop([0], inplace=True)  # 去除第一个文件之外所有文件的第一行以整合数据（文件第一行都没有数据）
        df.to_csv(SaveFile_Path + '\\' + MergeFile_Name, encoding="gbk", index=False, header=False, mode='a+')# 存入新的csv中,index,header=False不存储列索引和表头，mode='a+'允许附加

    print('\n已生成' + MergeFile_Name)

def pick_special(): #提取特征量
    df = pd.read_csv(SaveFile_Path + '\\' + MergeFile_Name, index_col=0, encoding='gbk')
    df = df.loc[:, ['发电量', 'DNI', '环境风速', '环境温度', '风向', '热盐罐熔盐均温', '过热器入口熔盐流量', '入口盐温YTE165', '出口过热蒸汽温度WTE302', '出口过热蒸汽压力', '出口过热蒸汽流量']]
    df.to_csv(SaveFile_Path + '\\' + '201803-12特征量.csv', encoding='gbk')  # 存入新的csv中
    print('\n特征提取成功')

if __name__ == '__main__':
    merge_data()   #合并所有数据到一个表格
    pick_special()    #提取特征数
