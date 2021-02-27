import os
import xlrd
import pandas as pd

Folder_Path = r'D:\Desktop\2018年5MW运行数据\2018年\2018-01'  # 原始数据所在的文件夹
SaveFile_Path = Folder_Path  # 要保存的文件路径，默认和原始数据路径相同
MergeFile_Name = r'文件夹所有文件合并后的数据.csv'

os.chdir(Folder_Path)  # 修改当前工作目录
file_list = os.listdir()  # 将该文件夹下的所有文件名存入一个列表

# if not os.path.exists(r'D:\Desktop\2018年5MW运行数据\2018年\2018-01csv'):  # 判断括号里的文件是否存在的意思，括号内的可以是文件路径。
#     os.makedirs(r'D:\Desktop\2018年5MW运行数据\2018年\2018-01csv')

# def xlsx_to_csv_pd():
#     for name in file_list:
#         print(name)
#         data_xls = pd.read_excel(name, index_col=0)
#         csv_name = name.replace('.xlsx', '.csv')
#         data_xls.to_csv(csv_name, encoding='gbk')
# for name in file_list:
#     csv_name = name.replace('.xlsx', '.csv')
#     print(csv_name)
#     os.rename(name,csv_name)
def merge_data():  # 合并数据
    df = pd.read_csv(Folder_Path + '\\' + file_list[0], encoding='gbk')  # 首先读取第一个CSV文件并包含表头
    df.to_csv(SaveFile_Path + '\\' + MergeFile_Name, encoding="gbk", index=False)  # 将读取的第一个CSV文件写入合并后的文件保存
    for i in range(0, len(file_list)):  # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
        print('合并进度:{}/{}'.format(i, len(file_list)))
        df = pd.read_csv(Folder_Path + '\\' + file_list[i], encoding='gbk')
        df.to_csv(SaveFile_Path + '\\' + MergeFile_Name, encoding="gbk", index=False, header=False, mode='a+')
    print('已生成'+MergeFile_Name)








if __name__ == '__main__':
    merge_data()





