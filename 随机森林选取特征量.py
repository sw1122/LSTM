import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''导入数据'''
SaveFile_Path = r'D:\Desktop\2018年5MW运行数据\2018年\各月数据集合'  # 要读取和保存的文件路径
savename = r'201803-12.csv'
os.chdir(SaveFile_Path)  # 修改当前工作目录

df = pd.read_csv(SaveFile_Path + '\\' + '201803-12.csv', header=0, index_col=0, encoding='gbk')
df.drop('1#镜场清洁度', axis=1, inplace=True)
# array = df.applymap(lambda x: type(x) != str) #applymap对dataframe中所有元素进行相同操作，效率更高,此处返回一个逻辑矩阵
# out_df = df[array]

# values = df.values  #转换成numpy格式
# row = df.shape[0]
# col = df.shape[1]
# for i in range(row):
#     for j in range(col):
#         if type(values[i, j]) == str:
#             values[i, j] = np.nan
# values = values.astype('float32')
# n_train_hours = 2000 #2000行作训练集，1000作测试集
# train = values[:n_train_hours, :]
# test = values[n_train_hours:3000, :]

train = df.head(200000)

#先用皮尔逊系数粗略的选择出相关性系数的绝对值大于0.3的属性列，这样不需要训练过多不重要的属性列
#可以这么做而且不会破坏实验的控制变量原则，因为根据皮尔逊相关系数选择出的重要性排名前10的属性列
#它们与要预测的属性列的皮尔逊相关系数均大于0.3，可以当成步骤1中也进行了同样的取相关系数为0.3的操作
features = train.corr().columns[train.corr()['发电量'].abs()> .1]
features = features.drop('发电量')

#使用随机森林模型进行拟合的过程
X_train = train[features]
y_train = train['发电量']
feat_labels = X_train.columns

rf = RandomForestRegressor(n_estimators=100, random_state=5, max_depth=None)
#用 StandardScaler() 进行归一化，同时用 SimpleImputer(strategy='median') 来填充缺失值。
rf_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('standardize', StandardScaler()), ('rf', rf)])
rf_pipe.fit(X_train, y_train)

# 根据随机森林模型的拟合结果选择特征
rf = rf_pipe.__getitem__('rf')
importance = rf.feature_importances_

# np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
imp_result = np.argsort(importance)[::-1][:60]

# 按重要性从高到低输出属性列名和其重要性
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))

#对属性列，按属性重要性从高到低进行排序
feat_labels = [feat_labels[i] for i in imp_result]

# '''测试集测试一下'''
# test = df.iloc[[1000, 1001, 1002, 1003, 1004, 1005, 1006]]
# feature = train.corr().columns[train.corr()['发电量'].abs()> .1]
# feature = feature.drop('发电量')
# X_test = test[feature]
# y_test = test['发电量']
# feat_label = X_test.columns

# Y_pre = rf.predict(X_test)
# print(Y_pre)
# print(Y_pre-y_test)

#绘制特征重要性图像
plt.title('Feature Importance')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(len(imp_result)), feat_labels, rotation=90)
plt.xlim([-1, len(imp_result)])
plt.tight_layout()
plt.savefig(r'D:\Desktop\2018年5MW运行数据\2018年\随机森林挑选特征' + '\\' + '(20万行seed=5)随机森林特征量.png')
plt.show()


