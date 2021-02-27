import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from tensorflow_core import keras
from numpy import concatenate


pd.set_option('display.max_columns', None)

'''修改数据'''
SaveFile_Path = r'D:\Desktop\2018年5MW运行数据\2018年\各月数据集合'  # 要读取和保存的文件路径
savename = r'201803-12特征量.csv'
os.chdir(SaveFile_Path)  # 修改当前工作目录

DIR = 'LSTM'
if not os.path.exists(DIR):  #判断括号里的文件是否存在的意思，括号内的可以是文件路径。
    os.makedirs(DIR)  #用于递归创建目录

TIME_STEP = 180 #时间步长180min 3h
DELAY = 0  #当前时刻
# RATIO1 = 0.9
BATCHSZ = 30
EACH_EPOCH = 1
LR = 0.0001  #keras.optimizers.Adam优化器中的学习率
# ratio = 1000

'''
绘制初始数据曲线
'''
def plot_init():
    df = pd.read_csv(SaveFile_Path + '\\' + '201803-12特征量.csv', header=0, index_col=0, encoding='gbk')
    values = df.values
    col = df.shape[1]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.figure()
    for i in range(0, col):
        plt.subplot(col, 1, i+1)
        plt.plot(values[:, i])
        plt.title(df.columns[i], y=0.5, loc='right')
    plt.show()

'''
LSTM数据准备
'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def cs_to_sl():
    #load dataset
    dataset = pd.read_csv(SaveFile_Path + '\\' + savename, header=0, index_col=0, encoding='gbk')
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))  #sklearn 归一化函数
    scaled = scaler.fit_transform(values)  #fit_transform(X_train) 意思是找出X_train的均值和​​​​​​​标准差，并应用在X_train上。
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 180, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990]], axis=1, inplace=True)
    return reframed, scaler

'''
构造模型
'''
def train_test(reframed):
    # split into train and test sets  将数据集进行划分，然后将训练集和测试集划分为输入和输出变量，最终将输入（X）改造为LSTM的输入格式，即[samples,timesteps,features]
    values = reframed.values
    n_train_hours = 2000 #20万行作训练集，11万作测试集
    train = values[:n_train_hours, :]
    test = values[n_train_hours:3000, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 180, 11))  #用reshape使其变为三维数组（样本数（行），时间步长，特征数（列））
    test_X = test_X.reshape((test_X.shape[0], 180, 11))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y

'''
搭建模型
'''
def model_build(train_datas): #train_datas = train_X
    # LSTM层
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(40, input_shape=(train_datas.shape[1:]), return_sequences=True, )) #, return_sequences=True 400记忆体个数
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(30, return_sequences=True)) # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(40, return_sequences=True))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(40))
    model.add(keras.layers.BatchNormalization())  #批标准化：对一小批数据（batch）做标准化处理（使数据符合均值为0，标准差为1分布）
    model.add(keras.layers.Dense(1))   #全连接层
#配置训练方法
    model.compile(optimizer=keras.optimizers.Adam(lr=LR, amsgrad=True), loss='mse', metrics=[rmse])  # mae: mean_absolute_error

    return model

'''模型拟合'''
def model_fit(model, train_datas, train_labels,x_test, y_test):    #train_X, train_y, test_X, test_y


    checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt" #模型保存位置

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    lr_reduce = keras.callbacks.ReduceLROnPlateau('val_loss',     #学习停止，模型会将学习率降低2-10倍，该hui
                                                  patience=4,
                                                  factor=0.7,
                                                  min_lr=0.00001)
    best_model = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,#保存模型
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='min',
                                                 )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=15)

    history = model.fit(
        train_datas, train_labels,
        validation_data=(x_test, y_test),
        batch_size=BATCHSZ,
        epochs=EACH_EPOCH,
        verbose=2,
        callbacks=[
        best_model,
        early_stop,
        lr_reduce,
                    ]
    )
    #model.save_weights('./{}/LSTM_model_weights.hdf5'.format(DIR))

    return model, history

'''评价部分'''
def rmse(y_true, y_pred):  #sqrt求元素平方根  mean求张量平均值
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

def model_evaluation(model, test_X, test_y, savename):
    yhat = model.predict(test_X)
    test_X = test_X[:, 0, :]
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    y_pred = scaler.inverse_transform(inv_yhat)  #预测值转化
    y_pred = y_pred[:, 0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_yact_hat = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_yact_hat)
    y_real = inv_y[:, 0]   #真实值转化
# '''
# 在这里为什么进行比例反转，是因为我们将原始数据进行了预处理（连同输出值y），
# 此时的误差损失计算是在处理之后的数据上进行的，为了计算在原始比例上的误差需要将数据进行转化。
# 同时笔者有个小Tips：就是反转时的矩阵大小一定要和原来的大小（shape）完全相同，否则就会报错。
# '''
    y_pred_df = pd.DataFrame(index=y_pred)
    y_pred_df.to_csv(r'./{}/LSTM_pred.csv'.format(DIR), encoding='gbk', sep=',')
    y_real_df = pd.DataFrame(index=y_real)
    y_real_df.to_csv(r'./{}/LSTM_real.csv'.format(DIR), encoding='gbk', sep=',')

    model_plot(y_real, y_pred, savename)  #下一个自定义model_plot函数

    RMSE = np.sqrt(mean_squared_error(y_pred, y_real))
    R2_SCORE = r2_score(y_pred, y_real)
    print('RMSE: {}\nR2_SCORE: {}\n'.format(RMSE, R2_SCORE))
    return RMSE, R2_SCORE

'''保存模型信息'''
def model_save(RMSE, R2_SCORE, savename):
    with open(r'./{}/LSTM.txt'.format(DIR), 'a') as fh:
        fh.write('参数设置：\nTIME_STEP: {}\tDELAY: {}\n'.format(TIME_STEP, DELAY))
        fh.write('RMSE: {}\nR2_SCORE: {}\n\n'.format(RMSE, R2_SCORE))
        print('%s模型信息保存成功！\n\n\n'% savename)

'''绘图相关'''
def model_plot(y_real, y_pred, savename):
    plt.cla()
    fig1 = plt.figure(figsize=(10, 14), dpi=80)
    plt.subplots_adjust(hspace=0.3)  #hspace=0.3为子图之间的空间保留的高度，平均轴高度的一部分.加了这个语句，子图会稍变小，因为空间也占用坐标轴的一部分

    ax1 = fig1.add_subplot(1, 1, 1)  # 1行x1列的网格里第一个
    ax1.plot(y_real, '-', c='blue', label='Real', linewidth=2)
    ax1.plot(y_pred, '-', c='red', label='Predict forecast', linewidth=2)
    ax1.legend(loc='upper right')
    ax1.set_xlabel('min')
    ax1.set_ylabel('KWH')
    ax1.grid()


    fig1.savefig('./{}/{}.png'.format(DIR, savename))
    plt.close()

if __name__ == '__main__':
    #plot_init()
    reframed, scaler = cs_to_sl()
    train_X, train_y, test_X, test_y = train_test(reframed)
    (train_X_shuffled, train_y_shuffled) = shuffle(train_X, train_y)

    # 设置gpu内存自增长
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    '''模型拟合'''
    model = model_build(train_X)
    model, history = model_fit(model, train_X_shuffled, train_y_shuffled, test_X, test_y)
    '''训练集评估'''
    RMSE_list, R2_SCORE_list = model_evaluation(model, train_X, train_y, '训练')
    model_save(RMSE_list, R2_SCORE_list, '训练')
    '''测试集评估'''
    RMSE_list, R2_SCORE_list = model_evaluation(model, test_X, test_y, '验证')
    model_save(RMSE_list, R2_SCORE_list, '验证')
