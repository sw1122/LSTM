"""相关性计算"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt
import matplotlib
#修改了全局变量
matplotlib.rcParams['font.size']=40
matplotlib.rcParams['font.family']='Microsoft Yahei'
matplotlib.rcParams['font.weight']='bold'

data_list = []
a5_list = []
d1_list = []
d2_list = []
d3_list = []
d4_list = []
d5_list = []
d6_list = []

DELAY = 14
BOTTOM_D = -0.2  # 设置图D1-D5非NOx参数y轴下限
BOTTOM_R = -5    # 设置图D1-D5NOx参数y轴下限
RANGE = 1        # 设置图A5, RAWDATA非NOx参数y轴下限
X_INTER = 20     # 设置X轴坐标间隔
WAVES = 4        # 设置小波分解层数
l_list = [data_list, a5_list, d1_list, d2_list, d3_list, d4_list, d5_list, d6_list][:WAVES+2]
name_list = ['Raw data', 'A5', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'][:WAVES+2]

"""数据预处理程序"""
def data_preprocessing():
	rawDatas = pd.read_csv(r'E://ghYin/2020软测量/datas/N1/20200902_20200920_new_5s.csv', encoding='gbk')
	print(rawDatas.columns)

	rawDatas['总给煤量'] = (rawDatas['给煤量1'] + rawDatas['给煤量2'] + rawDatas['给煤量3'] + rawDatas['给煤量4'])
	rawDatas['一次风量'] = rawDatas['一次风左'] + rawDatas['一次风右']
	rawDatas['二次风量'] = rawDatas['二次风左'] + rawDatas['二次风右']
	rawDatas['烟气含氧量'] = rawDatas['高省烟气含氧量左'] + rawDatas['高省烟气含氧量右']
	rawDatas['沸下温度'] = (rawDatas['沸下温度（前）'] + rawDatas['沸下温度（后）']) / 2
	rawDatas['炉膛出口烟温'] = rawDatas['炉膛出口温度左'] + rawDatas['炉膛出口温度右']

	datas = pd.DataFrame()
	chosen_cols = [
		'一次风量', '二次风量', '总给煤量',
		'沸下温度', # '沸中温度', '沸上温度',
		'炉膛出口烟温', '锅炉出口蒸汽流量', '烟气含氧量',
		'脱硝入口NO原始浓度']

	for col in chosen_cols:
		datas[col] = rawDatas[col]

	temp = datas['脱硝入口NO原始浓度'].values

	'''标准化'''
	scaler = np.load('../均值方差.npz')
	mean_ = scaler['mean_']
	std_ = scaler['std_']
	newdatas_df = (datas.iloc[:,:-1] - mean_[:-1]) / std_[:-1]
	# newdatas_df = newdatas_df.rolling(window=12, win_type='gaussian', center=True).mean(std=2)[6:-6]
	newdatas_df['脱硝入口NO原始浓度'] = temp

	newdatas_df.columns = [
		'一次风流量', '二次风流量', '给煤量',
		'炉床温度', #'Bed temperature(mid)', 'Bed temperature(up)',
		'出口烟气温度', '主蒸汽流量', 'O2体积浓度',
		'NOx浓度']
	newdatas_df.to_csv('./滤波后.csv', encoding='gbk')

def plot_signal_decomp(data, wavelet, WAVES):
	"""Decompose and plot a signal S.
	S = An + Dn + Dn-1 + ... + D1
	"""
	w = pywt.Wavelet(wavelet)#选取小波函数
	a = data
	ca = []#近似分量
	cd = []#细节分量

	for i in range(WAVES):
		(a, d) = pywt.dwt(a, w, mode='sym')#进行5阶离散小波变换
		ca.append(a)
		cd.append(d)

	rec_a = []
	rec_d = []

	for i, coeff in enumerate(ca):
		coeff_list = [coeff, None] + [None] * i
		rec_a.append(pywt.waverec(coeff_list, w))#重构

	for i, coeff in enumerate(cd):
		coeff_list = [None, coeff] + [None] * i
		rec_d.append(pywt.waverec(coeff_list, w))

	data_list.append(data)
	a5_list.append(rec_a[-1])
	for i in range(1, WAVES+1):
		globals()['d{}_list'.format(i)].append(rec_d[i-1])

	
	return rec_d[0][:len(rec_d[0])] + rec_d[1][:len(rec_d[0])] + rec_d[2][:len(rec_d[0])]# + rec_d[3][:len(rec_d[0])]+ rec_d[4][:len(rec_d[0])] #+ rec_d[5][:len(rec_d[0])]


def main():
	"""数据预处理"""
	data_preprocessing()

	datas = pd.read_csv('./滤波后.csv', encoding='gbk')[450:510]
	dfNew = pd.DataFrame()

	for col in datas.columns[1:]:
		print('正在执行%s小波分解...'%col)
		cd = plot_signal_decomp(datas[col].values, 'db4', WAVES)
		dfNew[col] = cd

	for (l,name) in zip(l_list, name_list):
		if (name != 'Raw data') and (name != 'A5'):
			fig = plt.figure(figsize=(40,30), dpi=80)

			ax = fig.subplots(4, 2)
			axes = ax.flatten()
			plt.subplots_adjust(hspace=0.3, wspace=0.4, left=0.3)
			for idx in range(len(l)):
				axes[idx].plot(l[idx], label=name, linewidth=3, marker='s', markersize=6)
				axes[idx].set_ylabel(datas.columns[idx+1])
				axes[idx].xaxis.set_major_locator(plt.MultipleLocator(X_INTER))
				axes[idx].grid()
				if idx != len(l)-1:
					axes[idx].set_ylim(bottom=BOTTOM_D, top=-BOTTOM_D)
					axes[idx].yaxis.set_major_locator(plt.MultipleLocator(-BOTTOM_D/2))
				else:
					axes[idx].set_ylim(bottom=BOTTOM_R, top=-BOTTOM_R)
					axes[idx].yaxis.set_major_locator(plt.MultipleLocator(-BOTTOM_R/2))
			fig.savefig(r'.\小波分解\%s.png'%name)
		else:
			fig = plt.figure(figsize=(40, 30), dpi=80)
			ax = fig.subplots(4, 2)
			axes = ax.flatten()
			plt.subplots_adjust(hspace=0.3, wspace=0.4, left=0.3)
			for idx in range(len(l)):
				axes[idx].plot(l[idx], label=name, linewidth=3, marker='s', markersize=6)
				axes[idx].set_ylabel(datas.columns[idx + 1])
				axes[idx].xaxis.set_major_locator(plt.MultipleLocator(X_INTER))
				up = l[idx].mean()+RANGE
				down = l[idx].mean()-RANGE
				axes[idx].grid()
				if idx != len(l) - 1:
					axes[idx].set_ylim(bottom=down, top=up)
					axes[idx].yaxis.set_major_locator(plt.MultipleLocator(RANGE/2))

			fig.savefig(r'.\小波分解\%s.png' % name)

	for (data, name) in zip(l_list, name_list):
		data_np = np.array(data).T
		pd.DataFrame(data_np, columns=datas.columns[1:]).to_csv(r'.\小波分解\%s.csv'%name, encoding='utf-8')

	# temp = dfNew['NO concentration']
	# dfNew.drop('NO concentration', axis=1, inplace=True)
	# dfNew['NO concentration'] = temp
	# dfNew.to_csv(r'.\小波分解\高频合并.csv', encoding='gbk')
	# print(dfNew)

if __name__ == '__main__':
	main()