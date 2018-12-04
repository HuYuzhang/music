from scipy import signal
import numpy as np
'''data config'''
fileList = ['5.wav', '5_2.wav', 'chun.wav', 'chun_2.wav', '12.wav', '12_2.wav']
labelMap = {'5.wav':0, '5_2.wav':0, 'chun.wav':1, 'chun_2.wav':1, '12.wav':2, '12_2.wav':2}
time = 30#每一个样本的时间长度
num_of_second = 100#每一秒采样次数
dataNum = 150 #每一个音频随机生成的训练样本数
testNum = 20 #每一个音频随机生成的测试样本数
'''mfcc config'''
fs = 8000
nw = 256 #帧长
winfunc = signal.hamming(nw)
inc = 80 #步长
paramNum = 13#每一帧的mfcc系数
frameNum = 18#帧的长度
'''dl config'''
stepNum = int(np.ceil((1.0 * time * num_of_second - nw + inc) / inc)) #时间序列长度（帧的个数）
classNum = 3#分类数目
layerNum = 4#RNN的层数
layerDepth = 50#隐层宽度
lr = 1e-3#训练速率
train_time = 400#训练批次
batch_size = 50#每一批的个数
