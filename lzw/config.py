from scipy import signal
'''data config'''
fileList = ['5.wav', '5_2.wav', 'chun.wav', 'chun_2.wav', '12.wav', '12_2.wav']
labelMap = {'5.wav':0, '5_2.wav':0, 'chun.wav':1, 'chun_2.wav':1, '12.wav':2, '12_2.wav':2}
time = 30
num_of_second = 1000
'''mfcc config'''
fs = 8000
nw = 256 #帧长
winfunc = signal.hamming(nw)
inc = 80 #步长
paramNum = 13#每一帧的mfcc系数
frameNum = 18#帧的长度
'''svm config'''
c = 20
kernel = 'rbf'
gamma = 0.1
decision_function_shape = 'ovr' #one vs rest
'''dl config'''
featureNum = 4849 # paramNum * frameNum#特征数目
classNum = 3#分类数目
layerDepth = 20#隐层宽度
lr = 1e-3#训练速率
