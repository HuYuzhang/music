import numpy as np
from scipy.fftpack import dct
from config import *
from scipy import signal

def preprocess(wave_data):
    Df=5
    fs=8000
    N=fs/Df
    t = np.arange(0,(N-1)/fs,1/fs)
    wave_data=np.sin(2*np.pi*200*t)

    #预加重
    b, a = signal.butter(1, 1 - 0.97, 'high')
    return signal.filtfilt(b, a, wave_data)


def enframe(wave_data):
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频信号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔) （窗口宽度）
    inc:相邻帧的间隔（同上定义）（也就是窗口移动的步长）
    '''
    wlen = len(wave_data) #信号总长度
    if wlen <= nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else: #否则，计算帧的总长度
        nf = int(np.ceil((1.0 * wlen - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw) #所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - wlen,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros)) #填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) +\
              np.tile(np.arange(0, nf*inc, inc), (nw, 1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32) #将indices转化为矩阵
    frames = pad_signal[indices] #得到帧信号
    win = np.tile(winfunc, (nf, 1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵


def mfcc(X): #X是采样的帧
    frameNum = X.shape[0]  #获取帧数
    ret = []
    #归一化倒谱提升窗口
    lifts = 1 + 6 * np.sin(np.pi * np.asarray([i for i in range(1, paramNum + 1)]) / 12)
    for i in range(frameNum):
        y = X[i, :]
        yf = np.abs(np.fft.fft(y))
        # 谱线能量
        yf = yf ** 2
        # 梅尔滤波器系数
        nfilt = 24
        low_freq_mel = 0
        NFFT = 256
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # 把 Hz 变成 Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 将梅尔刻度等间隔
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 把 Mel 变成 Hz
        bin = np.floor((NFFT + 1) * hz_points / fs)
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(yf[0:129], fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
        filter_banks = 10 * np.log10(filter_banks)  # dB
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        # print(filter_banks)
        # DCT系数
        num_ceps = paramNum
        c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (num_ceps + 1)]  # Keep 2-13
        c2 *= lifts
        ret.append(c2)
    return np.asarray(ret).reshape([1, -1])
'''
Df=5
fs=8000
N=fs/Df
t = np.arange(0,(N-1)/fs,1/fs)
wave_data=np.sin(2*np.pi*200*t)

#预加重
#b,a = signal.butter(1,1-0.97,'high')
#emphasized_signal = signal.filtfilt(b,a,wave_data)
#归一化倒谱提升窗口
lifts=[]
for n in range(1,13):
    lift =1 + 6 * np.sin(np.pi * n / 12)
    lifts.append(lift)
#print(lifts)
'''
#分帧、加窗
'''
X=enframe(wave_data, 256, 80, winfunc)    #转置的原因是分帧函数enframe的输出矩阵是帧数*帧长

print(c2)
plt.plot(c2)
plt.show()
'''

'''
Df = 5
fs = 8000
N = fs/Df
t = np.arange(0, (N - 1) / fs, 1 / fs)
wave_data = np.sin(2*np.pi*200*t)

wave_data = preprocess(wave_data)
frame = enframe(wave_data)
feature = mfcc(frame)
print(feature.shape)
'''

'''
from svm import *
feature = feature.reshape(-1)
clf = svm_train([feature, feature * 2, feature * 3], [1, 2, 3])
svm_test(clf, [feature, feature * 2, feature * 3], [1, 2, 3])
'''

'''
from dl import *
sess, inputX, output, _ = dl_train(feature, [[1, 0, 0]])
dl_test(sess, inputX, output, feature, [[1, 0, 0]])
'''






