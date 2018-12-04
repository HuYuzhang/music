import wave
import random
from config import *
def read_wave_data(file_path):
    f = wave.open(file_path)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    wave_data = wave_data[0, ::framerate // num_of_second]
    leng = len(wave_data)
    train_leng = leng * 2 // 3
    data_leng = time * num_of_second
    ret = []
    for i in range(dataNum):
        t = random.randint(0, train_leng - data_leng - 1)
        ret.append(wave_data[t:t + data_leng])
    ret = np.asarray(ret)
    test = []
    for i in range(testNum):
        t = random.randint(train_leng, leng - data_leng - 1)
        test.append(wave_data[t:t + data_leng])
    test = np.asarray(test)
    return ret, test