import wave
import numpy as np
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
    wave_data = wave_data[0, :: framerate // num_of_second]
    wave_data = wave_data[:(wave_data.shape[0] // (time * num_of_second)) * time * num_of_second]
    wave_data = np.reshape(wave_data, [-1, time * num_of_second])
    return wave_data





'''
    f_w = wave.open(file_path + 'w', 'wb')
    f_w.setnchannels(nchannels)
    f_w.setsampwidth(1000)
    f_w.setframerate(framerate)
    f_w.writeframes(str_data)
    f_w.close()
'''