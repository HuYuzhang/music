from mylib import *
import numpy as np
import sys
import os

# Here de define mode 1 2 3, which respond to 5. , chunlv, 12 average
if __name__ == '__main__':

    filenames = {0:'../raw_data/5_2.wav', 1:'../raw_data/chun_2.wav', 2:'../raw_data/12_2.wav'}
    # filenames = {0:'../raw_data/music.wav', 1:'../raw_data/music.wav', 2:'../raw_data/music.wav'}
    mfcc_len = 198 * 12
    h5_path = '../train/valid.h5'
    sample_num = 6
    data_num = 1000
    
    handler = h5Handler(h5_path)
    raw_data = np.zeros((3, sample_num, mfcc_len), dtype=np.float32)
    raw_label = np.zeros((3, sample_num, 3))


    for i in range(3):
        filename = filenames[i]
        raw_data[i,:sample_num,:] = wav2mfcc(filename)
        raw_label[i,:sample_num,i] = 1

    # for st in range(0, data_num, sample_num):
    #     raw_data[:,st:st+sample_num,:] = raw_data[:,0:sample_num,:]
    #     raw_label[:,st:st+sample_num,:] = raw_label[:,0:sample_num,:]
    
    raw_data = raw_data.reshape([-1, mfcc_len])
    raw_label = raw_label.reshape([-1, 3])
    array_list = list(range(0, raw_data.shape[0]))
    np.random.shuffle(array_list)
    raw_data = raw_data[array_list,:]
    raw_label = raw_label[array_list,:]
    print(raw_data.shape, raw_label.shape)
    handler.write(raw_data, raw_label, create=True)

