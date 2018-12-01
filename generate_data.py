from mylib import *
import numpy as np
import sys
import os

# Here de define mode 1 2 3, which respond to 5. , chunlv, 12 average
if __name__ == '__main__':

    filenames = {0:'../raw_data/5.wav', 1:'../raw_data/chun.wav', 2:'../raw_data/12.wav'}
    # filenames = {0:'../raw_data/music.wav', 1:'../raw_data/music.wav', 2:'../raw_data/music.wav'}
    mfcc_len = 198 * 12
    h5_path = '../train/train.h5'
    sample_num = 10
    data_num = 10000
    
    handler = h5Handler(h5_path)
    raw_data = np.zeros((3, data_num, mfcc_len), dtype=np.float32)
    raw_label = np.zeros((3, data_num, 3))

    # ------------------ for debug -----------------------
    # for i in range(3):
    #     filename = '../raw_data/music.wav'
    #     raw_data[i,:sample_num,:] = wav2mfcc(filename)
    #     raw_label[i,:sample_num,i] = 1

    # print('ok')
    
    # handler.write(raw_data, raw_label, create=True)
    # exit(0)
    # ------------------ for debug -----------------------

    for i in range(3):
        filename = filenames[i]
        raw_data[i,:sample_num,:] = wav2mfcc(filename)
        raw_label[i,:sample_num,i] = 1
    
    for_train_data = raw_data[:,:sample_num-2,:]
    for_train_label = raw_label[:,:sample_num-2,:]
    for_valid_data = raw_data[:,sample_num-2:sample_num,:]
    for_valid_label = raw_label[:,sample_num-2:sample_num,:]
    # now 8 for train and 2 for test
    for st in range(0, int(0.8 * data_num),sample_num-2):
        raw_data[:,st:st+sample_num-2,:] = for_train_data
        raw_label[:,st:st+sample_num-2,:] = for_train_label

    for st in range(int(0.8 * data_num), data_num, 2):
        raw_data[:,st:st+2,:] = for_valid_data
        raw_label[:,st:st+2,:] = for_valid_label

    
    # consider that our data is not big, so read them to memory just in one time
    # now assume that we have 10 samples each, and 8 for train and 2 for test

    # emmm, we have two little samples, so I consider to repeat all the data and then shuffle them

    raw_data = raw_data.reshape([-1, mfcc_len])
    raw_label = raw_label.reshape([-1, 3])
    print(raw_data.shape, raw_label.shape)
    handler.write(raw_data, raw_label, create=True)

