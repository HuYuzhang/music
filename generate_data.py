from mylib import *
import numpy as np
import sys
import os

# Here de define mode 1 2 3, which respond to 5. , chunlv, 12 average
if __name__ == '__main__':

    filenames = {0:'five_tune.wav', 1:'just_tune.wav', 2:'twelve_tune.wav'}
    mfcc_len = 1000
    h5_path = 'train.h5'
    sample_num = 10
    data_num = 1000
    
    handler = h5Handler(h5_path)
    raw_data = np.zeros((3, data_num, mfcc_len), dtype=np.float32)
    raw_label = np.zeros((3, data_num, 3))
    for i in range(3):
        filename = filenames[i]
        raw_data[i,:sample_num,:] = mfcc(filename)
        raw_label[i,:sample_num,i] = 1
    
    for st in range(0,data_num,sample_num):
        raw_data[:,st:st+sample_num;:] = raw_data[:,:sample_num,:]
        raw_label[:,st:st+sample_num,:] = raw_label[:,:sample_num,:]

    
    # consider that our data is not big, so read them to memory just in one time
    # now assume that we have 10 samples each, and 8 for train and 2 for test

    # emmm, we have two little samples, so I consider to repeat all the data and then shuffle them

    raw_data = raw_data.reshape([-1, mfcc_len])
    raw_label = raw_label.reshape([-1, 3])
    handler.write(raw_data, raw_label, create=True)

