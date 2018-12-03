from mylib import *
import numpy as np
import sys
import os

# Here de define mode 1 2 3, which respond to 5. , chunlv, 12 average
if __name__ == '__main__':

    filenames = {0:'../raw_data/5.wav', 1:'../raw_data/chun.wav', 2:'../raw_data/12.wav'}
    # filenames = {0:'../raw_data/music.wav', 1:'../raw_data/music.wav', 2:'../raw_data/music.wav'}
    mfcc_len = 198 * 12
    train_path = '../train/train.h5'
    test_path = '../train/test.h5'
    train_num = 5000
    test_num = 1000
    sample_time = 30
    ratio = 0.8
    
    
    train_datas = np.zeros((3, train_num * 2, mfcc_len), dtype=np.float32)
    train_labels = np.zeros((3, train_num * 2, 3))
    test_datas = np.zeros((3, test_num * 2, mfcc_len), dtype=np.float32)
    test_labels = np.zeros((3, test_num * 2, 3))

    # First part of files
    for i in range(3):
        # For each file:
        filename = filenames[i]
        rate, signal = scipy.io.wavfile.read(fileName)
        # Only reserve the first channel
        signal = signal[:,0]
        time = int(len(signal) / rate)
        print('---> Dealing with file: %s, and its time is %d s'%(filename, time))
        # before the line, all data is for train, after the line for test
        div_line = int(time * ratio)
        for j in range(train_num):
            st = np.random.randint(0, div_line - sample_time)
            ed = st + sample_time
            seg_signal = signal[st:ed]
            mfcc = wav2mfcc(seq_signal, rate)
            train_datas[i,j,:] = mfcc
            train_labels[i,j,i] = 1
        for j in range(test_num):
            st = np.random.randint(div_line, time - sample_time)
            ed = st + sample_time
            seg_signal = signal[st:ed]
            mfcc = wav2mfcc(seq_signal, rate)
            valid_datas[i,j,:] = mfcc
            valid_labels[i,j,i] = 1

    # Deal with second part of input files
    filenames = {0:'../raw_data/5_2.wav', 1:'../raw_data/chun_2.wav', 2:'../raw_data/12_2.wav'}

    for i in range(3):
        # For each file:
        filename = filenames[i]
        rate, signal = scipy.io.wavfile.read(fileName)
        # Only reserve the first channel
        signal = signal[:,0]
        time = int(len(signal) / rate)
        print('---> Dealing with file: %s, and its time is %d s'%(filename, time))
        # before the line, all data is for train, after the line for test
        div_line = int(time * ratio)
        for j in range(train_num):
            st = np.random.randint(0, div_line - sample_time)
            ed = st + sample_time
            seg_signal = signal[st:ed]
            mfcc = wav2mfcc(seq_signal, rate)
            train_datas[i,train_num + j,:] = mfcc
            train_labels[i,train_num + j,i] = 1
        for j in range(test_num):
            st = np.random.randint(div_line, time - sample_time)
            ed = st + sample_time
            seg_signal = signal[st:ed]
            mfcc = wav2mfcc(seq_signal, rate)
            valid_datas[i,test_num + j,:] = mfcc
            valid_labels[i,test_num + j,i] = 1
    
    # Firstly shuffle the train_data
    array_list = list(range(2 * train_num))
    np.random.shuffle(array_list)
    train_datas = train_datas[:,array_list,:]
    train_labels = train_labels[:,array_list,:]

    # Then shuffle the test_data
    array_list = list(range(2 * test_num))
    np.random.shuffle(array_list)
    test_datas = test_datas[:,array_list,:]
    test_labels = test_labels[:,array_list,:]

    train_datas = train_datas.reshape([-1, mfcc_len])
    train_labels = train_labels.reshape([-1, 3])
    test_datas = test_datas.reshape([-1, mfcc_len])
    test_labels = test_labels.reshape([-1, 3])

    train_handler = h5Handler(train_path)
    test_path = h5Handler(test_path)
    train_handler.write(train_datas, train_labels, create=True)
    test_handler.write(test_datas, test_labels, create=True)
