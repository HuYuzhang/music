import cv2
import numpy as np
import h5py
import scipy.io.wavfile
from scipy.fftpack import dct

pre_emphasis = 0.97
frame_size=0.025
frame_stride=0.01

NFFT = 512
nfilt = 40
num_ceps = 12


#from fileName to mfcc vectors
def wav2mfcc(fileName):
    sample_rate,raw_signal=scipy.io.wavfile.read(fileName)
    music_len = 30
    mfcc_len = 198 * 12

    all_data = np.zeros([10, mfcc_len], dtype=np.float32)
    #exit(0)
    for i in range(0, 300, 30):
        signal = raw_signal[i * sample_rate: (i + 1) * sample_rate, :]

        #seems that here already finish the converting from 2D to 1D(two voice channel)
        
        # we don't define the channel, so the signal in function 'append' will do the operation of flatten
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # print('emphasize shape: ', emphasized_signal.shape)
        
        frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
        signal_length=len(emphasized_signal)
        frame_length=int(round(frame_length))
        frame_step=int(round(frame_step))
        num_frames=int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))
        # # --------------------for debug
        # print('some params: ', signal_length, frame_length, frame_step, num_frames)
        # # --------------------for debug
        pad_signal_length=num_frames*frame_step+frame_length
        z=np.zeros((pad_signal_length-signal_length))
        pad_signal=np.append(emphasized_signal,z)


        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

        frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)
        # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **
        # print('after hamming windows, the frames\' size is: ', frames.shape)

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        #print(mag_frames.shape)
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        # print('after fft windows, the mag_frames\' size is: ', mag_frames.shape)
        # print('after fft windows, the pow_frames\' size is: ', pow_frames.shape)


        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        # print(filter_banks.shape)
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
        (nframes, ncoeff) = mfcc.shape

        n = np.arange(ncoeff)
        cep_lifter =22
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  #*

        #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        all_data[int(i / 30), :] = mfcc.reshape([mfcc_len])
        #print(i, '-----------', mfcc.shape)
    return all_data


class h5Handler(object):
    def __init__(self, h5_path):
        self.h5_path = h5_path

    def read(self, key, start, end, step):
        fid = h5py.File(self.h5_path, 'r')
        ret = fid[key][start:end:step]
        fid.close()
        return ret

    # right now very bad way to assign 3072 and 1024, but not a big problem
    # assume that datas and labels are of size [n, c, h, w]
    def write(self, datas, labels, create=True):
        if create:
            f = h5py.File(self.h5_path, 'w')
            f.create_dataset('data', data=datas, maxshape=datas.shape, chunks=True, dtype='float32')
            f.create_dataset('label', data=labels, maxshape=labels.shape, chunks=True, dtype='float32')
            f.close()
        else:
            pass    
        #     # append mode
        #     f = h5py.File(self.h5_path, 'a')
        #     h5data = f['data']
        #     h5label = f['label']
        #     cursize = h5data.shape
        #     addsize = datas.shape

        #     # # --------------for debug------------------
        #     # print('-------now begin to add data------')
        #     # print(cursize)
        #     # # --------------for debug------------------

        #     h5data.resize([cursize[0] + addsize[0], 3072, 1, 1])
        #     h5label.resize([cursize[0] + addsize[0], 1024, 1, 1])
        #     h5data[-addsize[0]:,:,:,:] = datas
        #     h5label[-addsize[0]:,:,:,:] = labels
        #     f.close()