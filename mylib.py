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
    sample_rate,signal=scipy.io.wavfile.read(fileName)
    music_len = 30

    # print(sample_rate,len(signal))
    #读取前3.5s 的数据
    signal=signal[0:int(music_len*sample_rate)]

    #seems that here already finish the converting from 2D to 1D(two voice channel)
    
    # we don't define the channel, so the signal in function 'append' will do the operation of flatten
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # print('emphasize shape: ', emphasized_signal.shape)
    
    frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
    signal_length=len(emphasized_signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    num_frames=int(numpy.ceil(float(numpy.abs(signal_length-frame_length))/frame_step))
    # # --------------------for debug
    # print('some params: ', signal_length, frame_length, frame_step, num_frames)
    # # --------------------for debug
    pad_signal_length=num_frames*frame_step+frame_length
    z=numpy.zeros((pad_signal_length-signal_length))
    pad_signal=numpy.append(emphasized_signal,z)


    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    # print('after hamming windows, the frames\' size is: ', frames.shape)

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    #print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    # print('after fft windows, the mag_frames\' size is: ', mag_frames.shape)
    # print('after fft windows, the pow_frames\' size is: ', pow_frames.shape)


    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    # print(filter_banks.shape)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = numpy.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*

    #filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    return mfcc

# Assume that our video is I420
def read_frame(filename, idx, _height, _width, mode=0):
    pixel_num = _height * _width
    byte_num = int(pixel_num * 3 / 2)
    # print(byte_num)
    with open(filename, 'rb') as f:
        f.seek(idx * byte_num, 0)
        # only luma mode
        if mode == 0:
                data = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                return data.reshape([_height, _width])

        else:
                # Three color mode
                dataY = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                dataU = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                dataV = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                img = np.zeros([3, _height, _width])
                img[0,:,:] = dataY.reshape([_height, _width])
                img[1,0::2,0::2] = dataU.reshape([int(_height / 2), int(_width / 2)])
                img[1,0::2,1::2] = img[1,0::2,0::2]
                img[1,1::2,0::2] = img[1,0::2,0::2]
                img[1,1::2,1::2] = img[1,0::2,0::2]
                img[2,0::2,0::2] = dataV.reshape([int(_height / 2), int(_width / 2)])
                img[2,0::2,1::2] = img[2,0::2,0::2]
                img[2,1::2,0::2] = img[2,0::2,0::2]
                img[2,1::2,1::2] = img[2,0::2,0::2]
                img = img.astype(np.uint8)
                img = img.transpose(1,2,0)
                print(img.dtype)
                print('---', img.shape)
                img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
                return img

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