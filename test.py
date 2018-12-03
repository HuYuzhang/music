from mylib import wav2mfcc
import scipy.io.wavfile
path = '../raw_data/5_2.wav'
data = wav2mfcc(path)
print(data.shape)
print(data[5,:])

rate, sig = scipy.io.wavfile.read('../raw_data/chun_2.wav')
print(sig.shape[0] / rate)
