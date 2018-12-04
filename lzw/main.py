import mfcc
import numpy as np
import svm
import processData
import random
import dl
from config import *
def main():
    dataX = []
    dataY = []
    for file in fileList:
        wave_data = processData.read_wave_data('../../raw_data/' + file) #10 × 30000
        for wave in wave_data:
            data = mfcc.enframe(wave) #30000 ---> 373 * 256
            data = mfcc.mfcc(data).reshape(-1)  # 1 × 4849 = 373 × 13（373帧，每一帧的mfcc系数有13个）
            dataX.append(data)
            dataY.append(labelMap[file])
        #svm.
    dataX, dataY = np.asarray(dataX), np.asarray(dataY)
    leng = len(dataX)
    shuf = [i for i in range(leng)]
    random.shuffle(shuf)
    dataX, dataY = dataX[shuf], dataY[shuf]
    tail = leng * 2 // 3
    #clf = svm.svm_train(dataX[:tail], dataY[:tail])
    #svm.svm_test(clf, dataX[tail:], dataY[tail:])
    dataY = np.asarray(list(map(lambda x:[1 if x == i else 0 for i in range(3)], dataY)))
    sess, inputX, output, loss = dl.dl_train(dataX[:tail], dataY[:tail])
    dl.dl_test(sess, inputX, output, dataX[tail:], dataY[tail:])
if __name__ == '__main__':
    main()
