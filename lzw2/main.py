import mfcc
import processData
import dl
from config import *
def main():
    dataX = []
    dataY = []
    testX = []
    testY = []
    for file in fileList:
        wave_data, test = processData.read_wave_data('../../raw_data/' + file)
        for wave in wave_data:
            data = mfcc.enframe(wave)
            data = mfcc.mfcc(data).reshape(-1)
            dataX.append(data)
            dataY.append(labelMap[file])
        for wave in test:
            data = mfcc.enframe(wave)
            data = mfcc.mfcc(data).reshape(-1)
            testX.append(data)
            testY.append(labelMap[file])
    dataX, dataY = np.asarray(dataX), np.asarray(dataY)
    testX, testY = np.asarray(testX), np.asarray(testY)
    dataX = dataX.reshape([-1, stepNum, paramNum])
    dataY = np.asarray(list(map(lambda x: [1 if x == i else 0 for i in range(3)], dataY)))
    testX = testX.reshape([-1, stepNum, paramNum])
    testY = np.asarray(list(map(lambda x: [1 if x == i else 0 for i in range(3)], testY)))
    model_rnn = dl.Model('RNN', dataX, dataY, testX, testY)
    model_nn = dl.Model('NN', dataX, dataY, testX, testY)
    print("training data num:", len(dataX), "time step of each data", stepNum)
    print("\n\n\nuse Bi-LSTM to classify:")
    model_rnn.train()
    print("\nfinal accuracy:")
    model_rnn.test()
    print("\n\n\nuse NN to classify:")
    model_nn.train()
    print("\nfinal accuracy:")
    model_nn.test()
if __name__ == '__main__':
    main()
