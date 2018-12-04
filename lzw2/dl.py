import tensorflow as tf
from tensorflow.contrib import rnn
from config import *
import random

class Model:
    def __init__(self, model, trainX, trainY, testX, testY):
        self.model = model
        self.input = tf.placeholder(tf.float32, [None, stepNum, paramNum])
        self.y = tf.placeholder(tf.float32, [None, classNum])
        self.dropout = tf.placeholder(tf.float32)
        self.output = self.getModel()
        self.pred = tf.argmax(self.output, 1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        self.sess = tf.Session()
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def getModel(self):
        if self.model == 'RNN':
            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(paramNum),
                                          output_keep_prob=self.dropout, state_keep_prob=self.dropout)
                       for _ in range(layerNum)]
            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(paramNum), state_keep_prob=self.dropout)
                       for _ in range(layerNum)]
            trans_input = tf.transpose(self.input, [1, 0, 2])
            encoder_output, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cell_fw, cell_bw, trans_input, time_major=True, dtype=tf.float32
            )
            encoder_state = tf.concat((encoder_state_fw[layerNum - 1].h, encoder_state_bw[layerNum - 1].h), 1)
            liner = tf.nn.xw_plus_b(encoder_state, self.weight_variable([2 * paramNum, classNum]),
                                    self.bias_variable([classNum]))
            return liner
        elif self.model == 'NN':
            reshape_input = tf.reshape(self.input, [-1, stepNum * paramNum])
            W1 = self.weight_variable([stepNum * paramNum, layerDepth])
            b1 = self.bias_variable([layerDepth])
            layer1 = tf.nn.relu(tf.nn.xw_plus_b(reshape_input, W1, b1))
            layer1 = tf.nn.dropout(layer1, self.dropout)
            W2 = self.weight_variable([layerDepth, classNum])
            b2 = self.bias_variable([classNum])
            return tf.nn.xw_plus_b(layer1, W2, b2)
        elif self.model == 'CNN':pass

    def train(self):
        self.sess.run(tf.initialize_all_variables())
        leng = len(self.trainX)
        for i in range(train_time):
            t = random.sample([x for x in range(leng)], batch_size)
            _, l = self.sess.run([self.train_op, self.loss],
                    feed_dict={self.input: self.trainX[t], self.y: self.trainY[t], self.dropout: 0.5})
            if i % 50 == 0:
                print("i = "+str(i) +" loss =" + str(l))
                self.test()
    def test(self):
        predict = self.sess.run(self.pred, feed_dict={self.input: self.testX, self.dropout: 1})
        acc = 0
        for i, test in zip(predict, self.testY):
            acc += test[i]
        acc = acc / len(predict)
        print("test accuracy:", acc)