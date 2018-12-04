import tensorflow as tf
from config import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def getModel(input, y):
    W1 = weight_variable([featureNum, layerDepth])
    b1 = bias_variable([layerDepth])
    layer1 = tf.nn.relu(tf.matmul(input, W1) + b1)
    layer1 = tf.nn.l2_normalize(layer1, 1, epsilon=1e-12, name=None)
    W2 = weight_variable([layerDepth, layerDepth])
    b2 = bias_variable([layerDepth])
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    #layer2 = tf.nn.l2_normalize(layer2, 1, epsilon=1e-12, name=None)
    W3 = weight_variable([layerDepth, classNum])
    b3 = bias_variable([classNum])
    return tf.nn.softmax(tf.matmul(layer2, W3) + b3)


def dl_train(trainX, trainY):
    inputX = tf.placeholder(tf.float32, [None, featureNum])
    y = tf.placeholder(tf.float32, [None, classNum])
    output = getModel(inputX, y)
    loss = - tf.reduce_sum(tf.log(output) * y)
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(train, feed_dict={inputX: trainX, y: trainY})#是否要分batch？
    return sess, inputX, output, loss


def dl_test(sess, inputX, output, testX, testY):
    pred = sess.run(output, feed_dict={inputX: testX})
    print(pred, testY)
