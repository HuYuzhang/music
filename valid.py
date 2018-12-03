import tensorflow as tf
from mylib import *
import h5py
import sys
import os


if __name__ == '__main__':
    batch_size = 32
    init_lr = 0.00001
    h5_path = '../train/test.h5'
    mfcc_len = 2998 * 12
    weights_name = sys.argv[1]

    # prepare for the training data
    hf = h5py.File(h5_path)
    print('Loading data')
    x = np.array(hf['data'], dtype=np.float32)
    y = np.array(hf['label'], dtype=np.float32)
    length = x.shape[0]

    def val_generator():
        for i in range(0, length, batch_size)[:-1]:
            yield x[i:i+batch_size, :], y[i:i+batch_size, :]

    inputs = tf.placeholder(tf.float32, [batch_size, mfcc_len])
    targets = tf.placeholder(tf.float32, [batch_size, 3])
    

    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE): 
        _fc1 = tf.layers.dense(inputs, 1024, name='fc1')

        fc1 = tf.keras.layers.PReLU(shared_axes=[1], name='relu1')(_fc1)

        _fc2 = tf.layers.dense(fc1, 2048, name='fc2')

        fc2 = tf.keras.layers.PReLU(shared_axes=[1], name='relu2')(_fc2)

        _fc3 = tf.layers.dense(fc2, 3, name='fc3')

        fc3 = tf.keras.layers.PReLU(shared_axes=[1], name='relu3')(_fc3)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=fc3)
        print('finish construct the network')

        # Now finishing building the network, and begin the test
        saver = tf.train.Saver()
        right_num = 0
        tot_num = 0
        with tf.Session() as sess:
            if weights_name is not None:
                saver.restore(sess, weights_name)
                print('Sucessfully restore from weights: ', weights_name)
            else:
                print('Error!, no weights provided!')
                exit(0)

            val_gen = val_generator()
            loss_s = []
            for v_data, v_label in val_gen:
                #print(v_data.shape)
                v_loss, out = sess.run([loss, fc3], feed_dict={inputs: v_data, targets: v_label})
                loss_s.append(np.mean(v_loss))
                mode = np.argmax(out, axis=1)
                #print(mode.shape)
                #exit(0)
                # check if this is the right classification
                for i in range(batch_size):
                    tot_num = tot_num + 1
                    #print(v_label.shape)
                    if v_label[i, mode[i]] == 1:
                        right_num = right_num + 1


        print('Final loss: ', np.mean(loss_s))
        print('We test %d samples, and %d samples are right, Accurate rate is: %f'%(tot_num, right_num, right_num / tot_num))
