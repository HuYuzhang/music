import tensorflow as tf
from mylib import *
import h5py
import sys
import os


if __name__ == '__main__':
    batch_size = 64
    init_lr = 0.00001
    h5_path = '../train/train.h5'
    mfcc_len = 198 * 12
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
        _fc1 = tf.layers.dense(inputs, 3072, name='fc1')

        fc1 = tf.keras.layers.PReLU(shared_axes=[1], name='relu1')(_fc1)

        _fc2 = tf.layers.dense(fc1, 3072, name='fc2')

        fc2 = tf.keras.layers.PReLU(shared_axes=[1], name='relu2')(_fc2)

        _fc3 = tf.layers.dense(fc2, 3, name='fc3')

        fc3 = tf.keras.layers.PReLU(shared_axes=[1], name='relu3')(_fc3)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=fc3)

        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps = 10000, decay_rate=0.7)
        # optimizer = tf.train.AdamOptimizer(learning_rate=init_lr)
        # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        saver = tf.train.Saver()
        # checkpoint_dir = '../model/'
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        print('finish construct the network')
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
                #print(np.argmax(out, axis=1))

        print('Final result: ', np.mean(loss_s))
