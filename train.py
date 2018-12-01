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
    weights_name = None

    # prepare for the training data
    hf = h5py.File(h5_path)
    print('Loading data')
    x = np.array(hf['data'], dtype=np.float32)
    y = np.array(hf['label'], dtype=np.float32)
    length = x.shape[0]
    # array_list = list(range(0, length))
    # np.random.shuffle(array_list)
    bar = int(length*0.8)
    print('-------', bar, length)
    train_data = x[:bar, :]
    val_data = x[bar:, :]
    train_label = y[:bar, :]
    val_label = y[bar:, :]
    
    
    def train_generator():
        while True:
            for i in range(0, bar, batch_size)[:-1]:
                yield train_data[i:i+batch_size, :], train_label[i:i+batch_size, :]
            # np.random.shuffle(train_data)

    def val_generator():
        for i in range(0, length-bar, batch_size)[:-1]:
            yield val_data[i:i+batch_size, :], val_label[i:i+batch_size, :]

    inputs = tf.placeholder(tf.float32, [batch_size, mfcc_len])
    targets = tf.placeholder(tf.float32, [batch_size, 3])
    
    train_op = 0
    loss = 0

    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE): 
        _fc1 = tf.layers.dense(inputs, 3072, name='fc1')

        fc1 = tf.keras.layers.PReLU(shared_axes=[1], name='relu1')(_fc1)

        _fc2 = tf.layers.dense(fc1, 3072, name='fc2')

        fc2 = tf.keras.layers.PReLU(shared_axes=[1], name='relu2')(_fc2)

        _fc3 = tf.layers.dense(fc2, 3, name='fc3')

        fc3 = tf.keras.layers.PReLU(shared_axes=[1], name='relu3')(_fc3)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=fc3)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps = 10000, decay_rate=0.7)
        optimizer = tf.train.AdamOptimizer(learning_rate=init_lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        saver = tf.train.Saver(max_to_keep=30)
        checkpoint_dir = '../model/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('finish construct the network')
        with tf.Session() as sess:
            if weights_name is not None:
                saver.restore(sess, weights_name)
            else:
                sess.run(tf.global_variables_initializer())
            total_var = 0
            for var in tf.trainable_variables():
                shape = var.get_shape()
                par_num = 1
                for dim in shape:
                    par_num *= dim.value
                total_var += par_num
            print("Number of total variables: %d" %(total_var))
            options = tf.RunOptions()  # trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            data_gen = train_generator()
            interval = 500
            print('now begin to training')
            for i in range(5000):
                if i % interval == 0:
                    val_gen = val_generator()
                    val_loss_s = []
                    for v_data, v_label in val_gen:
                        val_loss = sess.run(loss, feed_dict={
                                                    inputs: v_data, targets: v_label})
                    #print(type(val_loss))
                    #print(val_loss.shape)    
                    val_loss_s.append(np.mean(val_loss))

                    print("step %8d, loss: %f" % (i, np.mean(val_loss_s)))
                    
                # ------------------- Here is the training part ---------------
                iter_data, iter_label = next(data_gen)
                # print(iter_data.shape)
                feed_dict = {inputs: iter_data, targets: iter_label}
                _, train_loss = sess.run([train_op, loss],
                                        feed_dict=feed_dict,
                                        options=options,
                                        run_metadata=run_metadata)
                if i % 10 == 0:
                    print('train loss: ', np.mean(train_loss)) 
                if i % 10000 == 0:
                    save_path = saver.save(sess, os.path.join(
                        checkpoint_dir, "%06d.ckpt" % (i)))
