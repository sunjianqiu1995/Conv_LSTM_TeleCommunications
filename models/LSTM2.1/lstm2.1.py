#-*-coding:utf-8 -*-
import os.path
import time

import numpy as np
import tensorflow as tf

import BasicConvLSTMCell
import layer_def as ld

"""
LSTM (从conv_LSTM3.1改，仅修改conv filter数目
Each batch, random timestamps, total 8784 - 12 + 1 timestamps
timestamps: 12
batch size: 12
seq_start: 10
"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train_store_lstm2.1',
                           """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 12,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start',10,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 400000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                          """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 12,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")


# fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
def load_data():
    # load data
    # total: timecount8784 * 100*100*5
    teledata = np.load('../batch_data/total.npy')
    return teledata
def load_test_data():
    # load data
    # total: timecount144 * 100*100*5
    data_test = np.load('../data_test/test_total.npy')
    return data_test

def network(inputs, hidden, lstm=True):
    conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
    # conv4
    conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
    y_0 = conv4
    if lstm:
        # conv lstm cell
        with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
            # 25 * 4 = 100
            cell = BasicConvLSTMCell.BasicConvLSTMCell([25, 25], [25, 25], 4)
            if hidden is None:
                hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
            y_1, hidden = cell(y_0, hidden)
    else:
        y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")

    # conv5
    conv5 = ld.transpose_conv_layer(y_1, 1, 1, 8, "decode_5")
    # conv6
    conv6 = ld.transpose_conv_layer(conv5, 3, 2, 8, "decode_6")
    # conv7
    conv7 = ld.transpose_conv_layer(conv6, 3, 1, 8, "decode_7")
    # x_1
    x_1 = ld.transpose_conv_layer(conv7, 3, 2, 5, "decode_8", True)  # set activation to linear

    return x_1, hidden


# make a template for reuse
network_template = tf.make_template('network', network)


def train():
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():
        # make inputs
        x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 100, 100, 5])

        # possible dropout inside
        keep_prob = tf.placeholder("float")
        x_dropout = tf.nn.dropout(x, keep_prob)

        # create network
        x_unwrap = []

        # conv network
        hidden = None
        for i in xrange(FLAGS.seq_length - 1):
            if i < FLAGS.seq_start:
                x_1, hidden = network_template(x_dropout[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_unwrap.append(x_1)

        # pack them all together
        x_unwrap = tf.stack(x_unwrap)
        ################################?
        x_unwrap = tf.transpose(x_unwrap, [1, 0, 2, 3, 4])

        # for test, generate future data
        x_unwrap_g = []
        hidden_g = None
        future_stamps = FLAGS.seq_length - 1 # (x_unwrap_g length: (Flags.seq_length - 1)--> prediction timestamps [1,Flags.seq_length - 1]
        for i in xrange(future_stamps): # predict n timestamps in the future
          if i < FLAGS.seq_start:
            x_1_g, hidden_g = network_template(x_dropout[:,i,:,:,:], hidden_g)
          else:
            x_1_g, hidden_g = network_template(x_1_g, hidden_g)
          x_unwrap_g.append(x_1_g)

        # pack them generated ones
        x_unwrap_g = tf.stack(x_unwrap_g)
        ################################?
        x_unwrap_g = tf.transpose(x_unwrap_g, [1,0,2,3,4])

        # calc total loss (compare x_t(groudtruth) to x_t(predict))
        loss = tf.nn.l2_loss(x[:, FLAGS.seq_start+1:, :, :, :] - x_unwrap[:, FLAGS.seq_start:, :, :, :])
        tf.summary.scalar('loss', loss)

        # training
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # List of all Variables
        variables = tf.global_variables()

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())

        # Summary op
        summary_op = tf.summary.merge_all()

        # Evaluate model
        # just use loss
        pred_loss = tf.nn.l2_loss(x[:, FLAGS.seq_start+1:, :, :, :] - x_unwrap_g[:, FLAGS.seq_start:, :, :, :])


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()

        # init if this is the very time training
        print("init network from scratch")
        sess.run(init)

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
        # load whole data
        teledata = load_data()
        test_data = load_test_data()
        # train_log
        if tf.gfile.Exists(FLAGS.train_dir+'/train_lstm2.1.log'):
            tf.gfile.Remove(FLAGS.train_dir+'/train_lstm2.1.log')
        train_log = open(FLAGS.train_dir+'/train_lstm2.1.log','w')

        # test_log
        if tf.gfile.Exists(FLAGS.train_dir+'/test_lstm2.1.log'):
            tf.gfile.Remove(FLAGS.train_dir+'/test_lstm2.1.log')
        test_log = open(FLAGS.train_dir+'/test_lstm2.1.log','w')


        # Training
        for step in xrange(FLAGS.max_step + 1):  # 1 step = 1 batches training
            # generate batch_data: random
            a = np.random.randint(1, 8784 - FLAGS.seq_length + 2) # [1, 8784 - 12 + 2)
            batch_data = teledata[(a - 1): (a - 1) + FLAGS.seq_length]  # (12, 100, 100, 5)
            for bt in range(1, FLAGS.batch_size):
                a = np.random.randint(1, 8784 - FLAGS.seq_length + 2)
                slices = teledata[(a - 1): (a - 1) + FLAGS.seq_length]  # (12, 100, 100, 5)
                batch_data = np.vstack((batch_data, slices))
            # batch_data: (144, 100, 100, 5)
            batch_data = batch_data.reshape(FLAGS.batch_size, FLAGS.seq_length, 100, 100, 5) # (FLAGS.batch_size, FLAGS.seq_length, 100, 100, 5)


            t = time.time()
            _, loss_r = sess.run([train_op, loss], feed_dict={x: batch_data, keep_prob: FLAGS.keep_prob})
            elapsed = time.time() - t

            if step % 200 == 0 and step != 0:
                summary_str = sess.run(summary_op, feed_dict={x: batch_data, keep_prob: FLAGS.keep_prob})
                summary_writer.add_summary(summary_str, step)
                print("time per batch is " + str(elapsed))
                print(step)
                ave_loss_r = loss_r/(100*100*FLAGS.batch_size*(FLAGS.seq_length-FLAGS.seq_start)*5)
                print(ave_loss_r)
                train_log.write("time per batch is " + str(elapsed) + '\n')
                train_log.write(str(step) + '\n')
                train_log.write(str(loss_r) + '  ave:' +str(ave_loss_r) + '\n')


            assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

            if step % 10000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("saved to " + FLAGS.train_dir)

                # Testing
                # test data
                # batch_data: (144, 100, 100, 5)
                test_data = test_data.reshape(FLAGS.batch_size, FLAGS.seq_length, 100, 100,
                                               5)  # (FLAGS.batch_size, FLAGS.seq_length, 100, 100, 5)

                telefuture, loss_test = sess.run([x_unwrap_g, pred_loss], feed_dict={x: test_data, keep_prob: FLAGS.keep_prob})

                print('test_step: '+str(step))
                ave_loss_test = loss_test/(100*100*FLAGS.batch_size*(FLAGS.seq_length-FLAGS.seq_start)*5)
                print('ave_loss_test: '+str(ave_loss_test))
                test_log.write(str(step) + '\n')
                test_log.write(str(loss_test) + '  ave:' +str(ave_loss_test) + '\n')

                print(telefuture.shape)
                diff = test_data[:, 11:, :, :, :] - telefuture[:, 10:, :, :, :]  # [12,2,10,10,5]
                f1 = np.sum(diff[:, :, :, :, 0]) / (12 * 2 * 10 * 10)
                f2 = np.sum(diff[:, :, :, :, 1]) / (12 * 2 * 10 * 10)
                f3 = np.sum(diff[:, :, :, :, 2]) / (12 * 2 * 10 * 10)
                f4 = np.sum(diff[:, :, :, :, 3]) / (12 * 2 * 10 * 10)
                f5 = np.sum(diff[:, :, :, :, 4]) / (12 * 2 * 10 * 10)
                test_log.write('Pred Mean: ' +str(f1) + str(',') + str(f2) +str(',') + str(f3) +str(',') + str(f4) +str(',') + str(f5) + '\n')
                # var
                v1 = np.std(test_data[:, :, :, :, 0])
                v2 = np.std(test_data[:, :, :, :, 1])
                v3 = np.std(test_data[:, :, :, :, 2])
                v4 = np.std(test_data[:, :, :, :, 3])
                v5 = np.std(test_data[:, :, :, :, 4])
                test_log.write(
                    'Y_true Mean: ' + str(v1) + str(',') + str(v2) + str(',') + str(v3) + str(',') + str(v4) + str(',') + str(v5) + '\n')
                print(f1 / v1, f2 / v2, f3 / v3, f4 / v4, f5 / v5)
                test_log.write('%f,%f,%f,%f,%f'%(f1 / v1, f2 / v2, f3 / v3, f4 / v4, f5 / v5))

        train_log.close()
        test_log.close()


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
