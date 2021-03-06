import numpy as np
import os
import pandas as pd
import tensorflow as tf
import dataset
import model

trial = 1
n_channels = 64

save_path = 'checkpoints/lstm/trial'+str(trial)+'/'+str(n_channels)+'/'

lstm_size = 64 * 3  # 3 times the amount of channels
lstm_layers = 2  # Number of layers
batch_size = 80  # Batch size
seq_len = 160  # Number of steps
learning_rate = 0.001 # 0.0000001
epochs = 100

n_hidden_1 = 200  # 1st layer number of neurons
n_hidden_2 = 200  # 2nd layer number of neurons

n_input = lstm_size
n_classes = 109
keep_prob = 0.5

train_acc = []
train_loss = []

def train():

    tf.reset_default_graph()
    sess = tf.Session()

    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    inputs, labels, total_count = dataset.csv_inputs(batch_size, epochs, n_classes, n_channels, seq_len, trial)
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)
    total_count = tf.cast(total_count, tf.float32)
    logits = model.lstm_inference(inputs, keep_prob_, n_channels,
                                  lstm_size, lstm_layers, seq_len,
                                  batch_size, n_classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    tf.summary.scalar("cost", cost)

    train_op = tf.train.AdamOptimizer(learning_rate_)
    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    lg = tf.argmax(logits, 1)
    ll = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    #############################################################################
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    #############################################################################

    writer_train = tf.summary.FileWriter(save_path+'train_accuracy/', sess.graph)

    print("epoch looping")
    index = 0

    # Feed dictionary
    feed = {keep_prob_: keep_prob, learning_rate_: learning_rate}
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    e = 0
    loss_pre = 1000
    try:
        while not coord.should_stop():
            index += 1
            logits_val, labels_val, loss, _, acc, s_t, row_count = sess.run([lg, ll, cost, optimizer, accuracy, summ, total_count], feed_dict=feed)

            # print(logits_val)
            # print(labels_val)

            writer_train.add_summary(s_t, index)
            train_acc.append(acc)
            train_loss.append(loss)

            if index % np.floor(row_count / batch_size) == 0:
                e += 1

            if loss < loss_pre:
                saver.save(sess, save_path + 'save.ckpt')
                loss_pre = loss

            if loss < 0.0000001 and acc == 1:
                print('loss < 0.0000001')
                saver.save(sess, save_path + 'save.ckpt')
                break

            # Print at each 1000 iterations
            if index % 100 == 0:
                if acc > 0.5:
                    feed = {keep_prob_: keep_prob, learning_rate_: 0.001}
                    if acc > 0.7:
                        feed = {keep_prob_: keep_prob, learning_rate_: 0.0001}
                        if acc > 0.8:
                            feed = {keep_prob_: keep_prob, learning_rate_: 0.00001}

                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(index),
                      "Train loss: {:.10f}".format(loss),
                      "Train acc: {:.4f}".format(acc))

    except tf.errors.OutOfRangeError:
        print('epoch reached!')
    finally:
        print("Epoch: {}/{}".format(e, epochs),
              "Iteration: {:d}".format(index),
              "Final train loss: {:.10f}".format(loss_pre))
        coord.request_stop()
        coord.join(threads)

    sess.close()


def main():

    train()
    print('Done training!')
    print(trial)
    print('LSTM')


if __name__ == '__main__':
    main()
