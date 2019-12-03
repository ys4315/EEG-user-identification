import numpy as np
import os
import pandas as pd
import tensorflow as tf
import dataset
import model
import time
import argparse


parser = argparse.ArgumentParser(description='CLDNN testing')
parser.add_argument('trial', help='trial number')
args = parser.parse_args()

trial = int(args.trial)
n_channels = 64

save_path = 'checkpoints/cldnn/trial'+str(trial)+'/'+str(n_channels)+'/'
result_path = 'results/cldnn/trial'+str(trial)+'/'+str(n_channels)+'/'

lstm_size = 64 * 3  # 3 times the amount of channels
lstm_layers = 2  # Number of layers
batch_size = 80  # Batch size
seq_len = 160  # Number of steps
learning_rate = 0.0001
epochs = 100

n_hidden_1 = 200  # 1st layer number of neurons
n_hidden_2 = 200  # 2nd layer number of neurons

n_input = lstm_size
n_classes = 109
keep_prob = 0.5

test_acc = []
test_loss = []
test_labels = []
predictions = []
probabilities = []

def test():

    t = time.time()
    tf.reset_default_graph()
    sess = tf.Session()

    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    inputs, labels, trial_num, total_count = dataset.csv_test(batch_size, n_classes, n_channels, seq_len, trial)
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)
    total_count = tf.cast(total_count, tf.uint16)

    logits = model.cldnn_inference(inputs, keep_prob_, weights, biases, lstm_size, lstm_layers, seq_len, batch_size,
                                   n_classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint(save_path))

    print("epoch looping")
    index = 0

    feed = {keep_prob_: 1, learning_rate_: 1}
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    elapsed = time.time() - t
    t1 = time.time()

    try:
        while not coord.should_stop():
            index += 1
            batch_acc, batch_loss, batch_logits, batch_labels, total_counts = sess.run([accuracy, cost, logits, labels, total_count],
                                                                           feed_dict=feed)

            probabilities.append(tf.nn.softmax(batch_logits).eval(session=sess))
            test_labels.append(tf.argmax(batch_labels, 1).eval(session=sess))
            predictions.append(tf.argmax(tf.nn.softmax(batch_logits), 1).eval(session=sess))
            test_acc.append(batch_acc)
            test_loss.append(batch_loss)

            print("Iteration: {}/{}".format(index, np.floor(total_counts/batch_size).astype(int)),
                  "Batch test accuracy: {:.6f}".format(batch_acc))

    except tf.errors.OutOfRangeError:
        print('epoch reached!')
    finally:
        coord.request_stop()
        coord.join(threads)

    elapsed1 = time.time() - t1
    clabels = np.concatenate(test_labels, axis=0)
    cpredictions = np.concatenate(predictions, axis=0)
    cprobabilities = np.concatenate(probabilities, axis=0)
    confusion_matrix = tf.confusion_matrix(labels=clabels, predictions=cpredictions).eval(session=sess)
    sess.close()

    print("Mean test accuracy: {:.6f}".format(np.mean(test_acc)))

    df = pd.DataFrame(confusion_matrix)
    df.to_csv(result_path+'confusion_matrix.csv')
    df1 = pd.DataFrame(clabels)
    df1.to_csv(result_path+'labels.csv', header=None, index=None)
    df2 = pd.DataFrame(cpredictions)
    df2.to_csv(result_path+'predictions.csv', header=None, index=None)
    df3 = pd.DataFrame(test_acc)
    df3.to_csv(result_path+'test_acc.csv', header=None, index=None)
    df4 = pd.DataFrame(cprobabilities)
    df4.to_csv(result_path+'probabilities.csv', header=None, index=None)
    df5 = pd.DataFrame(test_loss)
    df5.to_csv(result_path + 'test_loss.csv', header=None, index=None)
    df6 = pd.DataFrame(['CLDNN', trial, np.mean(test_acc), np.mean(test_loss), elapsed, elapsed1/index])
    df6.to_csv(result_path + 'result.csv', header=None, index=None)

    # rows_sums = confusion_matrix.sum(axis=1)
    # normalised_confusion_matrix = confusion_matrix/rows_sums[:, np.newaxis]
    print(confusion_matrix)
    print(trial)
    print('CLDNN')
    print('mean test accuracy: ', np.mean(test_acc))
    print('mean test loss: ', np.mean(test_loss))
    print('T_model: ', elapsed)
    print('T_batch: ', elapsed1/index)

def main():
    test()


if __name__ == '__main__':
    main()
