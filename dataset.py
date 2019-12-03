import h5py
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale

train_csv = [[1, 2, 7, 8, 9, 10, 11, 12, 13, 14],
             [1, 2, 3, 4, 5, 6, 11, 12, 13, 14],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_csv = [[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]

def _batch_normalization(tensor_in, epsilon=.0001):
    mean, variance = tf.nn.moments(tensor_in, axes=[0])
    tensor_normalized = (tensor_in-mean)/(variance+epsilon)
    return tensor_normalized

def _read_py_function(filename):
    if isinstance(filename, np.ndarray):
        mat = h5py.File(filename[0], 'r')
    else:
        mat = h5py.File(filename, 'r')
    labels = mat['temp_label']
    labels = np.squeeze(labels)
    eeg_data = mat['temp_data']

    # eeg_data = np.transpose(eeg_data, (1, 0))
    # eeg_data = scale(eeg_data, with_std=False, axis=1)

    trial_num = mat['r']

    return eeg_data, labels, trial_num[0]

def csv_inputs(batch_size, epochs, n_classes, n_channels, sample_size, trial):

    if trial == 3:
        trial = 0
    cwd = os.getcwd()
    path = []
    row_count = 0
    for index in train_csv[trial]:
        csv_name = 'csv/R' + str(index) + '.csv'
        path.append(os.path.join(cwd, csv_name))
        fileobject = open(os.path.join(cwd, csv_name), 'r')
        row_count = row_count+sum(1 for row in fileobject)

    filename_queue = tf.train.string_input_producer(path, num_epochs=epochs, shuffle=True)
    reader = tf.TextLineReader()
    key, serialized_example = reader.read(filename_queue)
    record_defaults = [["path"]]
    filename = tf.decode_csv(serialized_example, record_defaults=record_defaults)

    eeg_data, label, trial_num = tf.py_func(_read_py_function, [filename], [np.float64, np.float64, np.float64])
    eeg_data.set_shape([sample_size, n_channels])

    eeg_data = _batch_normalization(eeg_data, epsilon=.0001)

    label.set_shape([n_classes])
    trial_num.set_shape([1])

    eeg_datas, labels, trial_nums = tf.train.shuffle_batch(
        [eeg_data, label, trial_num],
        batch_size=batch_size,
        num_threads=5,
        capacity=100+3*batch_size,
        min_after_dequeue=100
    )
    return eeg_datas, labels, row_count

def csv_test(batch_size, n_classes, n_channels, sample_size, trial):

    if trial == 3:
        trial = 0

    cwd = os.getcwd()
    path = []
    row_count = 0

    for index in test_csv[trial]:
        csv_name = 'csv/R' + str(index) + '.csv'
        path.append(os.path.join(cwd, csv_name))
        fileobject = open(os.path.join(cwd, csv_name), 'r')
        row_count = row_count + sum(1 for row in fileobject)

    filename_queue = tf.train.string_input_producer(path, num_epochs=1, shuffle=False)
    reader = tf.TextLineReader()
    key, serialized_example = reader.read(filename_queue)
    record_defaults = [["path"]]
    filename = tf.decode_csv(serialized_example, record_defaults=record_defaults)

    eeg_data, label, trial_num = tf.py_func(_read_py_function, [filename], [np.float64, np.float64, np.float64])
    eeg_data.set_shape([sample_size, n_channels])

    eeg_data = _batch_normalization(eeg_data, epsilon=.0001)

    label.set_shape([n_classes])
    trial_num.set_shape([1])

    eeg_datas, labels, trial_nums = tf.train.batch(
        [eeg_data, label, trial_num],
        batch_size=batch_size,
        num_threads=4,
        capacity=100+3 * batch_size,
        allow_smaller_final_batch=False
    )
    return eeg_datas, labels, trial_nums, row_count

