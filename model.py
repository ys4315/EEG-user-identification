import tensorflow as tf

def cldnn_inference(inputs, keep_prob, weights, biases, lstm_size, lstm_layers, seq_len, batch_size, n_classes):

    conv1 = tf.layers.conv1d(inputs=inputs, filters=128, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    conv2 = tf.layers.conv1d(inputs=conv1, filters=256, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    conv3 = tf.layers.conv1d(inputs=conv2, filters=512, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    conv4 = tf.layers.conv1d(inputs=conv3, filters=1024, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    n_ch = 64 * 16

    # Construct the LSTM inputs and LSTM cells
    lstm_in = tf.transpose(conv4, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*N, n_channels)
    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu
    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)
    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state=initial_state)

    layer_1 = tf.add(tf.matmul(outputs[-1], weights['h1']), biases['b1'], name='MLP1')
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='MLP2')
    logits = tf.matmul(layer_2, weights['out']) + biases['out']

    # logits = tf.layers.dense(outputs[-1], n_classes, name='logits')
    return logits

def cnn_inference(inputs, keep_prob, n_classes):

    conv1 = tf.layers.conv1d(inputs=inputs, filters=128, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=256, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=512, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=1024, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

    flat = tf.reshape(max_pool_4, (-1, 64 * 160))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob)

    # Predictions
    logits = tf.layers.dense(flat, n_classes)

    return logits

def lstm_inference(inputs, keep_prob, n_ch, lstm_size, lstm_layers, seq_len, batch_size, n_classes):

    # Construct the LSTM inputs and LSTM cells
    lstm_in = tf.transpose(inputs, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*N, n_channels)
    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu
    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)
    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state=initial_state)

    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

    return logits
