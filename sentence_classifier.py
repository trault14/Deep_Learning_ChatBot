import tensorflow as tf
from random import shuffle
from app import sentence_representation
from utils import fully_connected, binary_cross_entropy
import numpy as np

# Define placeholder for the data
X = tf.placeholder(name='X', shape=[None, 15, 50, 1], dtype=tf.float32)
# Placeholder for the labels
Y = tf.placeholder(name='Y', shape=[None, 4], dtype=tf.float32)

H = X
# Number of filters and layers of the CNN
n_filters = [9, 9, 9, 9]
"""
for layer_i, n_filters_i in enumerate(n_filters):
    H, W = conv2d(H, n_filters_i, k_h=3, k_w=3, d_h=1, d_w=1, name=str(layer_i))
    H = tf.nn.relu(H)
    if layer_i % 2 == 1:
        H = tf.layers.max_pooling2d(H, pool_size=(3, 3), strides=(2, 2), padding='SAME', name=str(layer_i))
"""

# Number of filters and layers of the FCN
layers = [16, 16, 4]
for layer_i, n_output_i in enumerate(layers):
    H, W = fully_connected(H, n_output=n_output_i, name=layer_i)
    if layer_i == len(layers) - 1:
        H = tf.nn.softmax(H)
    else:
        H = tf.nn.relu(H)


"""
for layer_i, n_output_i in enumerate(layers):
    with tf.variable_scope("FullyConnected" + str(layer_i), reuse=tf.AUTO_REUSE):
        if len(H.get_shape()) != 2:
            H = flatten(H, reuse=tf.AUTO_REUSE)

        W = tf.get_variable(
            name='W',
            shape=[H.get_shape().as_list()[1], n_output_i],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        b = tf.get_variable(
            name='b',
            shape=[n_output_i],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )

        H = tf.nn.bias_add(
            name='h',
            value=tf.matmul(H, W),
            bias=b
        )

        if layer_i == len(layers) - 1:
            H = tf.nn.softmax(H)
        else:
            H = tf.nn.relu(H)
"""
Y_predicted = H

# Cost function
loss = binary_cross_entropy(Y_predicted, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

predicted_y = tf.argmax(Y_predicted, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

n_epochs = 200

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def train_classifier(data):
    x, y = [], []
    for intent, sentences in data.items():
        x.extend(sentences)
        y.extend([intent] * len(sentences))

    order = [e for e in range(len(x))]
    shuffle(order)
    Xs, ys = [], []
    for shuffled_index in order:
        Xs.append(x[shuffled_index])
        ys.append(y[shuffled_index])

    intents = ["intent:greet", "intent:restaurant_search", "intent:weather_query", "intent:goodbye"]
    Xs = list(map(lambda s: s.replace(" ", "_"), Xs))
    Xs = list(map(lambda s: s.replace("[", ""), Xs))
    Xs = list(map(lambda s: s.replace("]", ""), Xs))
    Xs = list(map(lambda s: s.replace("(location)", ""), Xs))
    Xs = list(map(lambda s: s.replace("(cuisine)", ""), Xs))
    Xs = list(map(lambda s: s.replace("(time)", ""), Xs))

    Xs = list(map(sentence_representation, Xs))
    for i, x in enumerate(Xs):
        for j in range(15 - len(Xs[i])):
            Xs[i].append(np.array([0] * 50).reshape(50, 1))

    for i, intent in enumerate(ys):
        converted_intent = [0, 0, 0, 0]
        converted_intent[intents.index(intent)] = 1
        ys[i] = converted_intent

    # Xs = np.array(Xs).reshape(len(Xs), 1)
    Xs = np.array(Xs)
    Xs.reshape(37, 15, 50, 1)
    ys = np.array(ys)
    for epoch_i in range(n_epochs):
        this_accuracy = sess.run([accuracy, optimizer], feed_dict={X: Xs, Y: ys})[0]
        print('Epoch:', epoch_i, ' Accuracy:', this_accuracy)
    return this_accuracy


def classify_sentence(sentence):
    sentence = sentence_representation(sentence)
    for j in range(15 - len(sentence)):
        sentence.append(np.array([0] * 50).reshape(50, 1))
    sentence = np.array(sentence)
    sentence.reshape(1, 15, 50, 1)
    sentence = np.array([sentence])
    return sess.run(tf.argmax(tf.reduce_mean(Y_predicted, 0), 0), feed_dict={X: sentence})
