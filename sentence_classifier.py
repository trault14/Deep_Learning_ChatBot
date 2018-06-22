import tensorflow as tf
from random import shuffle
from sentence_representation import sentence_representation
from utils import fully_connected, binary_cross_entropy, conv2d
import numpy as np

# Define placeholder for the data
X = tf.placeholder(name='X', shape=[None, 15, 50, 1], dtype=tf.float32)
# Placeholder for the labels
Y = tf.placeholder(name='Y', shape=[None, 4], dtype=tf.float32)

H = X
"""
# Number of filters and layers of the CNN
n_filters = [3, 2, 1]
for layer_i, n_filters_i in enumerate(n_filters):
    H, W = conv2d(H, n_filters_i, k_h=3, k_w=3, d_h=1, d_w=1, name=str(layer_i))
    H = tf.nn.relu(H)
    if layer_i % 2 == 1:
        H = tf.layers.max_pooling2d(H, pool_size=(2, 2), strides=(1, 1), padding='SAME', name=str(layer_i))
"""

# Number of filters and layers of the FCN
layers = [100, 100, 4]
for layer_i, n_output_i in enumerate(layers):
    H, W = fully_connected(H, n_output=n_output_i, name=layer_i)
    if layer_i == len(layers) - 1:
        H = tf.nn.softmax(H)
    else:
        H = tf.nn.relu(H)

Y_predicted = H

# Cost function
loss = binary_cross_entropy(Y_predicted, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

# Measure of accuracy
predicted_y = tf.argmax(Y_predicted, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Training parameters
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
n_epochs = 200

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def train_classifier(data):
    x, y = [], []
    for intent, sentences in data.items():
        x.extend(sentences)
        y.extend([intent] * len(sentences))

    # Format the sentences
    x = format_data(x)

    # Convert the sentences to their numerical representation
    x = list(map(sentence_representation, x))
    # Pad the sentences that contain less than 15 words with arrays of zeros
    for i in range(len(x)):
        for j in range(15 - len(x[i])):
            x[i].append(np.array([0] * 50).reshape(50, 1))

    # Convert the labels to their numerical values : one-hot value of the position in the 'intents' list
    intents = ["intent:greet", "intent:restaurant_search", "intent:weather_query", "intent:goodbye"]
    for i, intent in enumerate(y):
        converted_intent = [0, 0, 0, 0]
        converted_intent[intents.index(intent)] = 1
        y[i] = converted_intent

    # Train the network
    for epoch_i in range(n_epochs):
        # Shuffle the order of the sentences
        order = [e for e in range(len(x))]
        shuffle(order)
        Xs, ys = [], []
        for shuffled_index in order:
            Xs.append(x[shuffled_index])
            ys.append(y[shuffled_index])
        Xs = np.array(Xs)
        Xs.reshape(len(Xs), 15, 50, 1)
        ys = np.array(ys)

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


def format_data(x):
    x = list(map(lambda s: s.replace(" ", "_"), x))
    x = list(map(lambda s: s.replace("[", ""), x))
    x = list(map(lambda s: s.replace("]", ""), x))
    x = list(map(lambda s: s.replace("(location)", ""), x))
    x = list(map(lambda s: s.replace("(cuisine)", ""), x))
    x = list(map(lambda s: s.replace("(time)", ""), x))
    return x
