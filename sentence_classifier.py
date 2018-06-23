import tensorflow as tf
from random import shuffle
from utils import fully_connected, binary_cross_entropy
import numpy as np

# Define placeholder for the data
X = tf.placeholder(name='X', shape=[None, 15, 50, 1], dtype=tf.float32)
# Placeholder for the labels
Y = tf.placeholder(name='Y', shape=[None, 4], dtype=tf.float32)

H = X
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
n_epochs = 25

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def train_classifier(sentences, intents):
    # Pad the sentences that contain less than 15 words with arrays of zeros
    for i in range(len(sentences)):
        for j in range(15 - len(sentences[i])):
            sentences[i].append(np.array([0] * 50).reshape(50, 1))

    # Train the network
    for epoch_i in range(n_epochs):
        # Shuffle the order of the sentences
        order = [e for e in range(len(sentences))]
        shuffle(order)
        Xs, ys = [], []
        for shuffled_index in order:
            Xs.append(sentences[shuffled_index])
            ys.append(intents[shuffled_index])
        Xs = np.array(Xs)
        Xs.reshape(len(Xs), 15, 50, 1)
        ys = np.array(ys)

        this_accuracy = sess.run([accuracy, optimizer], feed_dict={X: Xs, Y: ys})[0]
        print('Epoch:', epoch_i, ' Accuracy:', this_accuracy)
    return this_accuracy


def classify_sentence(represented_sentence):
    for j in range(15 - len(represented_sentence)):
        represented_sentence.append(np.array([0] * 50).reshape(50, 1))
    represented_sentence = np.array(represented_sentence)
    represented_sentence.reshape(1, 15, 50, 1)
    represented_sentence = np.array([represented_sentence])
    return sess.run(tf.argmax(tf.reduce_mean(Y_predicted, 0), 0), feed_dict={X: represented_sentence})
