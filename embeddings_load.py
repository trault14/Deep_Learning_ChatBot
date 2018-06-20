import numpy as np


def load_embeddings_table():
    # load the whole embedding into memory
    table = dict()
    file = open('/Users/timrault/PycharmProjects/hf-code-exercise/glove.6B/glove.6B.50d.txt')
    for line in file:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        table[word] = coefficients
    file.close()
    print('Loaded %s word vectors.' % len(table))
    return table
