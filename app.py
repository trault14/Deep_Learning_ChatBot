import sentence_classifier
import numpy as np
import json
from flask import Flask, request
app = Flask(__name__)


@app.route("/train_classifier", methods=['GET', 'POST'])
def train_classifier():
    data = json.loads(request.form.get('data'))
    intents = data[0]
    sentences = np.array(data[1])
    sentences = list(map(lambda x: np.array(x).reshape(len(x), 50, 1).tolist(), sentences))
    accuracy = sentence_classifier.train_classifier(sentences, intents)
    return "Classifier trained with final training accuracy: " + str(accuracy)


@app.route("/classify_sentence", methods=['POST'])
def classify_sentence():
    represented_sentence = json.loads(request.form.get('data'))
    represented_sentence = list(map(lambda x: np.array(x).reshape(50, 1), represented_sentence))
    intents = ['intent:greet', 'intent:restaurant_search', 'intent:weather_query', 'intent:goodbye']
    intent_index = sentence_classifier.classify_sentence(represented_sentence)
    return intents[intent_index]
