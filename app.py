import sentence_classifier
import sentence_representation
import json
from flask import Flask, request
app = Flask(__name__)


@app.route("/sentence_representation/<message>")
def sentence_representation(message):
    return sentence_representation.sentence_representation(message=message)


@app.route("/train_classifier", methods=['GET', 'POST'])
def train_classifier():
    data = json.loads(request.form.get('data'))
    accuracy = sentence_classifier.train_classifier(data)
    return "Classifier trained with final training accuracy: " + str(accuracy)


@app.route("/classify_sentence/<sentence>", methods=['GET'])
def classify_sentence(sentence):
    intents = ['intent:greet', 'intent:restaurant_search', 'intent:weather_query', 'intent:goodbye']
    intent_index = sentence_classifier.classify_sentence(sentence)
    return intents[intent_index]
