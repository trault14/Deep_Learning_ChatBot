import sentence_classifier
from embeddings_load import load_embeddings_table
from flask import Flask
app = Flask(__name__)


embeddings_table = load_embeddings_table()


@app.route("/sentence_representation/<message>")
def sentence_representation(message):
    words = list(map(str.lower, message.split("_")))
    s = []
    for word in words:
        if word in embeddings_table.keys():
            s.append(embeddings_table[word].reshape(50, 1))

    return s


@app.route("/train_classifier", methods=['GET', 'POST'])
def train_classifier():
    data = {
        "intent:greet":
            ["hey", "howdy", "hey there", "hello", "hi", "good morning", "good evening",  "dear sir"],
        "intent:restaurant_search":
            ["i'm looking for a place to eat", "I want to grab lunch", "I am searching for a dinner spot",
             "i'm looking for a place in the [north](location) of town", "show me [chinese](cuisine) restaurants",
             "show me [chinese](cuisine) restaurants in the [north](location)",
             "show me a [mexican](cuisine) place in the [centre](location)",
             "i am looking for an [indian](cuisine) spot called olaolaolaolaolaola", "search for restaurants",
             "anywhere in the [west](location)", "anywhere near [18328](location)",
             "I am looking for [asian fusion](cuisine) food", "I am looking a restaurant in [29432](location)",
             "I am looking for [mexican indian fusion](cuisine)", "[central](location) [indian](cuisine) restaurant"],
        "intent:weather_query":
            ["is it warm outside", "what's the weather like in [paris](location)",
             "is it raining in [new york](location) [today](time)",
             "will it snow in [london](location) [tomorrow](time)",
             "I wonder what the temperature will be like today", "How cold is it now"],
        "intent:goodbye":
            ["bye", "goodbye", "good bye", "stop", "end", "farewell", "Bye bye", "have a good one"]
    }
    accuracy = sentence_classifier.train_classifier(data)
    return "Classifier trained with final training accuracy: " + str(accuracy)


@app.route("/classify_sentence/<sentence>", methods=['GET'])
def classify_sentence(sentence):
    intents = ['intent:greet', 'intent:restaurant_search', 'intent:weather_query', 'intent:goodbye']
    intent_index = sentence_classifier.classify_sentence(sentence)
    return intents[intent_index]
