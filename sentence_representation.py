import embeddings_load


embeddings_table = embeddings_load.load_embeddings_table()


def sentence_representation(message):
    words = list(map(str.lower, message.split("_")))
    s = []
    for word in words:
        if word in embeddings_table.keys():
            s.append(embeddings_table[word].reshape(50, 1))

    return s
