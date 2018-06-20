import sys
import numpy as np


message = sys.argv[1]
embeddings_index = sys.argv[2]
words = message.split(" ")
sum = 0
for word in words:
    sum += abs(embeddings_index[word])

sum = np.mean(sum)
print(sum)
sys.stdout.flush()
