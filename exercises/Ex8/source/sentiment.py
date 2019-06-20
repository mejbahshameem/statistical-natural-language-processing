import csv
import random
import re
import math
import numpy as np
from collections import Counter
from tqdm import tqdm


N_CLASSES = 2


class NB_Classifier:

    def __init__(self, V):
        self.classes = N_CLASSES
        self.V = V
        self.D = {i: 0 for i in range(1, self.classes + 1)} #the total number of documents labelled with class k
        self.N = {i: Counter() for i in range(1, self.classes + 1)} #counter of the documents of class k
        self.c_size = {i: 0 for i in range(1, self.classes + 1)} #number of words from V in entire class i

    def learn(self, text, c):
        self.D[c] += 1
        self.N[c].update(text)

    def priors(self, c):
        return self.D[c] / sum(self.D.values())

    def likelihood(self, w, c):
        if self.c_size[c] == 0:
            self.c_size[c] += sum([self.N[c][word] for word in self.V])
        return 1 + self.N[c][w] / (len(self.V) + self.c_size[c])

    def predict_class(self, d):
        return np.argmax([math.log(self.priors(c), 2)
                          + sum([math.log(self.likelihood(word, c), 2)
                                 for word in d]) for c in range(1, self.classes + 1)]) + 1


if __name__ == '__main__':
    folder = '../corpora/imdb_movie_reviews/'
    with open(folder + 'imdb_dataset.csv', encoding='utf-8', newline='') as file:
        samples = csv.reader(file)
        samples = [(int(row[1]), re.findall('[a-z]+', row[0].lower())) for row in samples]
        random.shuffle(samples)

    k = 10
    valid_size = int(np.floor(len(samples) / k))

    accuracy = [0] * k
    for i in tqdm(range(k)):
        test_set = samples[i * valid_size: (i + 1) * valid_size]
        train_set = samples[: i * valid_size] + samples[(i + 1) * valid_size:]

        vocab = Counter()
        for sample in train_set:
            vocab.update(sample[1])
        vocab = [word for word, count in vocab.items() if count > 1]
        model = NB_Classifier(vocab)

        # Model training
        for cat, text in train_set:
            model.learn(text, cat)

        # Model testing
        for cat, text in test_set:
            if model.predict_class(text) == cat:
                accuracy[i] += 1 / len(test_set)

    mean = sum(accuracy) / k
    std_dev = sum([math.sqrt((acc - mean) ** 2 / (k - 1)) for acc in accuracy])

    print(f'Accuracy for each fold: {[round(acc, 4) for acc in accuracy]}')
    print(f'Mean: {mean}')
    print(f'Standard deviation: {std_dev}')


