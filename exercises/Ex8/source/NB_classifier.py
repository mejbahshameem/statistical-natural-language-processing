import re
import math
import numpy as np
from collections import Counter
from nltk.tokenize import ToktokTokenizer


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


class NB_Classifier:
    """docstring for NB_Classifier"""

    def __init__(self, V):
        self.classes = 4
        self.V = V
        self.D = [0, 0, 0, 0]
        self.N = [Counter(), Counter(), Counter(), Counter()]

    def learn(self, text, c):
        self.D[c] += 1
        self.N[c].update(text)

    def priors(self, c):
        return self.D[c] / sum(self.D)

    def likelihood(self, w, c):
        return 1 + self.N[c][w] / (len(self.V) + sum([self.N[c][word] for word in self.V]))

    def predict_class(self, d):
        return np.argmax([math.log(self.priors(c), 2) + sum([math.log(self.likelihood(word, c), 2) for word in d]) for c in range(self.classes)]) + 1


if __name__ == '__main__':
    if True:
        corpora = '../corpora/'
        toktok = ToktokTokenizer
        with open(corpora + 'ag_news_csv_cleaned/train_cleaned.csv', 'r', encoding='utf-8') as f:
            sent = toktok.tokenize(toktok, re.sub(
                r'[^a-zA-Z ]', ' ', f.read()), return_str=False)
            C = Counter(sent)
            V = [k for k, c in C.most_common() if c > 1]

        NB = NB_Classifier(V)

        with open(corpora + 'ag_news_csv_cleaned/train_cleaned.csv', 'r', encoding='utf-8') as f:
            for line in f:
                c = line[1]
                text = toktok.tokenize(toktok, re.sub(
                    r'[^a-zA-Z ]', ' ', line[5:]), return_str=False)
                NB.learn(text, int(c) - 1)

        with open(corpora + 'ag_news_csv_cleaned/test_cleaned.csv', 'r', encoding='utf-8') as f:
            cs = []
            ps = []
            conf = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            for line in f:
                c = int(line[1])
                text = toktok.tokenize(toktok, re.sub(
                    r'[^a-zA-Z ]', ' ', line[5:]), return_str=False)
                p = NB.predict_class(text)
                cs.append(c)
                ps.append(p)
                conf[c - 1][p - 1] += 1

        print(str(len([cs[i] for i in range(len(cs))
                       if cs[i] == ps[i]]) / len(cs) * 100) + '%')
        print(conf)
