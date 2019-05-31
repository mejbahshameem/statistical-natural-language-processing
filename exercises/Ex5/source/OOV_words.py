import re
import random
import math
from collections import Counter
from matplotlib import pyplot as plt


class CorporaHandler:
    def __init__(self, text):
        self.text_tokens = self.tokenize(text)
        self.train_corpus, self.test_corpus = self.split_train_test()
        self.vocabulary = self.construct_vocabulary()

    @staticmethod
    def tokenize(text):
        return re.findall(r'\b\w+', re.sub(r'\d', '', text))

    def split_train_test(self):
        random.shuffle(self.text_tokens)
        train_corpus = self.text_tokens[:int((len(self.text_tokens) + 1) * .80)]
        test_corpus = self.text_tokens[int((len(self.text_tokens) + 1) * .80):]
        return train_corpus, test_corpus

    def construct_vocabulary(self):
        vocabulary = Counter(self.train_corpus)
        return vocabulary.most_common(15000)

    def oov_rate(self, vocab_size=15000):
        vocabulary = dict(self.vocabulary[:vocab_size])
        unseen = 0
        for token in self.test_corpus:
            if token not in vocabulary:
                unseen += 1
        return unseen / len(self.test_corpus)


if __name__ == '__main__':
    print('Computing OOV-rate...')
    plt.figure(1)
    plt.xlabel('Vocabulary size, log')
    plt.ylabel('OOV-rate, log')

    path = '../corpora'
    languages = [('.fi', 'Finnish'), ('.de', 'German'), ('.bg', 'Bulgarian'),
                 ('.el', 'Greek'), ('.mt', 'Maltese'), ('.fr', 'French')]
    for lang in languages:
        print(f'...{lang[1]}')
        with open(path + '/corpus' + lang[0], 'r', encoding='UTF-8') as file:
            corpora = file.read()

        handler = CorporaHandler(corpora)

        oov_rate_list = []
        partition = range(1000, 15001, 1000)
        for p in partition:
            oov = handler.oov_rate(vocab_size=p)
            oov_rate_list.append(math.log(oov * 100))
        x = [math.log(i) for i in partition]
        plt.plot(x, oov_rate_list, label=lang[1])

    print('Plotting the graph...')
    plt.legend(loc='upper right', shadow=True)
    plt.savefig('oov_rate.png')
    plt.show()
