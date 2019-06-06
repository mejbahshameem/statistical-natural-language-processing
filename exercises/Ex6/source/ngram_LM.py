# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT VI

import re
import math
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from prettytable import PrettyTable, ALL


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())

def getLM(file):
    content = readsent(file)

    vocab = tokenize(' '.join(content))
    VOCAB = set(vocab)

    unigram_COUNTS = Counter()
    bigram_COUNTS = Counter()
    for line in content:
        unigram_COUNTS.update(word_ngrams(line, 1))
        bigram_COUNTS.update(word_ngrams(line, 2))

    unigram_LM = ngram_LM(1, unigram_COUNTS, VOCAB)
    VOCAB.update(('<s>', '</s>'))
    bigram_LM = ngram_LM(2, bigram_COUNTS, VOCAB)

    return unigram_LM, bigram_LM


def readsent(filename):
    file = open(filename, 'r', encoding='utf-8', errors='ignore')
    f = file.read()
    file.close()
    return sent_tokenize(f)


def word_ngrams(sent, n):
    """Givne a sent as str return n-grams as a list of tuple"""

    sent = tokenize(sent)
    if n > 1:
        sent.insert(0, '<s>')
        sent.append('</s>')
    ngrams = []
    if n == 1:
        for i in range(len(sent)):
            ngrams.append((sent[i],))
    else:
        for i in range(1, len(sent)):
            ngrams.append((sent[i - 1], sent[i]))

    return ngrams


class ngram_LM:
    """A class to represent a language model."""

    def __init__(self, n, ngram_counts, vocab, unk=False):
        """"Make a n-gram language model, given a vocab and
            data structure for n-gram counts."""

        self.n = n
        self.vocab = vocab
        self.V = len(vocab)
        self.ngram_counts = ngram_counts
        if n > 1:
            self.unigram_counts = Counter()
            for word in self.ngram_counts.items():
                self.unigram_counts.update({(word[0][0],): word[1]})

            assert sum(ngram_counts.values()) == sum(
                self.unigram_counts.values())

    def perplexity(self, T, alpha):
        res = 0
        M = 0
        for sent in T:
            M += len(sent)
            for word in sent:
                res += self.logP(word[0], word[-1 + self.n], alpha=alpha)
        return 2**(-res / M)

    def unseen(self, T):
        unseen = 0
        M = 0
        for word in T.keys():
            if self.ngram_counts[word] == 0:
                unseen += 1
        return unseen / len(T)

    def estimate_prob(self, history, word):
        if self.n == 1:
            return self.ngram_counts[(word,)] / sum(self.ngram_counts.values())

        if self.n > 1:
            return self.ngram_counts[(history, word)] / self.unigram_counts[(history,)]

    def estimate_smoothed_prob(self, history, word, alpha=0.5):
        """
        Estimate probability of a word given a history with Lidstone smoothing.

        :param history: a predecessor word: '' - for n=1 and string - for n=2
        :param word: a word conditioned on a predecessor word; string
        :param alpha: a smoothing parameter in (0,1]; float
        :return: the probability of a word conditioned on a predecessor word; float
        """
        if self.n == 1:
            return (alpha + self.ngram_counts[(word,)]) / \
                   (alpha * self.V + sum(self.ngram_counts.values()))

        if self.n > 1:
            return (alpha + self.ngram_counts[(history, word)]) / \
                   (alpha * self.V + self.unigram_counts[(history,)])

    def logP(self, history, word, alpha=0.5):
        """Return base-2 log probablity."""

        prob = self.estimate_smoothed_prob(history, word, alpha=alpha)
        return math.log(prob, 2)

    def score_sentence(self, sentence):
        """Given a sentence, return score."""

        sent = sent_tokenize(sentence)
        sent.insert(0, '<s>')
        sent.append('</s>')
        M = len(sent)
        score = 0
        for i in range(1, M):
            score += - self.logP(sent[i - 1], sent[i])
        return score / M

    def prob_dist(self, h):
        prob = dict()
        for word in self.vocab:
            prob[word] = self.estimate_prob(h, word)
        return sorted(prob.items(), reverse=True, key=lambda kv: (kv[1], kv[0]))

    def test_LM(self):
        """Test whether or not the probability mass sums up to one."""

        print('\nTEST STARTED FOR n = ' + str(self.n))

        precision = 10 ** -8

        if self.n == 1:

            P_sum = sum(self.estimate_prob('', w) for w in self.vocab)
            assert abs(
                1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'

        elif self.n == 2:
            histories = ['the', 'in', 'at', 'blue', 'white']

            for h in histories:
                P_sum = sum(self.estimate_prob(h, w) for w in self.vocab)

            for h in histories:
                P_sum = sum(self.estimate_prob(h, w) for w in self.vocab)
                assert abs(
                    1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h

        print('Test successful!')

    def test_smoohted_LM(self):
        """
        Test whether or not the smoothed probability mass sums up to one.
        """
        self.n = self.n
        precision = 10 ** -7
        print('\nTEST SMOOTHED LM STARTED FOR n = ' + str(self.n))

        if self.n == 1:
            P_sum = sum(self.estimate_smoothed_prob('', w) for w in self.vocab)
            assert abs(
                1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'

        elif self.n == 2:
            histories = ['the', 'in', 'at', 'blue', 'white']
            for h in histories:
                P_sum = sum(self.estimate_smoothed_prob(h, w)
                            for w in self.vocab)
                assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history "{}"'.format(
                    h)

        print('Test successful!')


def PMI(ngram, author):
    authors = ['dickens', 'doyle', 'twain']
    unigram_LMs = [dickunigramLM, doyunigramLM, twainunigramLM]
    bigram_LMs = [dickbigramLM, doybigramLM, twainbigramLM]
    if len(ngram) == 1:
        ngram_h, ngram_w = '', ngram[0]
        LM = unigram_LMs[authors.index(author)]
        P_f = dickunigramLM.estimate_prob(ngram_h, ngram_w) \
              + doyunigramLM.estimate_prob(ngram_h, ngram_w) \
              + twainunigramLM.estimate_prob(ngram_h, ngram_w)
    else:
        ngram_h, ngram_w = ngram[0], ngram[1]
        LM = bigram_LMs[authors.index(author)]
        P_f = dickbigramLM.estimate_prob(ngram_h, ngram_w) \
                  + doybigramLM.estimate_prob(ngram_h, ngram_w) \
                  + twainbigramLM.estimate_prob(ngram_h, ngram_w)

    P_c = 1 / 3
    P_f_c = LM.estimate_prob(ngram_h, ngram_w)
    return math.log(P_f_c / P_c / P_f)


if __name__ == '__main__':
    nltk.download('punkt')
    train = '../corpora/train/'
    test = '../corpora/test/'
    dickunigramLM, dickbigramLM = getLM(train + 'dickens.en.train')
    doyunigramLM, doybigramLM = getLM(train + 'doyle.en.train')
    twainunigramLM, twainbigramLM = getLM(train + 'twain.en.train')

    print('\n dickens unigrams')
    print(dickunigramLM.ngram_counts.most_common(15))

    print('\n dickens bigrams')
    print(dickbigramLM.ngram_counts.most_common(15))

    print('\n doyle unigrams')
    print(doyunigramLM.ngram_counts.most_common(15))

    print('\n doyle bigrams')
    print(doybigramLM.ngram_counts.most_common(15))

    print('\n twain unigrams')
    print(twainunigramLM.ngram_counts.most_common(15))

    print('\n twain bigrams')
    print(twainbigramLM.ngram_counts.most_common(15))

    # 2.2 Feature Selection

    unigram_LMs = [dickunigramLM, doyunigramLM, twainunigramLM]
    bigram_LMs = [dickbigramLM, doybigramLM, twainbigramLM]

    for LMs in [unigram_LMs, bigram_LMs]:
        counter = Counter()
        for LM in LMs:
            for ngram_LM in LM.ngram_counts.most_common(15):
                counter[ngram_LM[0]] += ngram_LM[1]

        authors = ['dickens', 'doyle', 'twain']
        PMI_dickens = {}
        PMI_doyle = {}
        PMI_twain = {}
        for ngram, count in counter.items():
            if count > 15:
                PMI_dickens[ngram] = round(PMI(ngram, 'dickens'), 4)
                PMI_doyle[ngram] = round(PMI(ngram, 'doyle'), 4)
                PMI_twain[ngram] = round(PMI(ngram, 'twain'), 4)

        dickens = sorted(PMI_dickens.items(), key=lambda kv: -kv[1])[:10]
        doyle = sorted(PMI_doyle.items(), key=lambda kv: -kv[1])[:10]
        twain = sorted(PMI_twain.items(), key=lambda kv: -kv[1])[:10]

        t = PrettyTable(hrules=ALL)
        t.title = 'Top 10 ngram features for each author'
        column_names = ['#', 'Dickens', 'Doyle', 'Twain']
        t.add_column(column_names[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        t.add_column(column_names[1], dickens)
        t.add_column(column_names[2], doyle)
        t.add_column(column_names[3], twain)
        print(t)
