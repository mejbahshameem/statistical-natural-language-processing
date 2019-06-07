# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT IV

import math
import re
from collections import Counter
from matplotlib import pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import os


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

        sent = tokenize(sentence)
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

    for file in os.listdir(test):
        content = readsent(test+file)

        bigram_corp = []
        unigram_corp = []
        unigrams = Counter()
        bigrams = Counter()
        for line in content:
            unigram = word_ngrams(line, 1)
            bigram = word_ngrams(line, 2)
            unigram_corp.append(unigram)
            bigram_corp.append(bigram)
            unigrams.update(unigram)
            bigrams.update(bigram)
        print('\n perplexity of '+file+' for Dickens LMs')
        print('unigrams:' + str(dickunigramLM.perplexity(unigram_corp, 0.2)))
        print('bigrams:' + str(dickbigramLM.perplexity(bigram_corp, 0.2)))

        print('\n perplexity of '+file+' for Doyle LMs')
        print('unigrams:' + str(doyunigramLM.perplexity(unigram_corp, 0.2)))
        print('bigrams:' + str(doybigramLM.perplexity(bigram_corp, 0.2)))

        print('\n perplexity of '+file+' for Twain LMs')
        print('unigrams:' + str(twainunigramLM.perplexity(unigram_corp, 0.2)))
        print('bigrams:' + str(twainbigramLM.perplexity(bigram_corp, 0.2)))