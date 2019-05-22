# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT IV

import math
import re
from collections import Counter


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


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
        for i in range(n, len(sent)):
            ngrams.append((sent[i-1], sent[i]))

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

            assert sum(ngram_counts.values()) == sum(self.unigram_counts.values())

    def estimate_prob(self, history, word):
        if self.n == 1:
            return self.ngram_counts[(word,)]/sum(self.ngram_counts.values())

        if self.n > 1:
            return self.ngram_counts[(history,word)]/self.unigram_counts[(history,)]

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

    def logP(self, history, word):
        """Return base-2 log probablity."""

        prob = self.estimate_smoothed_prob(history, word)
        return math.log(prob, 2)

    def score_sentence(self, sentence):
        """Given a sentence, return score."""
        
        sent = tokenize(sentence)
        sent.insert(0, '<s>')
        sent.append('</s>')
        M = len(sent)
        score = 0
        for i in range(1, M):
            score += - self.logP(sent[i-1], sent[i])
        return score / M
 
    def prob_dist(self, h):
        prob = dict()
        for word in self.vocab:
            prob[word] = self.estimate_prob(h, word)
        return sorted(prob.items(), reverse=True, key=lambda kv: (kv[1], kv[0]))

    def test_LM(self):
        """Test whether or not the probability mass sums up to one."""
        
        print('\nTEST STARTED FOR n = ' + str(self.n))
        precision = 10**-8
        if self.n == 1:
            P_sum = sum(self.estimate_prob('', w) for w in self.vocab)
            assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'
                 
        elif self.n == 2:
            histories = ['the', 'in', 'at', 'blue', 'white']
            for h in histories:
                P_sum = sum(self.estimate_prob(h, w) for w in self.vocab)
                assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h
                     
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
            assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'

        elif self.n == 2:
            histories = ['the', 'in', 'at', 'blue', 'white']
            for h in histories:
                P_sum = sum(self.estimate_smoothed_prob(h, w) for w in self.vocab)
                assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history "{}"'.format(h)

        print('Test successful!')


if __name__ == '__main__':
    corpora = '../corpora/corpus.sent.en.train'
    with open(corpora, 'r', encoding='utf-8') as file:
        content = file.readlines()
    corpus = tokenize(' '.join(content))
    VOCAB = set(corpus)

    # unigram_COUNTS = Counter()
    bigram_COUNTS = Counter()
    for line in content:
        # unigram_COUNTS.update(word_ngrams(line, 1))
        bigram_COUNTS.update(word_ngrams(line, 2))

    # unigram_LM = ngram_LM(1, unigram_COUNTS, VOCAB)
    VOCAB.update(('<s>', '</s>'))
    bigram_LM = ngram_LM(2, bigram_COUNTS, VOCAB)


    # Yoda's phrases assessment
    print('\nYODA\'S PHRASES ASSESSMENT')
    with open('../corpora/lm_eval/yodish.sent', 'r', encoding='utf-8') as yoda_file:
        yoda_phrases = yoda_file.readlines()

    with open('../corpora/lm_eval/english.sent', 'r', encoding='utf-8') as eng_file:
        eng_phrases = eng_file.readlines()

    scores = []
    # Handling phrases consisting of several sentences
    for phrase_y, phrase_en in zip(yoda_phrases, eng_phrases):
        p_y = re.findall(r'\w[\w\s,]+', phrase_y)
        p_en = re.findall(r'\w[\w\s,]+', phrase_en)
        score_y = sum([bigram_LM.score_sentence(p) for p in p_y]) / len(p_y)
        score_en = sum([bigram_LM.score_sentence(p) for p in p_en]) / len(p_en)
        scores.append((phrase_y, phrase_en, score_y, score_en))

    print('\nSCORES OF THE PAIRS')
    for score in scores:
        print('\n{} | {}\n({} | {})'.format(score[0], score[1], score[2], score[3]))

    difference = [abs(score[2] - score[3]) for score in scores]

    min_dif = min(difference)
    max_dif = max(difference)
    phrase_min = (yoda_phrases[difference.index(min_dif)],
                  eng_phrases[difference.index(min_dif)])
    phrase_max = (yoda_phrases[difference.index(max_dif)],
                  eng_phrases[difference.index(max_dif)])
    print('\nMINIMUM DIFFERENCE:\n{}\n{}'.format(phrase_min, min_dif))
    print('\nMAXIMUM DIFFERENCE:\n{}\n{}'.format(phrase_max, max_dif))



