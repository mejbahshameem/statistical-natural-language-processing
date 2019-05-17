# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT III 

import math
import re
import time
from collections import defaultdict, Counter
#from nltk import ngrams


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


def word_ngrams(sent, n):
    """Givne a sent as str return n-grams as a list of tuple"""
    
    # EXAMPLES 
    # > word_ngrams('hello world', 1)
    # [('hello',), ('world',)]
    # > word_ngrams('hello world', 2)
    # [('<s>', 'hello'), ('hello', 'world'), ('world', '</s>')]

    # YOUR CODE HERE
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

        # YOUR CODE HERE
        # START BY MAKING THE RIGHT COUNTS FOR THIS PARTICULAR self.n
   
        self.ngram_counts = ngram_counts
        if n > 1:
            self.unigram_counts = Counter()
            for word in self.ngram_counts.items():
                self.unigram_counts.update({(word[0][0],):word[1]})

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



# ONCE YOU HAVE N-GRAN COUNTS AND VOCAB, 
# YOU CAN BUILD LM OBJECTS AS ...




if __name__ == '__main__':
    # ADD YOUR CODE TO COLLECT COUTNS AND CONSTRUCT VOCAB
    corpora = '../corpora/corpus.sent.en.train'

    # ONCE YOU HAVE N-GRAN COUNTS AND VOCAB,
    # YOU CAN BUILD LM OBJECTS AS ...
    unigram_COUNTS = Counter()
    bigram_COUNTS = Counter()
    file = open(corpora, 'r', encoding='utf-8')
    for line in file:
        unigram_COUNTS.update(word_ngrams(line, 1))
        bigram_COUNTS.update(word_ngrams(line, 2))
    file.seek(0, 0)
    VOCAB = tokenize(file.read())
    length = len(VOCAB)
    VOCAB = set(VOCAB)
    file.close()

    unigram_LM = ngram_LM(1, unigram_COUNTS, VOCAB)
    VOCAB.update(('<s>', '</s>'))
    bigram_LM = ngram_LM(2, bigram_COUNTS, VOCAB)

    # THEN TEST YOUR IMPLEMENTATION AS ..
    unigram_LM.test_LM()
    bigram_LM.test_LM()
    unigram_LM.test_smoohted_LM()
    bigram_LM.test_smoohted_LM()

    #15 most common words
    print(unigram_COUNTS.most_common(15))
    print(bigram_COUNTS.most_common(15))

    #TTR
    print(float(len(VOCAB))/float(length))

    # Translations assessment
    print('\nTRANSLATION ASSESSMENT')
    print('Sentence: Gestern war ich zu Hause.')
    hypotheses = ['Yesterday was I at home.',
                  'Yesterday I was at home.',
                  'I was at home yesterday.']

    scores = [bigram_LM.score_sentence(hyp) for hyp in hypotheses]
    translation = hypotheses[scores.index(min(scores))]
    print('The most fluent translation: {}'.format(translation))


