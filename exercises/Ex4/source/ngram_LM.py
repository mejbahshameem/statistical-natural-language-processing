# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT III 

import math
import re
import time
from collections import defaultdict, Counter, OrderedDict
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

    def perplexity(self,T,alpha):
        res = 0
        M=0
        for sent in T:
            M += len(sent)
            for word in sent:
                res += self.logP(word[0],word[-1+self.n],alpha=alpha)
        return 1/2**(res/M)

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

    def logP(self, history, word,alpha=0.5):
        """Return base-2 log probablity."""

        prob = self.estimate_smoothed_prob(history, word,alpha=alpha)
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
 
    def prob_dist(self,h):
        prob = dict()
        for word in self.vocab:
            prob[word] = self.estimate_prob(h, word)
        return sorted(prob.items(),reverse=True, key = 
             lambda kv:(kv[1], kv[0]))

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
    corpora = '../../Ex3/corpora/corpus.sent.en.train'
    test_copora = '../exercise4_corpora/lm_eval/'

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

    for filename in ['simple.test','wiki.test']:
        bigram_corp = []
        unigram_corp = []
        file = open(test_copora+filename,'r',encoding='utf-8')
        for line in file:
            unigram_corp.append(word_ngrams(line, 1))
            bigram_corp.append(word_ngrams(line, 2))
        
        print('\nunigram perplexities for '+filename)
        print('with smoothed probabilities:')
        print(unigram_LM.perplexity(unigram_corp,0.2))

        print('\nbigram perplexities for '+filename)
        print('with smoothed probabilities:')
        print(bigram_LM.perplexity(bigram_corp,0.2))
