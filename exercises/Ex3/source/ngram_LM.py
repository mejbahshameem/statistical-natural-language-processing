# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT III 

import math
import re
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
        for i in range (len(sent)):
            ngrams.append((sent[i],))
    else:
        for i in range(n,len(sent)+1):
            ngrams.append((sent[n-1],sent[n]))

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

    def estimate_prob(self, history, word):
        if self.n == 1:
            return self.ngram_counts[(word,)]/sum(self.ngram_counts[(word,)] for word in self.vocab)

        if self.n > 1:
            return self.ngram_counts[(history,word)]/sum(self.ngram_counts[(history,word)] for word in self.vocab)

    
    def estimate_smoothed_prob(self, history, word, alpha = 0.5):
        """Estimate probability of a word given a history with Lidstone smoothing."""
        
       
    # YOUR CODE HERE
            

    def logP(self, history, word):
        """Return base-2 log probablity."""

    # YOUR CODE HERE
                 


    def score_sentence(self, sentence):
        """Given a sentence, return score."""
        
        # YOUR CODE HERE


 
    def test_LM(self):
        """Test whether or not the probability mass sums up to one."""
        
        print('TEST STARTED FOR n = '+str(n))

        precision = 10**-8
                 
        if self.n == 1:
                 
            P_sum = sum(self.estimate_prob('', w) for w in self.vocab)
            
            assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'
                 
        elif self.n == 2:
            histories = ['the', 'in', 'at', 'blue', 'white']
                 
            for h in histories:
                 
                P_sum = sum(self.estimate_prob(h, w) for w in self.vocab)
                
                assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h
                     
        print('TEST SUCCESSFUL!')



    def test_smoohted_LM(self):
        self.n = self.n
    #Test whether or not the smoothed probability mass sums up to one.
    # YOUR CODE HERE


# ADD YOUR CODE TO COLLECT COUTNS AND CONSTRUCT VOCAB
corpora = '../corpora/corpus.sent.en.train'
n = 1

# ONCE YOU HAVE N-GRAN COUNTS AND VOCAB, 
# YOU CAN BUILD LM OBJECTS AS ...
file = open(corpora, 'r')
COUNTS = Counter()
for line in file:
    COUNTS.update(word_ngrams(line,n))
file.seek(0, 0)
VOCAB = tokenize(file.read())
VOCAB = set(VOCAB)
if n > 1 :
    VOCAB.update(('<s>','</s>'))
file.close() 
unigram_LM = ngram_LM(n, COUNTS, VOCAB)


# THEN TEST YOUR IMPLEMENTATION AS ..
unigram_LM.test_LM() 