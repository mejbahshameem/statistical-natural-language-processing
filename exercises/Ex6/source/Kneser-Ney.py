import math
import re
import time
import math
from collections import defaultdict, Counter, OrderedDict


def tokenize(text):
    return re.findall('[a-z]+', text.lower())


def word_ngrams(sent, n):
    sent = tokenize(sent)
    if n > 1:
        sent.insert(0, '<s>')
        sent.append('</s>')
    ngrams = []
    if n == 1:
        for i in range(len(sent)):
            ngrams.append((sent[i],))
    else:
        for i in range(n - 1, len(sent)):
            ngrams.append((sent[i - 1], sent[i]))
    return ngrams


def lids_smoothing(w, unigram, alpha):
    d = (w + alpha) / ((alpha * len(unigram)) + sum(unigram.values()))
    return d


corpora = 'E:/SU/Semester 1 Sum-19/SNLP/ASSIGNMENT/6/Exercise_03_corpora/corpus.sent.en.train'
file = open(corpora, 'r', encoding='utf-8')
V = tokenize(file.read())
count_york = V.count('york')
count_matter = V.count('matter')
alpha = 1
bigram_COUNTS = Counter()
unigram_COUNTS = Counter()
file.seek(0, 0)
for line in file:
    unigram_COUNTS.update(word_ngrams(line, 1))
    bigram_COUNTS.update(word_ngrams(line, 2))

n1_count_york = 0
n1_count_matter = 0
for key, val in bigram_COUNTS.items():
    if key[1] == 'york':
        n1_count_york += 1
    if key[1] == 'matter':
        n1_count_matter += 1

pml_york = -1 * math.log2(count_york / sum(unigram_COUNTS.values()))
pml_matter = -1 * math.log2(count_matter / sum(unigram_COUNTS.values()))
plids_york = -1 * math.log2(lids_smoothing(count_york, unigram_COUNTS, alpha))
plids_matter = -1 * math.log2(lids_smoothing(count_matter, unigram_COUNTS, alpha))
pkn_york = -1 * math.log2(n1_count_york / len(bigram_COUNTS))
pkn_matter = -1 * math.log2(n1_count_matter / len(bigram_COUNTS))
print("N(w) 'york' = ", count_york, "'matter' = ", count_matter)
print("N1+ ( . w) 'york' = ", n1_count_york, "'matter' = ", n1_count_matter)
print("-log2 PML (w) 'york' = ", pml_york, "'matter' = ", pml_matter)
print("-log2 Plids (w) 'york' = ", plids_york, "'matter' = ", plids_matter)
print("-log2 PKL (w) 'york' = ", pkn_york, "'matter' = ", pkn_matter)
