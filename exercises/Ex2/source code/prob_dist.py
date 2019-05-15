import Frequncy_analysis as fa
from collections import defaultdict
import string
import math
import matplotlib.pyplot as plt

def entropy(prop_dist,S):
	E = 0
	for s in S:
		if(prob_dist[s]!=0):
			E -= prop_dist[s] * math.log(prop_dist[s])

	return E


def prob_dist(N,S,h):
	prob = defaultdict(float)
	sumNS = 0
	for c in S:
		sumNS += N[h+c]  #sums up N(c1,...,ck,s) for all s in S 

	if sumNS == 0:
		prob = prob_dist(N,S,'')
		return prob                 #returns prob_dist for empty historie if with current history no s in S was found in ngrams

	for c in S:
		prob[c] = N[h+c]/sumNS      #calculates the prob for c

	return prob

if __name__ == '__main__':
	lan = 'de' # de or en
	path = './corpora/' #path to the corpora folder

	h = 'a'
	S = list(string.ascii_lowercase)
	if lan == 'de':
		S = S + ['ä','ö','ü','ß']

	f = open(path+'corpus.'+lan,'r',encoding='UTF-8')
	corpus = f.read()
	handler = fa.CorporaHandler(corpus)
	handler.corpus_to_ngrams(len(h)+1)
	prob_dist = prob_dist(handler.ngrams,S,h)
	assert round(sum(prob_dist.values()),7)==1 

	E = entropy(prob_dist, S)
	print(E)

if False:    #turn to true if you want plot the distribution
	plt.figure(num=1, figsize=(1,len(S)), dpi=None, facecolor=None, edgecolor=None, frameon=True, clear=False)
	plt.plot(scalex=True, scaley=True, data=None)
	plt.bar(prob_dist.keys(),prob_dist.values())
	plt.show()
