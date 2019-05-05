import Frequncy_analysis as fa
from collections import defaultdict
import string

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
	lan = 'de' #or .en
	path = './corpora/' #path to the corpora folder

	h = ''
	S = list(string.ascii_lowercase)
	if lan == 'de':
		S = S + ['ä','ö','ü','ß']

	f = open(path+'corpus.'+lan,'r',encoding='UTF-8')
	corpus = f.read()
	handler = fa.CorporaHandler(corpus)
	handler.corpus_to_ngrams(len(h)+1)
	prob_dist = prob_dist(handler.ngrams,S,h)
	print(round(sum(prob_dist.values()),7))
	assert round(sum(prob_dist.values()),7)==2 #should throw error but nothing happens