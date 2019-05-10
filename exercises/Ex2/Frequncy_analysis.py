import re
from collections import Counter
import plotly.plotly as py
import plotly.graph_objs as go
import json

class CorporaHandler:
    def __init__(self, text):
        self.text_tokens = self.tokenize(text)
        self.ngrams = None

    @staticmethod
    def tokenize(text):
        '''

        :param text: plain text of language corpus, string
        :return: list of tokens, list of strings
        '''

        return re.findall(r'\b\w+', re.sub(r'\d', '', text))

    @staticmethod
    def char_ngrams(token, m):
        '''

        :param token: word token; string
        :param m: maximal length of n-grams; integer
        :return: a list of all possible n-grams in the token from n = 1 up to n = m; list of strings
        '''
        ngrams = []
        for n in range(1, m+1):
            for s in range(len(token) - n + 1):
                ngrams.append(token[s:s+n])
        return ngrams

    def corpus_to_ngrams(self, m):
        '''
        Converts text corpus were into a set of n-grams

        :param m: maximal length of n-grams; integer
        :return: a list of all possible n-grams in the corpus from n = 1 up to n = m; list of strings
        '''
        corpus_ngrams = []
        for token in self.text_tokens:
            for ngram in self.char_ngrams(token, m):
                corpus_ngrams.append(ngram)
        self.ngrams = Counter([x.lower() for x in corpus_ngrams])
        return self.ngrams

    def ngrams_statistics(self, probabilities=False):
        '''
        Distributes n-grams to unigrams, bigrams, trigrams, fourgrams

        :param probabilities: if True changes frequences to probabilities
        :return: n-grams and their frequences or probabilities; four Counter objects
        '''
        unigrams = Counter()
        bigrams = Counter()
        trigrams = Counter()
        fourgrams = Counter()
        for ngram, frequency in self.ngrams.items():
            if len(ngram) == 1:
                unigrams[ngram] = frequency
            elif len(ngram) == 2:
                bigrams[ngram] = frequency
            elif len(ngram) == 3:
                trigrams[ngram] = frequency
            else:
                fourgrams[ngram] = frequency
        if probabilities:
            for count in (unigrams, bigrams, trigrams, fourgrams):
                all_values = sum(count.values())
                for key, value in count.items():
                    count[key] = value / all_values
        return unigrams, bigrams, trigrams, fourgrams


if __name__ == '__main__':
    languages = [
        ('.en', 'English'),
        ('.de', 'German')
    ]
    path = 'corpora'
    for i, lang in enumerate(languages, start=1):
        with open(path + '/corpus' + lang[0], 'r', encoding='UTF-8') as file:
            corpus = file.read()

        handler = CorporaHandler(corpus)
        handler.corpus_to_ngrams(4)
        statistics = handler.ngrams_statistics()

        print(lang[1])
        for k in statistics:
            print(k.most_common(15))

        trace = go.Table(
            header=dict(values=['Unigrams', 'Bigrams', 'Trigrams', 'Fourgrams']),
            cells=dict(values=[[a[0] for a in statistics[0].most_common(15)],
                               [a[0] for a in statistics[1].most_common(15)],
                               [a[0] for a in statistics[2].most_common(15)],
                               [a[0] for a in statistics[3].most_common(15)]]))

        data = [trace]
        py.plot(data, filename=lang[1]+' n-grams')

        # calculation distributions of n-grams
        probs = handler.ngrams_statistics(probabilities=True)
        for j in range(4):
            with open('statistic_{}_{}.json'.format(lang[1], j + 1), 'w', encoding='utf-8') as output_stat:
                json.dump(probs[j], output_stat)
