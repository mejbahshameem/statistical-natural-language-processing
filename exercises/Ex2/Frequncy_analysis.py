import re
from collections import Counter


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
        ngrams = []
        for n in range(1, m+1):
            for s in range(len(token) - n + 1):
                ngrams.append(token[s:s+n])

        return ngrams

    def corpus_to_ngrams(self, m):
        corpus_ngrams = []
        for token in self.text_tokens:
            for ngram in self.char_ngrams(token, m):
                corpus_ngrams.append(ngram)
        self.ngrams = Counter([x.lower() for x in corpus_ngrams])
        return self.ngrams

    def ngrams_statistics(self):
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

        return unigrams, bigrams, trigrams, fourgrams

if __name__ == '__main__':
    languages = [
        ('.en', 'English'),
        ('.de', 'German')
    ]
    path = './corpora'
    for i, lang in enumerate(languages, start=1):
        with open(path + '/corpus' + lang[0], 'r', encoding='UTF-8') as file:
            corpus = file.read()

        handler = CorporaHandler(corpus)
        handler.corpus_to_ngrams(4)
        statistics = handler.ngrams_statistics()

        print(lang[1])
        for k in statistics:
            print(k.most_common(15))