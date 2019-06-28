from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),
                                    token_pattern=r'\D', min_df=1)

trigram_vectorizer = CountVectorizer(ngram_range=(3, 3),
                                     token_pattern=r'\D', min_df=1)

bigram_analyze = bigram_vectorizer.build_analyzer()
trigram_analyze = trigram_vectorizer.build_analyzer()

bilinear_svc = svm.SVC(kernel='linear', C=1)
birbf_svc = svm.SVC(kernel='rbf', C=1, gamma=0.01)
trilinear_svc = svm.SVC(kernel='linear', C=1)
trirbf_svc = svm.SVC(kernel='rbf', C=1, gamma=0.01)

f = open('../exercise9_corpora/TRAIN-ES.txt')
corpus = []
classes = []
for line in f:
    corpus.append(line[:-6])
    classes.append(line[-6:-1])
f.close()
Xbi = bigram_vectorizer.fit_transform(corpus).toarray()
Xtri = trigram_vectorizer.fit_transform(corpus).toarray()
bilinear_svc.fit(Xbi, classes)
birbf_svc.fit(Xbi, classes)
trilinear_svc.fit(Xtri, classes)
trirbf_svc.fit(Xtri, classes)

f = open('../exercise9_corpora/TEST-ES.txt')
testcorpus = []
testclasses = []
for line in f:
    testcorpus.append(line[:-6])
    testclasses.append(line[-6:-1])
f.close()
Xbitest = bigram_vectorizer.transform(testcorpus).toarray()
Xtritest = trigram_vectorizer.transform(testcorpus).toarray()
print('Bigrams with linear kernel have an accuracy of ' +
      str(accuracy_score(testclasses, bilinear_svc.predict(Xbitest)) * 100))
print('Bigrams with rbf kernel have an accuracy of ' +
      str(accuracy_score(testclasses, birbf_svc.predict(Xbitest)) * 100))
print('Trigrams with linear kernel have an accuracy of ' +
      str(accuracy_score(testclasses, trilinear_svc.predict(Xtritest)) * 100))
print('Trigrams with rbf kernel have an accuracy of ' +
      str(accuracy_score(testclasses, trirbf_svc.predict(Xtritest)) * 100))
