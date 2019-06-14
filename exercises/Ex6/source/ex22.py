from collections import defaultdict
from collections import Counter
import math 
import re
import numpy as np 

def tokenize(text):
    return re.findall('[a-zA-Z]+', text)

def n_grams(n, tokens):
    return Counter([tuple(tokens[i : i+n]) for i in range(0,len(tokens)-n+1)])

def get_feature_count(corpus, n):
    text = open(corpus,encoding="utf-8", errors = "replace").read().lower()
    tokens = tokenize(text)
    return Counter(n_grams(n,tokens))

def pmi(f, feature_count, c, P_c ):
    number_of_f = sum([feature_count[i][f] for i in feature_count])
    f_c = feature_count[c][f] if feature_count[c][f] > 0 else 1e-10
    return math.log(f_c/number_of_f/P_c[c],2)

def main():
    path_template = "../corpora/train/{}.en.train"
    labels = ["dickens", "doyle", "twain"]
    n = 2
    P_c = {}
    fc = defaultdict(Counter)
    master_counter = Counter()
    for idx,l in enumerate(labels):
        P_c[l] = 1/len(labels)
        fc[l] = get_feature_count(path_template.format(l),n)
        master_counter = master_counter + fc[l]
    
    sorted_features = master_counter.most_common()
    res = [[pmi(pair[0], fc, l, P_c) for l in labels] for pair in sorted_features if pair[1] >= 15]

    # get top 15 of each author 
    for c in range(0, len(labels)):
        top_idx = list(range(0, len(res)))
        top_idx = sorted(top_idx, key = lambda x : (-res[x][c], -fc[labels[c]][sorted_features[x][0]]))[:10]
        print("Top features of {}: ".format(labels[c]))
        for idx in top_idx:
            f = sorted_features[idx][0]
            print(("{}\t"*(len(labels)+2)).format(f,int(res[idx][c]*1000)/1000, *[fc[l][f] for l in labels]))


if __name__ == '__main__':
    main()