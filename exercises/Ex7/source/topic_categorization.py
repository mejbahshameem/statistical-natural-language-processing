import re
import math
import csv
from collections import Counter
from prettytable import PrettyTable, ALL


CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


def build_counters(corpus_file):
    counter_world = Counter()
    counter_sports = Counter()
    counter_business = Counter()
    counter_scitech = Counter()
    counters = {1: counter_world, 2: counter_sports, 3: counter_business, 4: counter_scitech}

    with open(corpus_file, newline='') as file:
        for row in csv.reader(file):
            cat, title, text = row
            counters[int(cat)].update(tokenize(text + title))
        c = counter_world + counter_sports + counter_business + counter_scitech
        for word, count in c.items():
            if count < 3:
                for counter in counters.values():
                    counter.pop(word, None)
        c = counter_world + counter_sports + counter_business + counter_scitech

    return c, counter_world, counter_sports, counter_business, counter_scitech


def pmi(f, c, counters):
    '''counters = {'Total': counter_total, 'World': counter_world, 'Sports': counter_sports,
                'Business': counter_business, 'Sci/Tech': counter_scitech}'''

    num_f_c = counters[c][f] if counters[c][f] > 0 else 1e-10
    num_f_total = counters['Total'][f]
    P_c = 1 / len(CATEGORIES)
    return math.log(num_f_c / num_f_total / P_c, 2)


def pmi_avg(t, counters):
    pmi_values = [pmi(t, c, counters) for c in CATEGORIES]
    return sum(pmi_values)


def pmi_max(t, counters):
    pmi_values = [(pmi(t, c, counters), c) for c in CATEGORIES]
    return sorted(pmi_values, key=lambda x: -x[0])[0]


if __name__ == "__main__":
    folder = '../corpora/ag_news_csv/'
    counter_total, counter_world, counter_sports, counter_business, counter_scitech = \
        build_counters(folder + 'train.csv')

    counters = {'Total': counter_total, 'World': counter_world, 'Sports': counter_sports,
                'Business': counter_business, 'Sci/Tech': counter_scitech}

    pmi_avg_val = [(word[0], pmi_avg(word[0], counters),
                    [count[word[0]] for count in counters.values()][1:])
                   for word in counter_total.most_common()]
    pmi_max_val = [(word[0], pmi_max(word[0], counters),
                    [count[word[0]] for count in counters.values()][1:])
                   for word in counter_total.most_common()]
    top_pmi_avg = sorted(pmi_avg_val, key=lambda x: (-x[1], -sum(x[2])))[:20]
    top_pmi_max = sorted(pmi_max_val, key=lambda x: (-x[1][0], -sum(x[2])))[:20]

    print(top_pmi_avg)
    print(top_pmi_max)

    print('Top 20 words by expected PMI')
    t = PrettyTable(hrules=ALL)
    t.title = 'Top 20 words by expected PMI'
    column_names = ['#', 'Word', 'Expected PMI', 'Counts by categories']
    t.add_column(column_names[0], range(1, 21))
    t.add_column(column_names[1], [e[0] for e in top_pmi_avg])
    t.add_column(column_names[2], [e[1] for e in top_pmi_avg])
    t.add_column(column_names[3], [e[2] for e in top_pmi_avg])
    print(t)

    print('Top 20 words by maximum PMI')
    t = PrettyTable(hrules=ALL)
    t.title = 'Top 20 words by maximum PMI'
    column_names = ['#', 'Word', 'Maximum PMI', 'Category', 'Counts by categories']
    t.add_column(column_names[0], range(1, 21))
    t.add_column(column_names[1], [e[0] for e in top_pmi_max])
    t.add_column(column_names[2], [e[1][0] for e in top_pmi_max])
    t.add_column(column_names[3], [e[1][1] for e in top_pmi_max])
    t.add_column(column_names[4], [e[2] for e in top_pmi_max])
    print(t)
