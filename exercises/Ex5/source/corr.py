# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT V
import re
from matplotlib import pyplot as plt
import numpy as np

plot = True


def tokenize(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


def correlation(w1, w2, D):
    # python 2 runs without ignoring errors. Python3 would throw error:
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf6 in position
    # 606437: invalid start byte
    file = open('../continuous.corpus.en', 'r',
                encoding='utf-8', errors='ignore')

    f = file.read()
    file.close()
    f = tokenize(f)
    ND = 0.
    Nw1 = 0.
    Nw2 = 0.
    for i in range(len(f) - D):
        if f[i] == w1:
            Nw1 += 1
        if f[i] == w2:
            Nw2 += 1
        if f[i] == w1 and f[i + D] == w2:
            ND += 1

    PD = ND / (len(f) - D)
    Pw1 = Nw1 / len(f)
    Pw2 = Nw2 / len(f)
    return PD / (Pw1 * Pw2)


def plot_moving_average(list, name, label, log=False):
    n = 5  # group size
    m = 4  # overlap size
    list = [sum(list[i:i + n]) / n for i in range(0, len(list) - n, n - m)]
    plt.figure(name)
    plt.title('correlation ' + name, fontdict={'fontsize': 10})
    plt.ylabel('correlation')
    if log:
        plt.semilogx(range(1, len(list) + 1), list, label=label)
        plt.legend()
        plt.xlabel('Moving average with window 5 of D log scale')
        plt.savefig('../figs/' + name + '.png')
    else:
        plt.plot(range(1, len(list) + 1), list, label=label)
        plt.xlabel('Moving average with window 5 of D')
        plt.legend()
        plt.savefig('../figs/' + name + '.png')


for pair in (('you', 'your'), ('he', 'his'), ('she', 'her'), ('they', 'their'), ('he', 'her'), ('she', 'his')):
    print('correlation for ' + str(pair))
    list = [correlation(pair[0], pair[1], i) for i in range(1, 101)]
    print(list)
    if plot:
        plot_moving_average(list, 'correlation', str(pair), log=False)
        plot_moving_average(list, 'correlation' + '_log', str(pair), log=True)

plot_moving_average([1 for i in range(100)],
                    'correlation', 'independend words', log=False)
plot_moving_average(
    [1 for i in range(100)], 'correlation' + '_log', 'independend words', log=True)
