import math
import re
import random
import json
from tqdm import tqdm


def swap_key(key, a, b):
    '''

    :param key: substitution key; string
    :param a: letter; string
    :param b: letter; string
    :return: substitution key with swapped letters; string
    '''
    return key.translate(str.maketrans(a+b, b+a))


def decrypt(msg, key):
    '''

    :param msg: ciphertext without spaces and punctuation; string
    :param key: substitution key; string
    :return: deciphered text; string
    '''
    return msg.translate(str.maketrans(key, alphabet))


def steepest_ascent(msg, key, decryption_fitness, num_steps):
    '''
    Optimization

    :param msg: ciphertext without spaces and punctuation; string
    :param key: substitution key; string
    :param decryption_fitness: fitness function
    :param num_steps: number of steps in the restart; integer
    :return: the best variant of decryption and its key; (string, string)
    '''
    decryption = decrypt(msg, key)
    value = decryption_fitness(decryption)
    neighbors = iter(neighboring_keys(key, decryption))

    for step in range(num_steps):
        next_key = next(neighbors)
        next_decryption = decrypt(msg, next_key)
        next_value = decryption_fitness(next_decryption)

        if next_value > value:
            key, decryption, value = next_key, next_decryption, next_value
            neighbors = iter(neighboring_keys(key, decryption))

    return decryption, key


# calculates n-grams of input string
def char_ngrams(msg, n):
    '''

    :param msg: text without spaces and punctuation; string
    :param n: parameter of n-grams; integer
    :return: a list of n-grams; list of strings
    '''
    return [msg[i:i+n] for i in range(len(msg) - (n-1))]


# return sum of trigram log probabilities
def trigram_string_prob(msg):
    '''
    Fitness function

    :param msg: text without spaces and punctuation; string
    :return: sum of probabilities of trigrams
    '''
    return sum(math.log10(trigram_char_prob.get(trigram, 10e-7)) for trigram in char_ngrams(msg, 3))


def neighboring_keys(key, decrypted_msg):
    '''

    :param key: key; string
    :param decrypted_msg: deciphered text; string
    :return: yields keys similar to input key
    '''
    bigrams = sorted(char_ngrams(decrypted_msg, 2),
                     key=lambda x: bigram_char_prob.get(x, 0.))[:30]

    for c1, c2 in bigrams:
        for a in shuffled(alphabet):
            if c1 == c2 and bigram_char_prob.get(a + a, 0) > bigram_char_prob.get(c1 + c2, 0):
                yield swap_key(key, a, c1)
            else:
                if bigram_char_prob.get(a + c2, 0) > bigram_char_prob.get(c1 + c2, 0):
                    yield swap_key(key, a, c1)
                if bigram_char_prob.get(c1 + a, 0) > bigram_char_prob.get(c1 + c2, 0):
                    yield swap_key(key, a, c2)
    while True:
        yield swap_key(key, random.choice(alphabet),
                      random.choice(alphabet))


def shuffled(s):
    '''

    :param s: string
    :return: shuffled string
    '''
    s_list = list(s)
    random.shuffle(s_list)
    return ''.join(s_list)


def preprocess_ciphertext(text):
    '''

    :param chars: ciphertext; string
    :return: lowercased ciphertext without spaces and punctuation; string
    '''
    return ''.join(re.findall('[a-z]+', text.lower()))


def crack_ciphertext(msg, num_steps=5000, restarts=30):
    '''

    :param msg: ciphertext; string
    :param num_steps: number of steps in the restart; integer
    :param restarts: number of restarts generating random key; integer
    :return: decryption and subbstitution key; (string, string)
    '''
    msg = preprocess_ciphertext(msg)
    startingKeys = [shuffled(alphabet) for i in range(restarts)]
    local_maxes = [steepest_ascent(msg, key, trigram_string_prob, num_steps)
                  for key in tqdm(startingKeys)]

    fitness = [trigram_string_prob(decryption_key[0]) for decryption_key in local_maxes]
    decryption, key = local_maxes[fitness.index(max(fitness))]
    return decryption, key, max(fitness)


if __name__ == "__main__":
    msg = """
    KOTP OX PSG AIOQVGKT OX PSG WOIVC TPGK XIOK VZBYEZTPZJ
    KZTPNHGT NBC TZKAVG KZTEBCGITPNBCZBYT. COB’P GFGI PNHG WOICT
    NP XNJG FNVEG. WSGB UOE TPGA ZBPO PSG ROBG OX VOFG, VNBYENYG
    NT WG HBOW ZP QGJOKGT OQTOVGPG. PSNP WSZJS JNBBOP QG AEP ZBPO
    WOICT JNB OBVU QG YINTAGC PSIOEYS TZVGBJG."""

    msg_decription = []

    for language in ("English", "German"):
        if language == "English":
            alphabet = "abcdefghijklmnopqrstuvwxyz"
        else:
            alphabet = "abcdefghijklmnopqrstuvwxyzäöüß"

        with open('statistic_{}_3.json'.format(language), encoding='utf-8') as trig_file:
            trigram_char_prob = json.load(trig_file)
        with open('statistic_{}_2.json'.format(language), encoding='utf-8') as bi_file:
            bigram_char_prob = json.load(bi_file)

        msg_decription.append(crack_ciphertext(msg, num_steps=20000, restarts=100))

    print(msg_decription)

    if msg_decription[0][2] > msg_decription[1][2]:
        print('Message: {}'.format(msg_decription[0][0]))
        print(msg_decription[0][1])
    else:
        print('Message: {}'.format(msg_decription[1][0]))
        print('Key: '.format(msg_decription[1][1]))
