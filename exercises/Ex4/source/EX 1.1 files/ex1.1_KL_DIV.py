from collections import defaultdict
from itertools import chain
import math
import matplotlib.pyplot as plt
import time
import prettytable
from prettytable import ALL as ALL
corpora = 'E:/SU/Semester 1 Sum-19/SNLP/ASSIGNMENT/4/Exercise_04_corpora/exercise4_corpora/ipa_corpus/'
#unified phoneme set
def unified_set(phoneme_en,phoneme_es,phoneme_fr,phoneme_it):
    all_phonemes_set = set(chain(phoneme_it,phoneme_fr,phoneme_en,phoneme_es))
    all_phonemes_set.remove('\n')
    return all_phonemes_set

#Phoneme Dictionary (keys,values)
def phonemecount(phonemeset):
    d=defaultdict(int)
    for cname in phonemeset:
        if cname!='\n':
            d[cname]+=1

    return d

#Probability distribution of the phoneme dictionary of each corpus with Lidstone smoothing, where, alpha = 1
def prob_dist_with_smoothing(phoneme_count,alpha,set_length):
    d=defaultdict(lambda : (alpha)/((alpha*set_length)+sum(phoneme_count.values())))
    for k, v in phoneme_count.items():  # will become d.items() in py3k
        d[k]=(v+alpha)/((alpha*set_length)+sum(phoneme_count.values()))

    return d

#Test unseen phonemes probability
def test_unseen(dist,test_val):
    return (dist[test_val])

#Calculation of KL Divergence for each Corpus
def KL_divergence(sample_space,P,Q):
    sum_KL=0
    for c in sample_space:
        sum_KL+=P[c]*(math.log2((P[c]/Q[c])))

    return sum_KL

#Plotting each phoneme Distribution
def generate_plot(prob_dist,name):
    d = []
    val = []
    for k, v in prob_dist.items():
        d.append(k)
        val.append(v)
    plt.style.use('ggplot')

    x = d
    energy = val
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, energy, color='c')
    plt.xlabel("Phonemes")
    plt.ylabel("Relative Frequency")
    plt.title('Phoneme Distribution of '+name+' corpus')
    plt.xticks(x_pos, x)
    plt.savefig(name+ '.png')
    plt.show()

#KL divergence Table for all corpus
def Generate_table(KL_divergence_all):
    t = prettytable.PrettyTable(hrules=ALL)
    t.title = 'KL-divergence(in Bits) for all language pair'
    t.field_names = ['', 'English', 'Spanish', 'French', 'Italian']
    t.add_row(
        ["English", KL_divergence_all[0][0], KL_divergence_all[0][1], KL_divergence_all[0][2], KL_divergence_all[0][3]])
    t.add_row(
        ["Spanish", KL_divergence_all[1][0], KL_divergence_all[1][1], KL_divergence_all[1][2], KL_divergence_all[1][3]])
    t.add_row(
        ["French", KL_divergence_all[2][0], KL_divergence_all[2][1], KL_divergence_all[2][2], KL_divergence_all[2][3]])
    t.add_row(
        ["Italian", KL_divergence_all[3][0], KL_divergence_all[3][1], KL_divergence_all[3][2], KL_divergence_all[3][3]])
    print(t)

#Read Corpus
phoneme_en = open(corpora+'corpus.ipa.en', 'r', encoding='utf-8')
phoneme_en = phoneme_en.read()
phoneme_es = open(corpora+'corpus.ipa.es', 'r', encoding='utf-8')
phoneme_es = phoneme_es.read()
phoneme_fr = open(corpora+'corpus.ipa.fr', 'r', encoding='utf-8')
phoneme_fr = phoneme_fr.read()
phoneme_it = open(corpora+'corpus.ipa.it', 'r', encoding='utf-8')
phoneme_it=phoneme_it.read()

#Make Unified Set
unified_Ph_set = unified_set(phoneme_en,phoneme_es,phoneme_fr,phoneme_it)

#Make Default Dictionary(Key and Value) for Each Corpus
pc_en=phonemecount(phoneme_en)
pc_es=phonemecount(phoneme_es)
pc_fr=phonemecount(phoneme_fr)
pc_it=phonemecount(phoneme_it)
alpha=1.0

#Calculate Probability Distribution with Smoothing
prob_dist_en=prob_dist_with_smoothing(pc_en,alpha,len(pc_en))
prob_dist_es=prob_dist_with_smoothing(pc_es,alpha,len(pc_es))
prob_dist_fr=prob_dist_with_smoothing(pc_fr,alpha,len(pc_fr))
prob_dist_it=prob_dist_with_smoothing(pc_it,alpha,len(pc_it))

#Test Unseen Phoneme
print('Testing for 2 Unseen Phonemes in different corpus\n')
print('Probability of phoneme '+'"$" in English corpus is: '+str(test_unseen(prob_dist_en,'$')))
print('Probability of phoneme '+'"$" in French corpus is: '+str(test_unseen(prob_dist_fr,'$')))
print('Probability of phoneme '+'"1" in Italian corpus is: '+str(test_unseen(prob_dist_it,'1')))
time.sleep(1)

#Generate Bar Chart of Probability Distribution for Each Corpus
generate_plot(prob_dist_en,'English')
generate_plot(prob_dist_es,'Spanish')
generate_plot(prob_dist_fr,'French')
generate_plot(prob_dist_it,'Italian')

#Calculate KL Divergence for All Language Pair
KL_divergence_all=[
               [KL_divergence(unified_Ph_set,prob_dist_en,prob_dist_en),
                  KL_divergence(unified_Ph_set,prob_dist_en,prob_dist_es),
                  KL_divergence(unified_Ph_set,prob_dist_en,prob_dist_fr),
                  KL_divergence(unified_Ph_set,prob_dist_en,prob_dist_it)],
               [KL_divergence(unified_Ph_set, prob_dist_es, prob_dist_en),
                  KL_divergence(unified_Ph_set, prob_dist_es, prob_dist_es),
                  KL_divergence(unified_Ph_set, prob_dist_es, prob_dist_fr),
                  KL_divergence(unified_Ph_set, prob_dist_es, prob_dist_it)],
               [KL_divergence(unified_Ph_set,prob_dist_fr,prob_dist_en),
                  KL_divergence(unified_Ph_set,prob_dist_fr,prob_dist_es),
                  KL_divergence(unified_Ph_set,prob_dist_fr,prob_dist_fr),
                  KL_divergence(unified_Ph_set,prob_dist_fr,prob_dist_it)],
               [KL_divergence(unified_Ph_set,prob_dist_it,prob_dist_en),
                  KL_divergence(unified_Ph_set,prob_dist_it,prob_dist_es),
                  KL_divergence(unified_Ph_set,prob_dist_it,prob_dist_fr),
                  KL_divergence(unified_Ph_set,prob_dist_it,prob_dist_it)]
               ]
print('\n******************KL Divergence******************')
print('Kl Divergence of Enlish')
print(KL_divergence_all[0][:])
print('Kl Divergence of Spanish')
print(KL_divergence_all[1][:])
print('Kl Divergence of French')
print(KL_divergence_all[2][:])
print('Kl Divergence of Italian')
print(KL_divergence_all[3][:])
print('Generating 4*4 table for comparison...')
time.sleep(3)
#Show Table of Comparison
Generate_table(KL_divergence_all)


