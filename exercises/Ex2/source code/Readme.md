####Top list of n-grams

You can launch the script to reproduce the list of n-grams directly from *frequency_analysis.py*

You need to have corpora folder in the same directory with the *frequency_analysis.py*

#####Get probabilities

In order to obtain probabilities of n-grams instead of frequencies set *probabilities=True* in *CorporaHandler.ngrams_statistics()*

####Cracking ciphertext

You can launch the script directly from *cryptoanalysis.py*

Firstly, you need to produce bigrams and trigrams distributions with *frequency_analysis.py* 

Then you need to have distributions in the same directory with the *cryptoanalysis.py*

####Probability Distribution

Set language and History. Entropy gets printed and figure of distribution gets generated.