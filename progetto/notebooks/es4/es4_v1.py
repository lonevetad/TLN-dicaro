import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

from notebooks.utilities.cacheVarie import CacheSynsetsBag
from notebooks.utilities.functions import preprocessing, SynsetToBagOptions

'''
import sys
#sys.path.append(".")
#sys.path.append(".\\..\\..\\utilities")

sys.path.append(".\\..\\utilities")
print("what the file:")
print(__file__)
print("let's go ....")
'''

'''
sys.path.append(".\\..\\aaa")
import bbb
bbb.ccc()

if __name__ == '__main__':
	print("main")
	bbb.ccc()
	print("FINE main")
'''

# sys.path.append(".\\..\\")
# from utilities import *
# import utilities
# from utilities import *
# from utilities import cacheVarie
# from ..utilities import *

# import cacheVarie

# from ..utilities import cacheVarie

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))

print("\n\nlet's do it")


# cacheVarie.main()


def newCache():
    # return utils.CacheSynsetsBag()
    return CacheSynsetsBag()


cache = newCache()


# ------------
# ------------
# ------------------ func ------------------
# ------------
# ------------


def get_preprocessed_words(text):
    return preprocessing(text)


# return preprocessing(text)


def document_segmentation(list_of_sentences):
    if not (isinstance(list_of_sentences, list)):
        return None
    words_each_sentences = [get_preprocessed_words(sentence) for sentence in list_of_sentences]
    words_counts_and_bags = {}  # set()
    option = SynsetToBagOptions(bag=set())
    # option = SynsetToBagOptions(bag=set())
    for words in words_each_sentences:
        for w in words:
            # all_words.add(w)
            if w in words_counts_and_bags:
                words_counts_and_bags[w][0] += 1
            else:
                words_counts_and_bags[w] = [1, (
                w, cache.get_extended_bag_for_words(option=option))]  # bag_of_word_expanded

    # TODO: ora usare le frasi e questi bags per computarne le similarità ed eseguire il text tiling
    return words_counts_and_bags


sentences = [
    "I love to pet my cat while reading fantasy books.",
    "Reading books makes my fantasy fly over everyday problems.",
    "One problem of those is my cat vomiting on my pants."
]

wcabs = document_segmentation(sentences)
for word, cab in wcabs.items:
    print(word, " -> (", cab[0], ";;;", cab[1])

'''

'''
