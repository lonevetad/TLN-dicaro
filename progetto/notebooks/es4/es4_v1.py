import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

from notebooks.utilities.cacheVarie import CacheSynsets
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
    # return utils.CacheSynsets()
    return CacheSynsets()


cache = newCache()


# ------------
# ------------
# ------------------ func ------------------
# ------------
# ------------


def get_preprocessed_words(text):
    return preprocessing(text)

    # return preprocessing(text)


def doc_tiling_v1(similarity_subsequent_sentences, windows_count, max_iterations=0):
    """
        :param similarity_subsequent_sentences: the result of
        the function "compute_similarity_lists_subsequent_sentences"
        :param windows_count: the amount of paragraph to find. It's greater by 1 than the length of
        the returned list
        :param max_iterations: the inner algorithm improves iteratively the tiling; this parameter
        sets an upper bound of iterations
        :return: a list of breakpoints: indexes (between one sentence and the next) where one paragraph ends and
        the next starts. The length is lower by 1 than the parameter "windows_count", since those indexes identifies
        a boundary within two different paragraphs. The indexes are to be considered as "inclusive"
        """
    #:param list_of_sentences: the parameter of the function "document_segmentation"
    if max_iterations < 1:
        max_iterations = 10
    len_sss = len(similarity_subsequent_sentences)
    # initial_window_size = float(windows_count) / float(len_sss)
    # iniziamo con finestre equamente distribuite
    breakpoints_amount = windows_count - 1
    # breakpoint_indexes = [int(initial_window_size * (1+i)) for i in range(0, breakpoints_amount)]
    breakpoint_indexes = None
    '''
        if self.algorithm_tiling == 0:  # K-Means
            # breakpoint_indexes = [int(initial_window_size * (1+i)) for i in range(0, breakpoints_amount)]
            breakpoint_indexes = [((windows_count * (1 + i)) / len_sss) for i in range(0, breakpoints_amount)]
            means = [0.0] * windows_count
            breakpoints_amount = len(breakpoint_indexes)

            def recalculate_mean():
                # calcola le medie
                i = 0
                while i <= breakpoints_amount:  # see "limit"
                    start = breakpoint_indexes[i]
                    # the last step is the length of the whole array
                    limit = len_sss if (i == breakpoints_amount) else breakpoint_indexes[i + 1]
                    paragraph_length = (limit - start) + 1
                    summ = 0.0
                    while start < limit:
                        summ += similarity_subsequent_sentences[start]
                        start += 1
                    means[i] = summ

            recalculate_mean()
        '''

    '''
        silliest implementation:
        the scores varies a bit, but some sentences varies more.
        Let's compute the mean [and variance??] first. The K paragraphs
        (indexes where the last sentence of a p. meets the first
        of the new p.) are then the k farthest indexes which 
        have lower value than the mean
        '''
    mean = sum(similarity_subsequent_sentences) / float(len_sss)
    deltas_from_mean = sorted(
        [((similarity_subsequent_sentences[i] - mean), i) for i in range(0, len_sss)]
        , key=lambda t: t[0])
    breakpoint_indexes = [0] * breakpoints_amount
    i = 0
    while i < breakpoints_amount:
        breakpoint_indexes[i] = deltas_from_mean[i][1]
        i += 1
    return breakpoint_indexes


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
