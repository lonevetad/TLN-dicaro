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


def newCache():
    return CacheSynsetsBag()


# ------------
# ------------
# ------------------ func ------------------
# ------------
# ------------


def get_preprocessed_words(text):
    return preprocessing(text)


# return preprocessing(text)


class WeightsSynsetsInfo(object):
    def __init__(self, high=4, medium=2, low=1):
        self.high = high
        self.medium = medium
        self.low = low


WEIGHTS_SYNSET_INFO = WeightsSynsetsInfo()
SYNSET_INFORMATIONS_WEIGHTS = {
    "name": WEIGHTS_SYNSET_INFO.high,
    "lemmas": WEIGHTS_SYNSET_INFO.high,
    "synonyms": WEIGHTS_SYNSET_INFO.high,
    "definition": WEIGHTS_SYNSET_INFO.medium,
    "hypernyms": WEIGHTS_SYNSET_INFO.medium,
    "hyponyms": WEIGHTS_SYNSET_INFO.low,
    "holonyms": WEIGHTS_SYNSET_INFO.low,
    "meronyms": WEIGHTS_SYNSET_INFO.low,
    "examples": WEIGHTS_SYNSET_INFO.low
}


class SynsetInfoExtractionOptions(object):
    class SynsetInfoExtrOption(object):
        def __init__(self, name, isEnabled=True, weight=1):
            self.name = name
            self.isEnabled = isEnabled
            self.weight = weight

    def __init__(self):
        self.informationsEnabled = {}
        for infoName, infoWeight in SYNSET_INFORMATIONS_WEIGHTS.items():
            self.informationsEnabled[infoName] = self.SynsetInfoExtrOption(infoName, True, infoWeight)

    def get_option(self, optionName):
        return self.informationsEnabled[optionName]

    def is_enabled(self, optionName):
        return self.get_option(optionName).isEnabled  # ["isEnabled"]

    def get_weight(self, optionName):
        return self.get_option(optionName).weight  # ["weight"]


DEFAULT_SYNSET_EXTRACTION_OPTIONS = SynsetInfoExtractionOptions()


#
#
# start document pre-processing
#
#


def get_weighted_synset_map(synset_name, cache=None, options=None):
    if cache is None:
        cache = newCache()
    if options is None:
        options = DEFAULT_SYNSET_EXTRACTION_OPTIONS
    synsets = cache.get_synsets(synset_name)
    if synsets is None:
        return None

    mapped_weights = None
    opt = options.get_option("name")
    if opt.isEnabled:
        mapped_weights = {synset_name: opt.weight}
    else:
        mapped_weights = {}

    #
    for synset in synsets:

        opt = options.get_option("definition")
        if opt.isEnabled:
            defin = synset.definition()
            # since it's a sentence, let's extract the useful word
            defin_refined = preprocessing(defin)
            for def_word in defin_refined:
                mapped_weights[def_word] = opt.weight

        # synonyms
        opt = options.get_option("lemmas")
        if opt is None:
            opt = options.get_option("synonyms")
        if opt.isEnabled:
            for synonym in synset.lemmas():
                mapped_weights[synonym.name()] = opt.weight

        opt = options.get_option("examples")
        if opt.isEnabled:
            for exampl in synset.examples():
                ex_refined = preprocessing(exampl)
                for ex_word in ex_refined:
                    mapped_weights[ex_word] = opt.weight

        # collect some stuffs
        synsets_collections_weighted = []
        opt = options.get_option("hypernyms")
        if opt.isEnabled:
            synsets_collections_weighted.append((synset.hypernyms(), opt.weight))

        opt = options.get_option("hyponyms")
        if opt.isEnabled:
            synsets_collections_weighted.append((synset.hyponyms(), opt.weight))

        opt = options.get_option("holonyms")
        if opt.isEnabled:
            synsets_collections_weighted.append((synset.member_holonyms(), opt.weight))
            synsets_collections_weighted.append((synset.part_holonyms(), opt.weight))
            synsets_collections_weighted.append((synset.substance_holonyms(), opt.weight))

        opt = options.get_option("meronyms")
        if opt.isEnabled:
            synsets_collections_weighted.append((synset.part_meronyms(), opt.weight))
            synsets_collections_weighted.append((synset.member_meronyms(), opt.weight))
            synsets_collections_weighted.append((synset.substance_meronyms(), opt.weight))

        # add the stuffs
        for coll_weighted in synsets_collections_weighted:
            we = coll_weighted[1]
            for syn in coll_weighted[0]:
                mapped_weights[syn.name()] = we

    return mapped_weights


def get_weighted_word_map_for_sentence(sentence, cache=None):
    """
    :param sentence: an English sentence in string variable
    :param cache: a CacheSynsetsBag object or None
    :return: a map <string, float> mapping each non-stop word in the given sentence into a float positive value
    """
    if cache is None:
        cache = newCache()
    sent_filtered = preprocessing(sentence)
    all_weighted_maps = [get_weighted_synset_map(w, cache=cache) for w in sent_filtered]
    sent_filtered = None  # clear the memory
    final_weighted_map = {}
    '''
    prima si collezionano tutti i pesi per una data parola della frase
    (differenti parole potrebbero aver ri-trovato la stessa parola in momenti
    diversi, ergo assegnando pesi diversi). poi si calcola una sorta
    di media e la si assegna a quella parola nella mappatura finale
    '''
    weights_collectors = {}
    for wm in all_weighted_maps:  # per ogni dizionario (ergo, per ogni parola non-stop nella frase) ..
        for wordd, weight in wm.items():  # scorro tutte le parole soppesate del dizionario
            # raccolgo i pesi in una collezione
            if wordd in weights_collectors:
                weights_collectors[wordd].append(weight)
            else:
                weights_collectors[wordd] = [weight]
    all_weighted_maps = None  # clear the memory
    # some sort of averaging ... like "arithmetic" ones
    for wordd, weights in weights_collectors.items():
        if len(weights) > 0:
            # calculate the "mean"
            # ... or just the sum ...
            final_weighted_map[wordd] = sum(weights)  # float(sum(weights) / len(weights))
        else:
            final_weighted_map[wordd] = weights[0]
    return final_weighted_map


#
#
# end pre-processing del documento
#
#


#
#
# start processing the whole document
#
#


# Deprecated, per ora
def document_segmentation(list_of_sentences, cache=None):
    if cache is None:
        cache = newCache()
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

'''
wcabs = document_segmentation(sentences)
for word, cab in wcabs.items():
    print(word, " -> (", cab[0], ";;;", cab[1])
'''

print("\n\n now the last made function: get_weighted_word_map_for_sentence")
local_cache = newCache()
for sent in sentences:
    print("\n\n\n\ngiven the sentence:\n\t--", sent, "--")
    print(get_weighted_word_map_for_sentence(sent, cache=local_cache))

print("\n\n\n\n end")
