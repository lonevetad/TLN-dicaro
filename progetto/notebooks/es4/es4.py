import math

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


class WordWeighted(object):
    def __init__(self, word):
        self.word = word
        self.cumulativeWeight = 0
        self.countPresenceInDocument = 0
        self.cacheTotalValue = -1  # invalidates cache

    def recalculateTotalValue(self):
        # the additional 1 inside the second factor is provided to still consider the word's weight,
        # even if it's not present in the document
        return int(math.floor(self.cumulativeWeight * (1 + math.log2(1 + self.countPresenceInDocument))))

    def getTotalWeigth(self):
        if self.cacheTotalValue < 0:
            self.cacheTotalValue = self.recalculateTotalValue()
        return self.cacheTotalValue

    def addWeight(self, weight, isPresentInDocument=False):
        self.cumulativeWeight += weight
        if isPresentInDocument:
            self.countPresenceInDocument += 1
            self.cacheTotalValue = -1  # invalidates cache


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


class DocumentSegmentator(object):
    def __init__(self, cache=None, options=None):
        """
        :param cache:  a CacheSynsetsBag object or None, used to cache synsets to speed up and reduce the use of Internet
        (at the cost of more memory usage)
        :param options:  a SynsetInfoExtractionOptions object or None, used to specify what information are required to be
        collected from synsets
        """
        if cache is None:
            cache = newCache()
        if options is None:
            options = DEFAULT_SYNSET_EXTRACTION_OPTIONS
        self.cache = cache
        self.options = options
        self.map_sentence_to_bag = {}
        self.algorithm_tiling = 0  # will see

    #
    #
    # start document pre-processing
    #
    #

    def get_weighted_synset_map(self, synset_name):
        """
        :param synset_name:
        :return: a map, that given a word (a synset's name), maps the weight of each words in
        that synset's definition.
        """
        synsets = self.cache.get_synsets(synset_name)
        if synsets is None:
            return None

        mapped_weights = None
        opt = self.options.get_option("name")
        if opt.isEnabled:
            mapped_weights = {synset_name: opt.weight}
        else:
            mapped_weights = {}

        #
        for synset in synsets:
            # definition
            opt = self.options.get_option("definition")
            if opt.isEnabled:
                defin = synset.definition()
                # since it's a sentence, let's extract the useful word
                defin_refined = preprocessing(defin)
                for def_word in defin_refined:
                    mapped_weights[def_word] = opt.weight

            # synonyms
            opt = self.options.get_option("lemmas")
            if opt is None:
                opt = self.options.get_option("synonyms")
            if opt.isEnabled:
                for synonym in synset.lemmas():
                    mapped_weights[synonym.name()] = opt.weight

            # examples
            opt = self.options.get_option("examples")
            if opt.isEnabled:
                for exampl in synset.examples():
                    ex_refined = preprocessing(exampl)
                    for ex_word in ex_refined:
                        mapped_weights[ex_word] = opt.weight

            # collect some stuffs
            synsets_collections_weighted = []
            opt = self.options.get_option("hypernyms")
            if opt.isEnabled:
                synsets_collections_weighted.append((synset.hypernyms(), opt.weight))

            opt = self.options.get_option("hyponyms")
            if opt.isEnabled:
                synsets_collections_weighted.append((synset.hyponyms(), opt.weight))

            opt = self.options.get_option("holonyms")
            if opt.isEnabled:
                synsets_collections_weighted.append((synset.member_holonyms(), opt.weight))
                synsets_collections_weighted.append((synset.part_holonyms(), opt.weight))
                synsets_collections_weighted.append((synset.substance_holonyms(), opt.weight))

            opt = self.options.get_option("meronyms")
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

    def get_weighted_word_map_for_sentence(self, sentence):
        """
        :param sentence:  an English sentence in string variable
        :return:  a map <string, float> mapping each non-stop word in the given sentence
        into a float positive value: a weight indicating how much that word helps in
        describing the argument of the sentence
        """
        sent_filtered = None
        #print(".... processing sentence: ", sentence)
        if sentence in self.map_sentence_to_bag:
            sent_filtered = self.map_sentence_to_bag[sentence]
        else:
            sent_filtered = preprocessing(sentence)
            self.map_sentence_to_bag[sentence] = sent_filtered
        all_weighted_maps = [self.get_weighted_synset_map(w) for w in sent_filtered]
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

    def weighted_intersection(self, word_weight_map, string_set1, string_set2):
        if len(string_set2) < len(string_set1):
            return self.weighted_intersection(word_weight_map, string_set2, string_set1)
        # consider the first as the smaller set
        summ = 0
        for w in string_set1:
            if w in string_set2:
                summ += word_weight_map[w].getTotalWeigth()
        return summ

    def weighted_overlap(self, word_weight_map, string_set1, string_set2):
        wi_sum = self.weighted_overlap(word_weight_map, string_set1, string_set2)
        return wi_sum / min(len(string_set1), len(string_set2))

    def similarity(self, word_weight_map, string_set1, string_set2):
        return self.weighted_intersection(word_weight_map, string_set1, string_set2)

    #

    def compute_similarity_lists_subsequent_sentences(self, list_of_sentences, word_weight_map):
        """
        :param list_of_sentences: the parameter of the function "document_segmentation"
        :param word_weight_map: map <string, WordWeighted> that maps the weight of each words
        :return: a list, with length equal to "len(document_segmentation)-1", containing
        holding the similarity score between two subsequent sentences
        (the similarity between sentences in index 0 and 1 is stored in index 0).
        """
        i = 0
        leng = len(list_of_sentences) - 1
        simils = [0] * leng
        while i < leng:
            simils[i] = self.similarity(word_weight_map, list_of_sentences[i], list_of_sentences[i + 1])
            i += 1
        return simils

    def doc_tiling(self, similarity_subsequent_sentences, windows_count, max_iterations=0):
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

    def document_segmentation(self, list_of_sentences, desiredParagraphAmount=0):
        if not (isinstance(list_of_sentences, list)):
            return None
        if desiredParagraphAmount < 2:
            desiredParagraphAmount = 2
        words_mapped_each_sentences = [self.get_weighted_word_map_for_sentence(sentence) for sentence in
                                       list_of_sentences]
        words_weight = {}  # contiene i pesi finali delle parole
        # aggreghiamo i pesi delle parole:
        # ispirandosi all term-frequency, il peso finale è
        # la somma di tutti i pesi moltiplicta per floor(1+log(numero occorrenze nel documento))
        i = 0
        for map_for_a_sent in words_mapped_each_sentences:
            for word, weight in map_for_a_sent.items():
                bag_of_word_of_sentence = self.map_sentence_to_bag[list_of_sentences[i]]
                is_in_sentence = word in bag_of_word_of_sentence
                if word in words_weight:
                    words_weight[word].addWeight(weight, isPresentInDocument=is_in_sentence)
                else:
                    w = WordWeighted(word)
                    w.addWeight(weight, isPresentInDocument=is_in_sentence)
                    words_weight[word] = w
            i += 1

        # pre-compute the similarity of each sentence
        similarity_subsequent_sentences = self.compute_similarity_lists_subsequent_sentences(
            list_of_sentences, words_weight)

        # now segment
        breakpoint_indexes = self.doc_tiling(similarity_subsequent_sentences, desiredParagraphAmount)
        breakpoint_indexes.append(len(similarity_subsequent_sentences))
        i = 0
        start = 0
        subdivision = [None] * desiredParagraphAmount
        while i < desiredParagraphAmount:
            subdivision = list_of_sentences[start: breakpoint_indexes[i]]
            start = breakpoint_indexes[i]
            i += 1
        return subdivision


'''
# Deprecated, per ora
def document_segmentation_v1(list_of_sentences, cache=None):
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
'''

sentences = [
    "I love to pet my cat while reading fantasy books.",
    "The cat's furry is so fluffy that chills me.",
    "It entertain me, its tail sometimes goes over my pages and meows.",
    "Sometimes, it plays with me but hurts me with its claws",
    "Even so, its agile and soft body warms me with its cute fur, meow and purr."
    
    "There are tons of books I like, from literature, romance, fantasy and sci-fi.",
    "When i hold a book, the stress flushes out and I start reading the whole life wit an external and more critical mindset.",
    "Sometimes, some author or topic can help to better understand the world, as just by giving a meaning",
    "Mostly, I grab a book to just distract the mind in some other reality.",
    "Fantasy is my best genre because let my imagination to run freely.",
    "Reading books makes my fantasy fly over everyday problems.",

    "One problem of those is my cat vomiting on my pants.",
    "But, dealing with people is way more problematic and causes me stomach issues."
    "I find difficult to deal with people, they are usually focused in their egoistic desires and purposes.",
    "Most of them just treat others as resources to achieve their objectives and gets upset if You don't fulfill their expectations.",
    "Sometimes they also undervaluate your own problem, like they are nothing compared to theirs.",
    "Sometimes they even stop listening as you start talking about your own problems, like they are annoyed.",
    "It's a no surprise if happens that I found a better dialogue with a book or my cat."
]

'''
wcabs = document_segmentation(sentences)
for word, cab in wcabs.items():
    print(word, " -> (", cab[0], ";;;", cab[1])
'''

print("\n\n now the last made function: get_weighted_word_map_for_sentence")
local_cache = newCache()
'''
for sent in sentences:
    print("\n\n\n\ngiven the sentence:\n\t--", sent, "--")
    print(get_weighted_word_map_for_sentence(sent, cache=local_cache))
'''
ds = DocumentSegmentator(cache=local_cache)
paragraphs = ds.document_segmentation(sentences, 3)
for p in paragraphs:
    print("\n\n paragraph:")
    for s in p:
        print("----", s)

print("\n\n\n\n end")
