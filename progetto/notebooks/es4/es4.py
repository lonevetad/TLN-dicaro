import math

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

from sortedcontainers import SortedDict

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


def get_preprocessed_words(text):
    return preprocessing(text)


# ------------
# ------------
# ------------------ classes ------------------
# ------------
# ------------


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


#
#
# start CachePairwiseSentenceSimilarity
#
#

class CachePairwiseSentenceSimilarity:
    def __init__(self, map_words_weights, list_word_bags_from_sentences, sentence_similarity_function):
        """
        :param map_words_weights: map <string, int> representing the words' weights
        :param list_word_bags_from_sentences: a list of set of string, i.e. a list of bag of words
        each extracted from a respective sentence
        :param sentence_similarity_function: a function that accepts a map <string, int> (words' weights),
        and two strings (sentences) and returns a float (the similarity between those)
        """
        self.map_words_weights = map_words_weights
        self.list_word_bags_from_sentences = list_word_bags_from_sentences
        self.sentence_similarity_function = sentence_similarity_function
        self.cache_simil = [None] * len(list_word_bags_from_sentences)

    def get_similarity_by_sentence_indexes(self, index_sentence1, index_sentence2):
        minind = 0
        maxind = 0
        if index_sentence1 > index_sentence2:
            minind = index_sentence2
            maxind = index_sentence1
        elif index_sentence1 == index_sentence2:
            raise ValueError(
                "Shouldn't calculate the similarity of the same sentence (index: " + str(index_sentence1) + ")")
        else:
            minind = index_sentence1
            maxind = index_sentence2
        m = self.cache_simil[minind]
        if m is None:
            m = {}
            self.cache_simil[minind] = m
        if maxind in m:
            return m[maxind]
        else:
            sim = self.sentence_similarity_function(self.map_words_weights,
                                                    self.list_word_bags_from_sentences[index_sentence1],
                                                    self.list_word_bags_from_sentences[index_sentence2])
            m[maxind] = sim
            return sim


#
#
# end CachePairwiseSentenceSimilarity
#
#

#
#
# start Paragraph
#
#

class Paragraph(object):
    def __init__(self, documentSegmentator):  # , cache_pairwise_sentence_similarity
        if not (isinstance(documentSegmentator, DocumentSegmentator)):
            raise ValueError("The first constuctor parameter must be a DocumentSegmentator")

        # if not (isinstance(cache_pairwise_sentence_similarity, CachePairwiseSentenceSimilarity)):
        #    raise ValueError("The second constuctor parameter must be a CachePairwiseSentenceSimilarity")
        self.documentSegmentator = documentSegmentator
        self.score = -1
        # self.cache_pairwise_sentence_similarity = cache_pairwise_sentence_similarity
        # self.map_sentence_by_index = SortedDict()  # all sentences that builds the paragraph
        self.lowest_index_sentence = 0
        self.highest_index_sentence = -1  # ESTREMI INCLUSI
        self.previous_paragraph = None
        self.next_paragraph = None
        if not self.is_empty():
            raise ValueError("MA NON HA SENSO")

    def is_empty(self):
        # return len(self.map_sentence_by_index) == 0
        return self.lowest_index_sentence > self.highest_index_sentence

    def __adjust_indexes__(self):
        # fix wrong data
        if self.lowest_index_sentence < 0:
            self.lowest_index_sentence = 0
        if self.highest_index_sentence >= len(self.documentSegmentator.list_of_sentences):
            self.highest_index_sentence = len(self.documentSegmentator.list_of_sentences) - 1

    def raiseNonContiguousError(self, par):
        raise ValueError("Non contiguous paragraphs: self:(" + str(self.lowest_index_sentence) + ";" +
                         str(self.highest_index_sentence) + "), given:(" +
                         str(par.lowest_index_sentence) + ";" + str(par.highest_index_sentence) + ")")

    def merge_paragraph(self, par):
        """
        Merge the "lowest" (in term of starting index) paragraph
        into the "highest".
        Will rise an exception if they are not contiguous.
        :param par: a given paragraph
        :return: the remaining paragraph, or None in case of non-Paragraph parameter
        """
        if not isinstance(par, Paragraph):
            return None
        if self.lowest_index_sentence > par.lowest_index_sentence:
            return par.merge_paragraph(self)
        # I'm the lowest
        if ((
                    self.highest_index_sentence + 1) != par.lowest_index_sentence) or self.next_paragraph != par or par.previous_paragraph != self:
            self.raiseNonContiguousError(par)
        self.highest_index_sentence = par.highest_index_sentence
        # merge links
        if par.next_paragraph is not None:
            par.next_paragraph.previous_paragraph = self
        # if self.previous_paragraph is not None:
        #    self.previous_paragraph.next_paragraph
        self.next_paragraph = par.next_paragraph
        par.next_paragraph = None
        par.previous_paragraph = None
        return self

    def add_sentence(self, sentence, i, is_start_of_paragraph=True):
        self.score = -1
        # self.map_sentence_by_index[i] = sentence
        if self.is_empty():
            self.lowest_index_sentence = i
            self.highest_index_sentence = i
        else:
            print("NOT empty: i: ", str(i))
            if is_start_of_paragraph:
                self.lowest_index_sentence = i
            else:
                self.highest_index_sentence = i
        self.__adjust_indexes__()

    def remove_sentence(self, i):
        self.score = -1
        # self.map_sentence_by_index.pop(i)
        if i == self.lowest_index_sentence:
            self.lowest_index_sentence += 1
        elif i == self.highest_index_sentence:
            self.lowest_index_sentence -= 1
        self.__adjust_indexes__()

    def get_score(self):
        if self.is_empty():
            return 0
        if self.score >= 0:
            return self.score
        self.score = 0
        # for i, sent1 in self.map_sentence_by_index.items():
        #    for j, sent2 in self.map_sentence_by_index.items():
        for i in range(self.lowest_index_sentence, self.highest_index_sentence + 1):
            for j in range(self.lowest_index_sentence, self.highest_index_sentence + 1):
                if i != j:
                    # self.score += self.cache_pairwise_sentence_similarity.get_similarity_by_sentence_indexes(
                    self.score += self.documentSegmentator.cache_bag_sentence_similarity \
                        .get_similarity_by_sentence_indexes(
                        # self.documentSegmentator.get_bag_of_word_of_sentence_by_index(i),
                        # self.documentSegmentator.get_bag_of_word_of_sentence_by_index(j)
                        i, j
                    )
        self.score /= float(len(self.map_sentence_by_index) * (len(self.map_sentence_by_index) - 1))
        return self.score

    def get_first_sentence_index(self):
        """
        The extremes are included.
        :return: the index of the first sentence held by this paragraph
        """
        if self.is_empty():
            return -1
        return self.lowest_index_sentence

    def get_first_sentence(self):
        if self.is_empty():
            return None
        # return self.map_sentence_by_index.peekitem(0)
        return self.documentSegmentator.list_of_sentences[self.lowest_index_sentence]

    def get_last_sentence_index(self):
        """
        The extremes are included.
        :return: the index of the last sentence held by this paragraph
        """
        if self.is_empty():
            return -1
        return self.highest_index_sentence

    def get_last_sentence(self):
        if self.is_empty():
            return None
        # return self.map_sentence_by_index.peekitem(0)
        return self.documentSegmentator.list_of_sentences[self.highest_index_sentence]

    def toString(self):
        sss = "P[(" + str(self.lowest_index_sentence) + "; " + str(self.highest_index_sentence) + ")"
        if self.previous_paragraph is None:
            sss += ", no prev"
        else:
            sss += ", prev-" + str(self.previous_paragraph.lowest_index_sentence)
        if self.next_paragraph is None:
            sss += ", no next"
        else:
            sss += ", next-" + str(self.next_paragraph.lowest_index_sentence)
        return sss + "]"


#
#
# end Paragraph
#
#

#
#
# start DocumentSegmentator
#
#

class DocumentSegmentator(object):
    def __init__(self, list_of_sentences, cache=None, options=None):
        """
        :param list_of_sentences: list of sentences
        :param cache:  a CacheSynsetsBag object or None, used to cache synsets to speed up and reduce the use of Internet
        (at the cost of more memory usage)
        :param options:  a SynsetInfoExtractionOptions object or None, used to specify what information are required to be
        collected from synsets
        """
        if cache is None:
            cache = newCache()
        if options is None:
            options = DEFAULT_SYNSET_EXTRACTION_OPTIONS
        self.list_of_sentences = list_of_sentences
        self.cache = cache
        self.options = options
        self.map_sentence_to_bag = {}
        self.bag_from_sentence_list = []
        self.algorithm_tiling = 0  # will see
        self.cache_bag_sentence_similarity = None

    # start document pre-processing

    def firstIndexOf(self, stri, a_char):
        i = 0
        le = len(stri)
        if le == 0:
            return -1
        notFound = True
        while notFound and i < le:
            notFound = stri[i] != a_char
            if notFound:
                i += 1
        return -1 if notFound else i

    def get_sentence_by_index(self, i):
        return self.list_of_sentences[i]

    def get_bag_of_word_of_sentence_by_sentence(self, sentence):
        return self.map_sentence_to_bag[sentence]

    def get_bag_of_word_of_sentence_by_index(self, i):
        return self.get_bag_of_word_of_sentence_by_sentence(self.list_of_sentences[i])

    def get_weighted_synset_map(self, synset_name):
        """
        :param synset_name:
        :return: a map, that given a word (a synset's name), maps the weight of each words in
        that synset's definition.
        """
        if len(synset_name) < 3:
            return None
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
                    if len(def_word) > 1:
                        mapped_weights[def_word] = opt.weight

            # synonyms
            opt = self.options.get_option("lemmas")
            if opt is None:
                opt = self.options.get_option("synonyms")
            if opt.isEnabled:
                for synonym in synset.lemmas():
                    nnn = synonym.name()
                    index_dot = self.firstIndexOf(nnn, '.')
                    if index_dot >= 0:
                        nnn = nnn[0:index_dot]
                    if len(nnn) > 1:
                        mapped_weights[synonym.name()] = opt.weight

            # examples
            opt = self.options.get_option("examples")
            if opt.isEnabled:
                for exampl in synset.examples():
                    ex_refined = preprocessing(exampl)
                    for ex_word in ex_refined:
                        if len(ex_word) > 1:
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
                    namee = syn.name()
                    index_dot = self.firstIndexOf(namee, '.')
                    if index_dot >= 0:
                        namee = namee[0:index_dot]
                    if len(namee) > 1:
                        mapped_weights[namee] = we
        return mapped_weights

    def get_weighted_word_map_for_sentence(self, sentence):
        """
        :param sentence:  an English sentence in string variable
        :return:  a map <string, float> mapping each non-stop word in the given sentence
        into a float positive value: a weight indicating how much that word helps in
        describing the argument of the sentence
        """
        sent_filtered = None
        # print(".... processing sentence: ", sentence)
        if sentence in self.map_sentence_to_bag:
            sent_filtered = self.map_sentence_to_bag[sentence]
        else:
            sent_filtered = preprocessing(sentence)
            self.map_sentence_to_bag[sentence] = sent_filtered
            self.bag_from_sentence_list.append(sent_filtered)
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
            if wm:
                for wordd, weight in wm.items():  # scorro tutte le parole soppesate del dizionario
                    # raccolgo i pesi in una collezione
                    if len(wordd) < 2:
                        raise ValueError("WTF 2: --" + str(wordd) + "--")
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
        wi_sum = self.weighted_intersection(word_weight_map, string_set1, string_set2)
        return wi_sum / min(len(string_set1), len(string_set2))

    def similarity(self, word_weight_map, string_set1, string_set2):
        """
        similarity function
        :param word_weight_map: a map <string, int> representing the words' weights
        :param string_set1: a set of words, extracted from a sentence
        :param string_set2: as the previous parameter
        :return: a float, indicating how much those sentences are similar
        """
        return self.weighted_overlap(word_weight_map, string_set1, string_set2)

    #

    def compute_similarity_lists_subsequent_sentences(self, bags_sentences, word_weight_map):
        """
        :param bags_sentences: list of bags of words, extracter from the sentences
        (whose are the parameter of the function "document_segmentation") during the computation
        of the function "get_weighted_synset_map"
        :param word_weight_map: map <string, WordWeighted> that maps the weight of each words
        :return: a list, with length equal to "len(document_segmentation)-1", containing
        holding the similarity score between two subsequent sentences
        (the similarity between sentences in index 0 and 1 is stored in index 0).
        """
        i = 0
        leng = len(bags_sentences) - 1
        simils = [0] * leng
        while i < leng:
            # simils[i] = self.similarity(word_weight_map, bags_sentences[i], bags_sentences[i + 1])
            simils[i] = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(bags_sentences[i],
                                                                                              bags_sentences[i + 1])
            i += 1
        return simils

    def doc_tiling_v1(self, similarity_subsequent_sentences, windows_count, max_iterations=0):
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

    def doc_tiling(self, windows_count, max_iterations=0):
        """
        :param windows_count: the amount of paragraph to find. It's greater by 1 than the length of
        the returned list
        :param max_iterations: the inner algorithm improves iteratively the tiling; this parameter
        sets an upper bound of iterations
        :return: a list of breakpoints: indexes (between one sentence and the next) where one paragraph ends and
        the next starts. The length is equal to the parameter "windows_count", so the last index is the last sentence.
        So, the indexes are to be considered as "inclusive".
        """
        '''
        all'inizio:
        - si genera un Paragraph per ogni frase
        - per ogni Paragraph (tranne primo ed ultimo, che è scontato)
            si calcola quale frase tra la precedente e la successiva
            è la migliore candidata per la fusione dei paragrafi
        - (convertire i backpointer per comodita', vedere dopo)
        - fondere ogni paragrafo con quello puntato
        
        POI
        
        '''
        sentences = self.list_of_sentences
        sentences_amount = len(sentences)
        paragraphs_by_starting_index = SortedDict()

        # maps < a paragraph's lowest index -> the par.'s lowest index wish to merge into
        preferences = SortedDict()

        def print_paragraph_map(p_m):
            for ind, ppp in p_m.items():
                print("with index", ind, ", paragraph ->", ppp.toString())

        # inizializzazione
        i = 0
        prevParagraph = None
        while i < sentences_amount:  # creo i paragrafi
            par = Paragraph(self)
            par.add_sentence(sentences[i], i)
            if prevParagraph is not None:
                prevParagraph.next_paragraph = par
                par.previous_paragraph = prevParagraph
            paragraphs_by_starting_index[i] = par
            prevParagraph = par
            i += 1

        # cerco le preferenze iniziali
        for j, par in paragraphs_by_starting_index.items():
            if j == 0:
                # scelta obbligata
                preferences[0] = 1
            elif j == (sentences_amount - 1):
                # scelta obbligata
                preferences[j] = j - 1  # il penultimo
            else:
                simil_prev = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(j - 1, j)
                simil_next = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(j, j + 1)
                if simil_next >= simil_prev:
                    preferences[j] = j + 1
                else:
                    preferences[j] = j - 1
                    '''
                    V2
                    a ragion veduta, si puo' risparmiare la conversione dei backpointer
                    mettendola qui, dato che le "back-chain" vengono costruite iterativamente
                    quando si cade, consecutivamente, in questo ramo
                    
                    pref_of_prev = preferences[j - 1]
                    if j > 1 and pref_of_prev < (j - 1):
                        preferences[j - 1] = j  # conversione del backpointer
                    '''

        print("\n\n preferences at start:")
        print(preferences)
        '''
        V1
        codice lasciato per ragioni "storiche", rimosso dopo il ragionamento (commento multilinea)
        di cui sopra
        ...
        '''
        # conversione di tutti i backpointers
        # perche' tanto andranno a finire nello stesso paragrafo
        i = sentences_amount - 1
        while i > 0:
            pref = preferences[i]
            if pref < i:
                # search for the start of a "back sequence": convert all other stuffs
                j = pref  # j == i-1 per costruzione
                # a do-while ...
                even_prev = preferences[j]
                # search_not_done = True
                # while search_not_done:
                while even_prev < j and 0 < j:
                    preferences[j] = j + 1
                    j = even_prev
                    even_prev = preferences[j]
                # j holds the last of the backward chain (i.e., the first to be redirected
                preferences[
                    j] = j + 1  # qualora non si entrasse nel ciclo (paragrafo grande 2), semplicemente si riconfermera' la precedenza
                i = j - 1
            else:
                i -= 1

        print("\n\n\n post-aggiustamento dei backpointer:\npreferenze:")
        print(preferences)
        print("paragraphs :")
        print_paragraph_map(paragraphs_by_starting_index)

        print("\n\n ora si fanno i merge")
        # ora i backpointers possono essere trattati come "terminatori di paragrafo"
        # merge delle preferenze:
        start = 0
        end = 0
        # V2
        # si procede a ritroso per semplicita
        end = sentences_amount - 1
        start = end
        # per ogni paragrafo
        while start > 0:
            start = end - 1
            # almeno due elementi nel paragrafo, per costruzione
            par_end = paragraphs_by_starting_index[end]
            paragraphs_by_starting_index[start] = paragraphs_by_starting_index[start].merge_paragraph(par_end)
            paragraphs_by_starting_index.pop(end)
            print("popped end:", end, " before hard work")
            start -= 1  # jump to the next (previous, tbh) sentence
            pref = preferences[start]
            if start < pref:
                # sequence not ended: the current paragraph (end-1) could be merged into start
                while 0 <= start < pref:
                    paragraphs_by_starting_index[start] = paragraphs_by_starting_index[start].merge_paragraph(
                        paragraphs_by_starting_index[pref])
                    paragraphs_by_starting_index.pop(pref)
                    start -= 1
                    if 0 <= start:
                        pref = preferences[start]
                end = start
            else:
                # the paragraph has ended: start is pointing backward
                end = start

        # manage the first element: nothing is done if start == 0
        # paragraphs_by_starting_index[0].merge_paragraph(paragraphs_by_starting_index[1])  # per costruzione dei pointer
        # paragraphs_by_starting_index.pop(1)

        print("\n\n paragrafi DOPO i merge")
        print(preferences)
        print("paragraphs :")
        print_paragraph_map(paragraphs_by_starting_index)

        '''
            V1
        while start < sentences_amount:
            end = preferences[start]
            if end < start:
                # backward link
                todo;

                start += 1
            else:
                canSearch = True
                last_end = end
                while canSearch:
                    end = preferences[end]
                    if end < last_end:
                        # backward pointer -> end of all
                        canSearch = False
                    else:
                        last_end = end
                end = start + 1 # riciclato come un iteratore
                while end <= last_end:
                    par_to_merge = paragraphs_by_starting_index[end]
                    par.merge_paragraph(par_to_merge)
                    #remove its reference
                    end += 1
                start = end #move to the next
        '''

        # WELL, INITIALIZATION HAS ENDED
        # now make the paragraphs-bubble boiling
        # need to use max_iterations
        '''
        for iter in range(0, max_iterations):
            par = paragraphs_by_starting_index[0]
            while par.next_paragraph is not None:
                current_score = par.get_score()
                next_score = par.next_paragraph.get_score()
                par_sent_index = par.get_last_sentence_index()
                next_sent_index = par.next_paragraph.get_first_sentence_index()
                # proviamo a spostare la first frase in par
                
                # ora spostiamo l'ultima di par nel next
                
                # selezione del migliore dei tre casi
                prev_score
                
                par = par.next_paragraph
'''
        # conversione in array di indici
        indexes = [par.highest_index_sentence for i, par in paragraphs_by_starting_index.items()]
        # for i, par in paragraphs_by_starting_index.items():
        print("\n\n\nindexes:")
        print(indexes)
        return indexes

    def document_segmentation(self, desiredParagraphAmount=0):
        if desiredParagraphAmount < 2:
            desiredParagraphAmount = 2
        words_mapped_each_sentences = [self.get_weighted_word_map_for_sentence(sentence) for sentence in
                                       self.list_of_sentences]
        words_weight = {}  # contiene i pesi finali delle parole
        # aggreghiamo i pesi delle parole:
        # ispirandosi all term-frequency, il peso finale è
        # la somma di tutti i pesi moltiplicta per floor(1+log(numero occorrenze nel documento))
        i = 0
        for map_for_a_sent in words_mapped_each_sentences:
            for word, weight in map_for_a_sent.items():
                bag_of_word_of_sentence = self.map_sentence_to_bag[self.list_of_sentences[i]]
                is_in_sentence = word in bag_of_word_of_sentence
                if len(word) <= 1:
                    print("\n\n WTF in sentence:")
                    print(self.list_of_sentences[i])
                    print("and word: ---", word, "---")
                    print("and bag")
                    print(map_for_a_sent)
                    print("\nweights collected since then")
                    print(words_weight)
                    raise ValueError("WTF ?" + word + "--\n\n")
                if word in words_weight:
                    words_weight[word].addWeight(weight, isPresentInDocument=is_in_sentence)
                else:
                    w = WordWeighted(word)
                    w.addWeight(weight, isPresentInDocument=is_in_sentence)
                    words_weight[word] = w
            i += 1

        self.cache_bag_sentence_similarity = CachePairwiseSentenceSimilarity(
            map_words_weights=words_weight,
            list_word_bags_from_sentences=self.bag_from_sentence_list,
            sentence_similarity_function=self.similarity
        )
        # print("words_weight:")
        # print(words_weight)
        # pre-compute the similarity of each sentence

        '''
        #V1, pre-version V2:
        similarity_subsequent_sentences = self.compute_similarity_lists_subsequent_sentences(
            self.bag_from_sentence_list, words_weight)
        print("\n\n generated similarity_subsequent_sentences:")
        print(similarity_subsequent_sentences)

        # now segment
        breakpoint_indexes = self.doc_tiling(similarity_subsequent_sentences, desiredParagraphAmount)
        breakpoint_indexes.append(len(similarity_subsequent_sentences))
        '''

        # now segment
        breakpoint_indexes = self.doc_tiling(desiredParagraphAmount)
        desiredParagraphAmount = len(breakpoint_indexes) # forzo il fatto di mantenere i paragrafi
        i = 0
        start = 0
        subdivision = [None] * desiredParagraphAmount
        print("\n\n generated breakpoint_indexes:")
        print(breakpoint_indexes)
        while i < desiredParagraphAmount:
            subdivision[i] = self.list_of_sentences[start: breakpoint_indexes[i] + 1]
            start = breakpoint_indexes[i] + 1
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

sentences_mocked = [
    "I love to pet my cat while reading fantasy books.",
    "The fur of my cat is so fluffy that chills me.",
    "It entertain me, its tail sometimes goes over my pages and meows, also showing me the weird pattern in its chest's fur.",
    "Sometimes, it plays with me but hurts me with its claws and once it scratched me so bad the one drop of blood felt over my favourite book's pages.",
    "Even so, its agile and soft body warms me with its cute fur, meow and purr, so I've forgave it."

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
ds = DocumentSegmentator(sentences_mocked, cache=local_cache)
paragraphs = ds.document_segmentation(3)
for p in paragraphs:
    print("\n\n paragraph:")
    for s in p:
        print("----", s)

print("\n\n\n\n end")
