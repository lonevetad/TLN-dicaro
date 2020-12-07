# from collections.abc import Iterable
from typing import Dict, Iterable

from notebooks.utilities.cacheVarie import CacheSynsets
from notebooks.utilities.functions import preprocessing, first_index_of


# questo file è nato per l'esercizio 4 e poi utilizzato per migliorare il 2


class WeightsSynsetsInfo(object):
    """
    Insieme (con nome) di pesi.
    """

    def __init__(self, high=4, medium=2, low=1):
        self.high = high
        self.medium = medium
        self.low = low


WEIGHTS_SYNSET_INFO = WeightsSynsetsInfo()  # una istanza di default

# assegnamento di pesi di default, usato nella classe "SynsetInfoExtractionOptions"
SYNSET_INFORMATIONS_WEIGHTS = {  # le varie parti di un synset, pesate
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
    """
    Classe usata soprattutto nella funzione "weighted_bag_words_from_word" per
    definire quali informazioni estrarre (ed accorpare) da un synset
    """

    class SynsetInfoExtrOption(object):
        def __init__(self, name, isEnabled=True, weight=1):
            self.name = name
            self.isEnabled = isEnabled
            self.weight = weight

    def __init__(self, iterable_predefined_options=SYNSET_INFORMATIONS_WEIGHTS.items()):
        """
        Builds an instance of SynsetInfoExtractionOptions
        :param iterable_predefined_options: iterable of pre-defined options in a pair <string, int> (<name, weight>). Only the enabled options should be provided here.
        """
        self.informationsEnabled = {}
        for infoName, infoWeight in iterable_predefined_options:
            # self.informationsEnabled[infoName] = self.SynsetInfoExtrOption(infoName, True, infoWeight)
            self.add_option(infoName, infoWeight, True)

    # getter

    def get_option(self, optionName):
        return self.informationsEnabled[optionName]  # if optionName in self.informationsEnabled else None

    def is_enabled(self, optionName):
        return self.get_option(optionName).isEnabled if optionName in self.informationsEnabled else False

    def get_weight(self, optionName):
        return self.get_option(optionName).weight  # ["weight"]

    # setter

    def set_weight(self, optionName, weight):
        if optionName in self.informationsEnabled:
            self.get_option(optionName).weight = weight
        else:
            self.add_option(optionName, weight, True)

    def set_is_enabled(self, optionName, isEnabled):
        self.get_option(optionName).isEnabled = isEnabled

    # altro

    def add_option(self, optionName, weight, isEnabled):
        o = self.SynsetInfoExtrOption(optionName, isEnabled, weight)
        self.informationsEnabled[optionName] = o

    def remove_option(self, optionName):
        if optionName in self.informationsEnabled:
            del self.informationsEnabled[optionName]


# FINE CLASSE SynsetInfoExtractionOptions


DEFAULT_SYNSET_EXTRACTION_OPTIONS = SynsetInfoExtractionOptions()


#
#
#
#
#

#
#
# single word

def rightful_name_from_synset(name):
    index_dot = first_index_of(name, '.')
    if index_dot >= 0:
        name = name[0:index_dot]
    return name if len(name) > 1 else None


def weighted_bag_for_word(synset_name: str, cache_synset_and_bag=None, options=None) -> Dict[str, int] or None:
    """
    :param synset_name: the string representing the word of the synset that will
    generate the set of information required (this name is a bit a scammer ...)
    :param cache_synset_and_bag: an instance of CacheSynsets or None, used to retrieve
    synset instances
    :param options: an instance of SynsetInfoExtractionOptions, used to define
    what kind of information should be retrieved to build the map
    :return: a map, that given a word (a synset's name), maps the weight of each words in
    that synset's definition.
    """
    synset_name = synset_name.strip()
    if len(synset_name) < 2:
        return None
    if cache_synset_and_bag is None:
        cache_synset_and_bag = CacheSynsets()
    if options is None:
        options = SynsetInfoExtractionOptions()
    synsets = cache_synset_and_bag.get_synsets(word_text=synset_name)  # prendo il synset
    if synsets is None:
        return None

    mapped_weights = None
    if options.is_enabled("name"):
        namee = rightful_name_from_synset(synset_name)
        if namee is not None:
            mapped_weights = {namee: options.get_weight("name")}
    else:
        mapped_weights = {}

    #
    for synset in synsets:
        # definition
        if options.is_enabled("definition"):
            www = options.get_weight("definition")
            defin = synset.definition()
            # since it's a sentence, let's extract the useful word
            defin_refined = preprocessing(defin)
            for def_word in defin_refined:
                if len(def_word) > 1:
                    mapped_weights[def_word] = www

        # examples
        if options.is_enabled("examples"):
            www = options.get_weight("examples")
            for exampl in synset.examples():
                ex_refined = preprocessing(exampl)
                for ex_word in ex_refined:
                    if len(ex_word) > 1:
                        mapped_weights[ex_word] = www

        # collect some stuffs
        synsets_collections_weighted = []

        if options.is_enabled("hypernyms"):
            synsets_collections_weighted.append((synset.hypernyms(), options.get_weight("hypernyms")))

        if options.is_enabled("hyponyms"):
            synsets_collections_weighted.append((synset.hyponyms(), options.get_weight("hyponyms")))

        if options.is_enabled("holonyms"):
            www = options.get_weight("holonyms")
            synsets_collections_weighted.append((synset.member_holonyms(), www))
            synsets_collections_weighted.append((synset.part_holonyms(), www))
            synsets_collections_weighted.append((synset.substance_holonyms(), www))

        if options.is_enabled("meronyms"):
            www = options.get_weight("meronyms")
            synsets_collections_weighted.append((synset.part_meronyms(), www))
            synsets_collections_weighted.append((synset.member_meronyms(), www))
            synsets_collections_weighted.append((synset.substance_meronyms(), www))

        # synonyms
        opt = options.get_option("lemmas")
        if opt is None:
            opt = options.get_option("synonyms")
        if (opt is not None) and opt.isEnabled:
            synsets_collections_weighted.append((synset.lemmas(), opt.weight))

        # add the stuffs
        for coll_weighted in synsets_collections_weighted:
            we = coll_weighted[1]
            for syn in coll_weighted[0]:
                namee = rightful_name_from_synset(syn.name())  # sometimes the name is like "dog.n.01" ... not what I want (i.e. "dog")
                if namee is not None:
                    mapped_weights[namee] = we
    return mapped_weights


#
#
# full sentence

class FilteredWeightedWordsInSentence(object):
    """
    Holder of information (fields) extracted from a given sentence ("original_sentence", as constructor's parameter):
        - filtered_words:  a set of non-"stop", non-punctuation words taken from the sentence.
        - word_to_weight_mapping:  a map <string, int> mapping each non-stop word in the definitions of the words (see "weighted_bag_words_from_word") present in the given sentence into a float positive value: a weight indicating how much that word helps in describing the concept expressed by the sentence. Those weights are approximations.
        - mapping_word_to_its_weighted_bag: (it's a map<string, map<string, int>>) during the construction of "word_to_weight_mapping" (defined above), the function "weighted_bag_words_from_word" is applied to every words in the "original_sentence" to grab its definition and sense (a bag of weighted words), in some way. All of those weighted maps are combined to form the field "word_to_weight_mapping" (as stated before), but they could be useful somewhere: to not lose them and re-calculate from scratch, they are held by this mapping.
    """

    def __init__(self, original_sentence: str or Iterable[str]):
        """
        :param original_sentence: an English sentence as a string, or a set of words as strings
        """
        if isinstance(original_sentence, set):
            self.filtered_words = original_sentence
            original_sentence = " ".join(original_sentence)  # don't care: you'll be a single string again!
        elif not isinstance(original_sentence, str):
            raise ValueError("the given original_sentence is NOT a string nor a set")
        else:
            self.filtered_words = preprocessing(original_sentence)  # split me
        self.original_sentence = original_sentence
        # it's the final map-dooooooown !! tiritiii tuuuuu .. tiritittuttuuuuu ..
        self.word_to_weight_mapping: Dict[str, int] = {}  # /.\
        self.mapping_word_to_its_weighted_bag: Dict[str, Dict[str, int]] = {}  # serve nell'esercizio 2 ....


#def merge_weighted_bags(all_weighted_bags: Iterable[Dict[str:int]]) -> Dict[str:int]:
def merge_weighted_bags(all_weighted_bags):
    """
    :param all_weighted_bags: iterable of map<string,int>, which are a map assigning an integer weight to some words (string)
    :return: a huge map<string,int> merging all of the given ones. If some words are duplicated, the corresponding weights
    are merged in some way (like performing a mean, or just a simple sum).
    """
    final_weighted_map: Dict[str:int] = {}  # {"can you spot me? :D": 0}
    weights_collectors = {}
    for wm in all_weighted_bags:  # per ogni dizionario (ergo, per ogni parola non-stop nella frase) ..
        if wm:
            for wordd, weight in wm.items():  # scorro tutte le parole soppesate del dizionario
                # raccolgo i pesi in una collezione
                if wordd in weights_collectors:
                    weights_collectors[wordd].append(weight)
                else:
                    weights_collectors[wordd] = [weight]
    all_weighted_maps = None  # clear the memory
    # some sort of averaging ... like "arithmetic" ones
    for wordd, weights in weights_collectors.items():
        if len(weights) > 1:
            # calculate the "mean"
            # final_weighted_map[wordd] = float(sum(weights) / len(weights))
            # ... or just the sum ...
            final_weighted_map[wordd] = sum(weights)
        else:
            final_weighted_map[wordd] = weights[0]
    return final_weighted_map


def weighted_bag_for_sentence(sentence: str or Iterable[str], cache_synset_and_bag=None, options=None) \
        -> FilteredWeightedWordsInSentence:
    """
    :param sentence:  an English sentence as a string, or a set of words as strings
    :param cache_synset_and_bag:  an instance of CacheSynsets or None, used to retrieve
    synset instances
    :param options:  an instance of SynsetInfoExtractionOptions, used to define
    what kind of information should be retrieved to build the map
    :return:  an instance of FilteredWeightedWordsInSentence
    """
    if cache_synset_and_bag is None:
        cache_synset_and_bag = CacheSynsets()
    if options is None:
        options = SynsetInfoExtractionOptions()
    h = FilteredWeightedWordsInSentence(sentence)

    all_weighted_maps = [
        (w, weighted_bag_for_word(w, cache_synset_and_bag=cache_synset_and_bag, options=options)) for
        w in h.filtered_words]

    h.word_to_weight_mapping = merge_weighted_bags(t[1] for t in all_weighted_maps)
    for wm_tupla in all_weighted_maps:  # per ogni dizionario (ergo, per ogni parola non-stop nella frase) ..
        wm = wm_tupla[1]
        if wm:
            h.mapping_word_to_its_weighted_bag[wm_tupla[0]] = wm

    return h
