from collections import Iterable

from nltk.corpus import wordnet as wn

# import sys
# sys.path.append("../../../aaa/utilities")
from notebooks.utilities.functions import SynsetToBagOptions  # synsetToBagOfWords

# import functions


root_synset = wn.synsets('entity')
if isinstance(root_synset, list):
    root_synset = root_synset[0]


# bag_root_synset = functions.synsetToBagOfWords(root_synset)
# bag_root_synset = synsetToBagOfWords(root_synset)


class CacheSynsetsBag(object):
    """
    Classe che realizza una "cache" di synset (per ridurre l'utilizzo della rete
    ed accellerare il codice al prezzo di un po' di RAM).
    """

    def __init__(self):
        self.cache_synsets = None
        # self.cache_synsets_bag = None
        self.clear_cache()

    def clear_cache(self):
        self.cache_synsets = {'entity': root_synset}
        # self.cache_synsets_bag = {'entity': bag_root_synset}

    def get_synsets(self, word_text, cache_synsets_by_name=None):
        if cache_synsets_by_name is None:
            cache_synsets_by_name = self.cache_synsets
        if word_text in cache_synsets_by_name:  # se e' in cache ...
            s = cache_synsets_by_name[word_text]  # ricicla
            if not isinstance(s, Iterable):
                s = [s]
            return s
        s = wn.synsets(word_text)  # altrimenti, recuperalo ..
        cache_synsets_by_name[word_text] = s  # e salvalo in cache
        if not isinstance(s, Iterable):
            s = [s]
        return s


'''
    def get_synset_bag(self, syns, cache_synsets_bag_by_name=None, options=None):
        """
        DEPRECAED
        :param syns: 
        :param cache_synsets_bag_by_name: 
        :param options: 
        :return: 
        """
        if cache_synsets_bag_by_name is None:
            cache_synsets_bag_by_name = self.cache_synsets_bag
        name = syns.name()
        if name in cache_synsets_bag_by_name:
            return cache_synsets_bag_by_name[name]
        if options is None:
            # options = functions.SynsetToBagOptions(bag=set())
            # b = functions.synsetToBagOfWords(syns, options = options)
            options = SynsetToBagOptions(bag=set())
        b = synsetToBagOfWords(syns, options=options)
        cache_synsets_bag_by_name[name] = b
        return b

    def get_extended_bag_for_words(self, word, options=None):
        """
        DEPRECAED
        :param word: 
        :param options: 
        :return: 
        """
        synsets = self.get_synsets(word)
        if options == None:
            # options = functions.SynsetToBagOptions(bag=set())
            options = SynsetToBagOptions(bag=set())
        prevUseLemmas = options.useLemmas
        prevUseExamples = options.useExamples
        for synset in synsets:
            self.get_synset_bag(synset, cache_synsets_bag_by_name=None, options=options)
            options.useExamples = False  # davvero useExamples=False?
            options.useLemmas = False  # davvero useLemmas=False?
            for coll in [synset.hypernyms(), synset.hyponyms()]:
                for s in coll:
                    self.get_synset_bag(s, cache_synsets_bag_by_name=None, options=options)
            options.useLemmas = prevUseLemmas
            options.useExamples = prevUseExamples
        return options.bag
'''
