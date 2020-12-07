from collections import Iterable
from nltk.corpus import wordnet as wn

root_synset = wn.synsets('entity')
if isinstance(root_synset, list):
    root_synset = root_synset[0]


class CacheSynsets(object):
    """
    Classe che realizza una "cache" di synset (per ridurre l'utilizzo della rete
    ed accellerare il codice al prezzo di un po' di RAM).
    """

    def __init__(self):
        self.cache_synsets = None
        self.clear_cache()

    def clear_cache(self):
        self.cache_synsets = {'entity': root_synset}

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
