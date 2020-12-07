from typing import Dict, Iterable

from nltk.corpus import wordnet as wn
import collections

from notebooks.utilities import functions
from notebooks.utilities.cacheVarie import CacheSynsets
import notebooks.utilities.synsetInfoExtraction as sye


def newCache():
    return CacheSynsets()


cache = newCache()


# ------------
# ------------
# ------------------ major tools ------------------
# ------------
# ------------


root_synset = wn.synsets('entity')[0]  # top-most synset (only one exists)
cache_synsets = {'entity': root_synset}


def searchBestApproximatingSynset(bagOfWordFromDefinitions: Iterable[str], cacheSynsetsBagByName=None,
                                  shouldConsiderAntinomyes = True):
    # words_info_cache = {}  # words_seen = set()
    frontier = collections.deque()

    cache_synset_and_bag = CacheSynsets() if cacheSynsetsBagByName is None else cacheSynsetsBagByName
    weighted_context_info = sye.weighted_bag_for_sentence(bagOfWordFromDefinitions,
                                                          cache_synset_and_bag=cache_synset_and_bag)

    '''
    COSA HO FIN'ORA:

    - il "bagOfWordFromDefinitions" è un insieme di parole (string) ottenute dalla
        colonna di definizioni estratte dal file csv

    - weighted_context_info: istanza di "FilteredWeightedWordsInSentence" che contiene
        - word_to_weight_mapping:
            una mappa che associa ad ogni parola (parola di "bagOfWordFromDefinitions")
            un peso, indicante quanto è "utile" nel definire il concetto (quello da estrarre con il "contet-to-form")
        - filtered_words:
            alias di "bagOfWordFromDefinitions"
        - mapping_word_to_its_weighted_bag:
            struttura complessa, è una mappa che associa ad ogni parola una map<string, int>, ossia
            la sua "definizione pesata" (che è il risultato della funzione "weighted_bag_words_from_word" invocata internamente da "get_weighted_word_map_for_sentence"),
            la quale è a sua volta una mappatura (appunto) parola -> peso, ossia associa ad ogni parola della "definizione"
            una "utilita'" nella definizione stessa.

    - cache_synset_and_bag: una istanza di CacheSynsets
            
            
    in sintesi
    weighted_context_info.word_to_weight_mapping e' il GENUS
    (circa)
    '''
    genus_fat: Dict[str, int] = weighted_context_info.word_to_weight_mapping
    # slim_genus = {}

    options_for_antinomies = sye.SynsetInfoExtractionOptions([
        ("name", sye.SYNSET_INFORMATIONS_WEIGHTS["name"]),
        ("definition", sye.SYNSET_INFORMATIONS_WEIGHTS["definition"])
    ])
    differentia = None
    if shouldConsiderAntinomyes :
        all_antinomies = []
        for w in bagOfWordFromDefinitions:
            # slim_genus[w] = genus_fat[w]
            syns = cache_synset_and_bag.get_synsets(w)
            for syn in syns:
                for lem in syn.lemmas():
                    for a in lem.antonyms():
                        all_antinomies.append(a.synset())
        differentia = sye.merge_weighted_bags(sye.weighted_bag_for_word(s.name(), options=options_for_antinomies) for s in all_antinomies)

    def get_weighted_def(woo: str) -> Dict[str, int] or None:
        if woo in weighted_context_info.mapping_word_to_its_weighted_bag:
            return weighted_context_info.mapping_word_to_its_weighted_bag[woo]
        else:
            return sye.weighted_bag_for_word(woo, cache_synset_and_bag=cache_synset_and_bag)

    '''
        QUINDI COS'HO?
        1) "
    '''

    '''
    idea: costruisco un mega-contesto formato dall'insieme di "bag/info" delle parole
    della frase (cache_synset_and_bag), poi scansiono i synset delle parole delle frasi
    (contenute in bagOrSynset e, quindi, anche in frontier) per cercare il migliore
    
    ogni synset è un candidato ad essere il "migliore" ...
    estratta la sua "definizione" (usando get_weighted_def) calcolo il suo score:
    weighted overlap della definizione del dato synset con il contesto di TUTTA la frase
    MA sottratto al weighted overlap della definizione del dato synset con la definizione delle antinomie dei synsets della frase
    
    scelgo poi il synset con lo score migliore
    '''

    # inizializzazione ciclo di analisi dei synsets
    best_synset = None
    best_simil = 0.0
    words_seen = set()
    for word in bagOfWordFromDefinitions:
        # words_info_cache[word] = sye.weighted_bag_words_from_word(word)
        if word not in words_seen:
            frontier.append(word)
            words_seen.add(word)

    while len(frontier) > 0:
        current_word = frontier.popleft()
        current_synsets = cache_synset_and_bag.get_synsets(current_word)
        for curr_synset in current_synsets:
            # iniziamo la ricerca del migliore
            if curr_synset.name() not in words_seen: # SVILUPPI FUTURI: escludere i non-nomi (es: "point.v.XY" non va bene, "point.v.WZ" si)
                curr_bag = get_weighted_def(curr_synset.name())
                if curr_bag is not None:
                    curr_simil = functions.weighted_similarity(genus_fat, curr_bag)
                    if curr_simil > best_simil:
                        can_recurse = False
                        if best_synset is None:
                            best_synset = curr_synset
                            best_simil = curr_simil
                            if shouldConsiderAntinomyes:
                                best_simil -= functions.weighted_similarity(differentia, curr_bag)
                            can_recurse = True
                        else:
                            if shouldConsiderAntinomyes:
                                curr_simil -= functions.weighted_similarity(differentia, curr_bag)
                            if curr_simil > best_simil:
                                best_synset = curr_synset
                                best_simil = curr_simil
                                can_recurse = True
                            # else: scartalo
                        if can_recurse:  # RICORSIONE
                            for word, weight in curr_bag.items():
                                if word not in words_seen:
                                    frontier.append(word)
                                    words_seen.add(word)
                # else: scartalo
    return best_synset, best_simil
