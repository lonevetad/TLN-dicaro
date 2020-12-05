from typing import Dict, Iterable

from nltk.corpus import wordnet as wn
import collections

from notebooks.utilities import functions
from notebooks.utilities.cacheVarie import CacheSynsetsBag
# from notebooks.utilities import utils  # split_csv_row_string, deep_array_printer
import notebooks.utilities.synsetInfoExtraction as sye


def newCache():
    return CacheSynsetsBag()


cache = newCache()

# ------------
# ------------
# ------------------ utilities ------------------
# ------------
# ------------


# ------------
# ------------
# ------------------ major tools ------------------
# ------------
# ------------


root_synset = wn.synsets('entity')[0]  # top-most synset (only one exists)
# bag_root_synset = synsetToBagOfWords(root_synset)
cache_synsets = {'entity': root_synset}
# cache_synsets_bag = {'entity': bag_root_synset}
print(root_synset)
print("bags for entity:")


# print(bag_root_synset)


def searchBestApproximatingSynset(bagOfWordFromDefinitions: Iterable[str], addsAntinomies=True,
                                  usingOverlapSimilarity=True,
                                  cacheSynsetsBagByName=None):
    # words_info_cache = {}  # words_seen = set()
    frontier = collections.deque()

    cache_synset_and_bag = CacheSynsetsBag()
    # word_to_synset_map_cache = cache_synset_and_bag.cache_synsets
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

    - cache_synset_and_bag: una istanza di CacheSynsetsBag
            
            
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
    all_antinomies = []
    for w in bagOfWordFromDefinitions:
        # slim_genus[w] = genus_fat[w]
        syns = cache_synset_and_bag.get_synsets(w)
        for syn in syns:
            for lem in syn.lemmas():
                for a in lem.antonyms():
                    all_antinomies.append(a.synset())
    differentia = sye.merge_weighted_bags(sye.weighted_bag_for_word(s.name()) for s in all_antinomies)

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
            if curr_synset.name() not in words_seen:
                curr_bag = get_weighted_def(curr_synset.name())
                if curr_bag is not None:
                    curr_simil = functions.weighted_similarity(genus_fat, curr_bag)
                    if curr_simil > best_simil:
                        can_recurse = False
                        if best_synset is None:
                            best_synset = curr_synset
                            best_simil = curr_simil - functions.weighted_similarity(differentia,
                                                                                    curr_bag)  # no matter if it's negatives
                            can_recurse = True
                        else:
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


def searchBestApproximatingSynset_V1(bagOrSynset, addsAntinomies=True, usingOverlapSimilarity=True,
                                     cacheSynsetsBagByName=None):
    if cacheSynsetsBagByName is None:
        cacheSynsetsBagByName = newCache()
    if not isinstance(bagOrSynset, set):
        # assumption: it's a WordNet's synset:
        bagOrSynset = cache.get_synset_bag(bagOrSynset, cacheSynsetsBagByName)  # QUESTA FUNZIONE NON ESISTE PIU
    '''
    "macchia d'olio":
    per ogni parola nella bag:
    -) cercare il relativo synset
    -) calcolarne la bag
    -) overlap con la bag
    -) se l'overlap è migliore -> aggiornare migliore
    -) ricorsione
    '''
    words_seen = set()
    frontier = collections.deque()
    for word in bagOrSynset:
        words_seen.add(word)
        frontier.append(word)
    best_synset = None
    best_simil = 0.0
    # best_bag = None
    # original_words_bag_remaining = 0
    len_original_words_bag = len(bagOrSynset)
    while len(frontier) > 0:
        current_word = frontier.popleft()
        synsets_current_word = cache.get_synsets(current_word)
        if not (isinstance(synsets_current_word, list)):
            synsets_current_word = [synsets_current_word]

        # RACCOLTA
        # gather all usefull synsets from a given words:
        # 1) is synsets
        # 2) their sister terms
        # 3) synonyms
        useful_synsets_by_name = {}
        for s in synsets_current_word:
            s_name = s.name()

            # inizio precursore (== versione brutta) di synsetInfoExtraction.weighted_bag_words_from_word
            useful_synsets_by_name[s_name] = s
            lemmas = s.lemmas()
            matrix_synsets = [
                [l.synset() for l in lemmas],  # synonyms
                s.hypernyms()
            ]
            if addsAntinomies:  # L'INCRIMINATO PEZZO DI CODICE
                for l in lemmas:
                    ant = l.antonyms()
                    if ant:
                        matrix_synsets.append([a.synset() for a in ant])
            for synsets_collection in matrix_synsets:
                for syn in synsets_collection:
                    if s_name != syn.name():
                        useful_synsets_by_name[syn.name()] = syn

            # fine precursore (== versione brutta) di synsetInfoExtraction.weighted_bag_words_from_word

        # RICERCA DEL SYNSET MIGLIORE
        for synsName, current_node in useful_synsets_by_name.items():
            current_bag = cache.get_synset_bag(current_node)  # QUESTA FUNZIONE NON ESISTE PIU
            current_simil = functions.compute_sim(bagOrSynset, current_bag,
                                                  usingOverlapSimilarity=usingOverlapSimilarity)

            '''
            IPOTESI DI MIGLIORAMENTO DEL CODICE:
            decrementare "current_simil" di un ammotare pari alla similarità tra il
            "bag" delle antinomie con il "bag delle definizioni [colonna]" (ossia di "bagOrSynset")
            '''

            if current_simil > best_simil:  # SE E' MIGLIORE, LO SALVO E RICORSIONE
                # print("\t-> updating: new node ", current_node.name(), ", with simil:", current_simil, "and bag: ", current_bag)
                best_synset = current_node
                # best_bag = current_bag
                best_simil = current_simil
                # if original_words_bag_remaining >= len_original_words_bag:
                for word in current_bag:
                    if word not in words_seen:
                        words_seen.add(word)
                        frontier.append(word)  # RICORSIONE
        useful_synsets_by_name = None
    return best_synset, best_simil

# ------------
# ------------
# ------------------ main ------------------
# ------------
# ------------
