import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

# regex_punctuation = r'[^\w\s]'
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
from notebooks.utilities.cacheVarie import CacheSynsets
from notebooks.utilities.functions import synsetToBagOfWords, compute_sim, filter_and_lemmatize_words_in
from notebooks.utilities import utils # split_csv_row_string, deep_array_printer


def newCache():
    return CacheSynsets()


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
bag_root_synset = synsetToBagOfWords(root_synset)
cache_synsets = {'entity': root_synset}
cache_synsets_bag = {'entity': bag_root_synset}
print(root_synset)
print("bags for entity:")
print(bag_root_synset)


def searchBestApproximatingSynset(bagOrSynset, addsAntinomies=True, usingOverlapSimilarity=True,
                                  cacheSynsetsBagByName=None):
    if cacheSynsetsBagByName is None:
        cacheSynsetsBagByName = newCache()
    if not isinstance(bagOrSynset, set):
        # assumption: it's a WordNet's synset:
        bagOrSynset = cache.get_synset_bag(bagOrSynset, cacheSynsetsBagByName)
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
        # gather all usefull synsets from a given words:
        # 1) is synsets
        # 2) their sister terms
        # 3) synonyms
        useful_synsets_by_name = {}
        for s in synsets_current_word:
            s_name = s.name()
            useful_synsets_by_name[s_name] = s
            lemmas = s.lemmas()
            matrix_synsets = [
                [l.synset() for l in lemmas],  # synonyms
                s.hypernyms()
            ]
            if addsAntinomies:
                for l in lemmas:
                    ant = l.antonyms()
                    if ant:
                        matrix_synsets.append([a.synset() for a in ant])
            for synsets_collection in matrix_synsets:
                for syn in synsets_collection:
                    if s_name != syn.name():
                        useful_synsets_by_name[syn.name()] = syn

        for synsName, current_node in useful_synsets_by_name.items():
            current_bag = cache.get_synset_bag(current_node)
            current_simil = compute_sim(bagOrSynset, current_bag,
                                        usingOverlapSimilarity=usingOverlapSimilarity)
            if current_simil > best_simil:
                # print("\t-> updating: new node ", current_node.name(), ", with simil:", current_simil, "and bag: ", current_bag)
                best_synset = current_node
                # best_bag = current_bag
                best_simil = current_simil
                # if original_words_bag_remaining >= len_original_words_bag:
                for word in current_bag:
                    if word not in words_seen:
                        words_seen.add(word)
                        frontier.append(word)
        useful_synsets_by_name = None
    return best_synset, best_simil



# ------------
# ------------
# ------------------ main ------------------
# ------------
# ------------

