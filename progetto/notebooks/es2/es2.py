import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

# regex_punctuation = r'[^\w\s]'
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
from notebooks.utilities.cacheVarie import CacheSynsetsBag
from notebooks.utilities.functions import synsetToBagOfWords, compute_sim, filter_and_lemmatize_words_in
from notebooks.utilities import utils # split_csv_row_string, deep_array_printer


def newCache():
    return CacheSynsetsBag()


cache = newCache()


# ------------
# ------------
# ------------------ utilities ------------------
# ------------
# ------------


def read_csv():
    with open('Esperimento content-to-form - Foglio1.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        can_use = False  # la prima riga deve essere salata
        cols = None
        i = 0
        for row in csv_reader:
            if len(row) > 0:
                if can_use:
                    row_string = row[0]
                    i = 0
                    cols_in_this_row = utils.split_csv_row_string(row_string)
                    length = len(cols_in_this_row) - 1
                    while i < length:
                        cols[i].append(cols_in_this_row[i + 1])  # because the first column is reserved for indexes
                        i += 1
                else:
                    cols_names = row[0].split(",")
                    cols = [[] for i in range(len(cols_names) - 1)]  # because the first column is reserved for indexes
                    can_use = True
            i += 1
        return cols
    return None


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


def colum_csv_to_bag_of_words(column):
    bag = set()
    for c in column:
        if len(c) > 0:
            bags_of_c = filter_and_lemmatize_words_in(c)
            for w in bags_of_c:
                bag.add(w)
    return bag


# ------------
# ------------
# ------------------ main ------------------
# ------------
# ------------


def main():
    # print(get_synsets("play"))
    print("start :D\n\n")

    cols = read_csv()
    # print("\n\nCOLONNEEEEEE")
    # deep_array_printer(cols)

    similarities_to_test = 2

    while similarities_to_test > 0:
        similarities_to_test -= 1
        useOverlap = similarities_to_test > 0

        print("\n\n\n\n")
        for i in range(0, 5):
            print("-------------------------------------------------")
        print(" testing similarity", "overlap" if useOverlap else "jaccard")

        i = 0
        for column in cols:
            print("\n\n\n elaborating th given column:", i)
            #utils.deep_array_printer(column)
            bag_for_col = colum_csv_to_bag_of_words(column)
            best_synset, best_simil = searchBestApproximatingSynset(bag_for_col, addsAntinomies=True, usingOverlapSimilarity=useOverlap)
            print("found: ", best_synset, ", with similarity of:", best_simil)
            i += 1

    # beware of entries in columns having len(field) == 0 ....

    print("\n\n\n fine")


main()
