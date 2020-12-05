import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
import re
import spacy
from typing import Dict

nlp = spacy.load("en_core_web_sm")
regex_punctuation = r'[^\w\s]'
only_letters = re.compile('[A-z]+$')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def jaccard_similarity(sent1, sent2):
    len_sent1 = len(sent1)
    len_sent2 = len(sent2)
    if len_sent1 == 0 or len_sent2 == 0:
        return 0
    intCard = len(sent1.intersection(sent2))
    union_card = (len_sent1 + len_sent2) - (intCard)
    return intCard / union_card


def overlap_similarity(sent1, sent2):
    len_sent1 = len(sent1)
    len_sent2 = len(sent2)
    if len_sent1 == 0 or len_sent2 == 0:
        return 1 if (len_sent1 == 0 and len_sent2 == 0) else 0
    intCard = len(sent1.intersection(sent2))
    return intCard / min(len_sent1, len_sent2)


def compute_sim(sent1, sent2, usingOverlapSimilarity=True):
    if usingOverlapSimilarity:
        return overlap_similarity(sent1, sent2)
    else:
        return jaccard_similarity(sent1, sent2)


def weighted_similarity(weighted_bag_1: Dict[str, int], weighted_bag_2: Dict[str, int]) -> int:
    len_b1 = len(weighted_bag_1)
    len_b2 = len(weighted_bag_2)
    if len_b1 == 0 or len_b2 == 0:
        return 0
    if len_b2 < len_b1:
        return weighted_similarity(weighted_bag_2, weighted_bag_1)
    summ = 0
    for wo, we in weighted_bag_1.items():
        if wo in weighted_bag_2:
            summ += we + weighted_bag_2[wo]
    return summ


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class SynsetToBagOptions(object):
    def __init__(self, bag=None, useExamples=True, useLemmas=True):
        self.bag = bag
        self.useExamples = useExamples
        self.useLemmas = useLemmas


#
#
#


def first_index_of(stri, a_char):
    i = 0
    le = len(stri)
    if le == 0:
        return -1
    not_found = True
    while not_found and i < le:
        not_found = stri[i] != a_char
        if not_found:
            i += 1
    return -1 if not_found else i


def filter_and_lemmatize_words_in(text):
    no_punct = re.sub(regex_punctuation, '', text.lower())
    words = no_punct.split()
    words = list(filter(lambda x: x not in stop_words, words))
    return set([lemmatizer.lemmatize(w) for w in words])


def preprocessing(text):
    # return filter_and_lemmatize_words_in(text)
    tokens = nlp(text)
    res = set()
    for t in tokens:
        if t.pos_ != "PUNCT" and not t.is_stop:
            tt = t.text if t.pos_ == "PRON" else t.lemma_
            if only_letters.fullmatch(tt):
                res.add(tt)
    return res


'''
def synsetToBagOfWords(synset, options=None):
    """
	First version of a bag-of-word generator
	"""
    # the option is intentionally default
    if options == None:
        options = SynsetToBagOptions()
    if options.bag == None:
        options.bag = set()
    texts_to_process = None
    if options.useLemmas:
        texts_to_process = collections.deque([l.name() for l in synset.lemmas()])
    else:
        texts_to_process = collections.deque()
    texts_to_process.append(synset.definition())
    if options.useExamples:
        for e in synset.examples():
            texts_to_process.append(e)
    for t in texts_to_process:
        lemm = preprocessing(t)
        for l in lemm:
            if len(l) > 0:
                options.bag.add(l)
    return options.bag
'''
