import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
import re
import spacy

nlp = spacy.load("en_core_web_sm")
regex_punctuation = r'[^\w\s]'
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class SynsetToBagOptions(object):
    def __init__(self, bag=None, useExamples=True, useLemmas=True):
        self.bag = bag
        self.useExamples = useExamples
        self.useLemmas = useLemmas


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
            res.add(t.text if t.pos_ == "PRON" else t.lemma_)
    return res


def synsetToBagOfWords(synset, options=None):
    '''
	First version of a bag-of-word generator
	'''
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
