import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
import re


regex_punctuation = r'[^\w\s]'
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------



def jaccard_similarity(sent1, sent2):
	len_sent1 = len(sent1)
	len_sent2 = len(sent2)
	if len_sent1== 0 or len_sent2 == 0:
		return 0
	intCard = len(sent1.intersection(sent2))
	union_card = (len_sent1 + len_sent2) - (intCard)
	return intCard/union_card
	
def overlap_similarity(sent1, sent2):
	len_sent1 = len(sent1)
	len_sent2 = len(sent2)
	if len_sent1 == 0 or len_sent2 == 0:
		if len_sent1 == 0 and len_sent2 == 0:
			return 1
		else:
			return 0
	intCard = len(sent1.intersection(sent2))
	return intCard / min(len_sent1, len_sent2)

def compute_sim(sent1, sent2, usingOverlapSimilarity = True):
	if usingOverlapSimilarity:
		return overlap_similarity(sent1, sent2)
	else:
		return jaccard_similarity(sent1, sent2)



# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------



def filter_and_lemmatize_words_in(text):
	no_punct = re.sub(regex_punctuation, '', text.lower())
	words = no_punct.split()
	words = list(filter(lambda x: x not in stop_words, words))
	return set([lemmatizer.lemmatize(w) for w in words])
	
def synsetToBagOfWords(s, useExamples = True):
	bag = set()
	texts_to_process = collections.deque([l.name() for l in s.lemmas()])
	texts_to_process.append(s.definition())
	if useExamples:
		for e in s.examples():
			texts_to_process.append(e)
	for t in texts_to_process:
		lemm = filter_and_lemmatize_words_in(t)
		for l in lemm:
			bag.add(l)
	return bag
