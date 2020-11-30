import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from "utils.py" import *
#from "functions.py" import *

#sys.path.append(".\\..\\utils")
import sys
sys.path.append(".")
import utils
import functions



root_synset = wn.synsets('entity')
if isinstance(root_synset, list):
	root_synset = root_synset[0]
bag_root_synset = functions.synsetToBagOfWords(root_synset)

class CacheSynsetsBag(object):
	def __init__(self):
		self.cache_cache()


	def clear_cache(self):
		self.cache_synsets = { 'entity': root_synset }
		self.cache_synsets_bag = { 'entity': bag_root_synset }


	def get_synsets(self, word_text, cache_synsets_by_name = None):
		if cache_synsets_by_name is None:
			cache_synsets_by_name = self.cache_synsets
		if word_text in cache_synsets:
			return cache_synsets_by_name[word_text]
		s = wn.synsets(word_text)
		cache_synsets_by_name[word_text] = s
		return s

	def get_synset_bag(self, syns, cache_synsets_bag_by_name = None):
		if cache_synsets_bag_by_name is None:
			cache_synsets_bag_by_name = self.cache_synsets_bag
		name = syns.name()
		if name in cache_synsets_bag_by_name:
			return cache_synsets_bag_by_name[name]
		b = functions.synsetToBagOfWords(syns)
		cache_synsets_bag_by_name[name] = b
		return b

print("BAG OF entity:")
print(bag_root_synset)