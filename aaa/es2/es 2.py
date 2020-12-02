import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

import sys
sys.path.append(".\\..\\utilitiess")
import utils
import functions
from cacheVarie import *


#regex_punctuation = r'[^\w\s]'
#lemmatizer = WordNetLemmatizer()
#stop_words = set(stopwords.words('english'))

def newCache():
	return cacheVarie.CacheSynsetsBag()

cache = newCache()


#------------
#------------
# ------------------ utilities ------------------
#------------
#------------



def read_csv():
	with open('Esperimento content-to-form - Foglio1.csv', 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter='\n')
		can_use = False #la prima riga deve essere salata
		cols = None
		i = 0
		for row in csv_reader:
			if len(row) > 0:
				if can_use:
					row_string = row[0]
					i = 0
					cols_in_this_row = utils.split_csv_row_string(row_string)
					length = len(cols_in_this_row) -1
					while i < length:
						cols[i].append(cols_in_this_row[i + 1]) #because the first column is reserved for indexes
						i += 1
				else:
					cols_names = row[0].split(",")
					cols = [ [] for i in range(len(cols_names)-1) ] #because the first column is reserved for indexes
					can_use = True
			#else:
			#	print("row", i, "is empty")
			i += 1
		return cols
	return None






#------------
#------------
# ------------------ major tools ------------------
#------------
#------------


root_synset = wn.synsets('entity')[0] # top-most synset (only one exists) 
bag_root_synset = functions.synsetToBagOfWords(root_synset)
cache_synsets = { 'entity': root_synset }
cache_synsets_bag = { 'entity': bag_root_synset }
print(root_synset)
print("bags for entity:")
print(bag_root_synset)



 



def searchBestApproximatingSynset(bagOrSynset, addsAntinomies = True, usingOverlapSimilarity = True):
	if not isinstance(bagOrSynset, set):
		#assumption: it's a WordNet's synset:
		bagOrSynset = cache.get_synset_bag(bagOrSynset, cacheSynsetsBagByName)
	print("calculated bag:\n\t", bagOrSynset)
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
	#best_bag = None
	#original_words_bag_remaining = 0
	len_original_words_bag = len(bagOrSynset)
	while len(frontier) > 0:
		current_word = frontier.popleft()
		synsets_current_word = cache.get_synsets(current_word)
		if not (isinstance(synsets_current_word, list)):
			synsets_current_word = [synsets_current_word]
		#gather all usefull synsets from a given words:
		# 1) is synsets
		# 2) their sister terms
		# 3) synonyms
		useful_synsets_by_name = {}
		for s in synsets_current_word:
			s_name = s.name()
			useful_synsets_by_name[s_name] = s
			lemmas = s.lemmas()
			matrix_synsets = [
				[l.synset() for l in lemmas], #synonyms
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
						#print("____syn.name() ->", syn.name())
						useful_synsets_by_name[syn.name()] = syn
		 
		for synsName, current_node in useful_synsets_by_name.items():
		#for current_node in synsets_current_word:
			current_bag = cache.get_synset_bag(current_node)
			current_simil = functions.compute_sim(bagOrSynset, current_bag, usingOverlapSimilarity = usingOverlapSimilarity)
			if current_simil > best_simil :
				print("\t-> updating: new node ", current_node.name(), ", with simil:", current_simil, "and bag: ", current_bag)
				best_synset = current_node
				#best_bag = current_bag
				best_simil = current_simil
		#		if original_words_bag_remaining >= len_original_words_bag:
				for word in current_bag:
					if word not in words_seen:
						words_seen.add(word)
						frontier.append(word)
		useful_synsets_by_name = None
		#	if original_words_bag_remaining < len_original_words_bag:
		#		for word in current_bag:
		#			if word not in words_seen:
		#				words_seen.add(word)
		#				frontier.append(word)
		#if original_words_bag_remaining < len_original_words_bag:
		#	original_words_bag_remaining += 1
	return best_synset, best_simil


def searchBestApproximatingSynset_V2(bagOrSynset):
	if not isinstance(bagOrSynset, set):
		#assumption: it's a WordNet's synset:
		bagOrSynset = cache.get_synset_bag(bagOrSynset)
		#print("calculated bag:\n\t", bagOrSynset)
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
	best_simil = -1.0
	best_bag = None
	while len(frontier) > 0:
		current_word = frontier.popleft()
		synsets_current_word = cache.get_synsets(current_word)
		for current_node in synsets_current_word:
			current_bag = cache.get_synset_bag(current_node)
			current_simil = 0.0 if best_synset is None else functions.compute_sim(best_bag, current_bag, usingOverlapSimilarity = usingOverlapSimilarity)
			if current_simil > best_simil :
				print("\t -> updating: new node ", current_node.name(), ", with simil:", current_simil.name(), "and bag: ", current_bag)
				best_synset = current_node
				best_bag = current_bag
				best_simil = current_simil
				for word in current_bag:
					if word not in words_seen:
						words_seen.add(word)
						frontier.append(word)
	return best_synset, best_simil




def searchBestApproximatingSynset_V1(bagOrSynset, cacheSynsetsBagByName = None):
	if cacheSynsetsBagByName == None:
		cacheSynsetsBagByName = {}
	if not isinstance(bagOrSynset, set):
		#assumption: it's a WordNet's synset:
		bagOrSynset = cache.get_synset_bag(bagOrSynset, cacheSynsetsBagByName)
		#print("calculated bag:\n\t", bagOrSynset)
	best_synset = root_synset
	synonyms_already_seen = { best_synset.name(): best_synset }
	best_bag = bag_root_synset
	best_simil = functions.compute_sim(bagOrSynset, best_bag, usingOverlapSimilarity = usingOverlapSimilarity)
	frontier = collections.deque(root_synset.hyponyms())
	#print("\n\n original best simil: ", best_simil)
	while len(frontier) > 0:
		current_node = frontier.popleft()
		if current_node.name() not in synonyms_already_seen:
			#print("current node:", current_node.name())
			synonyms_already_seen[current_node.name()] = current_node
			current_bag = cache.get_synset_bag(current_node, cacheSynsetsBagByName)
			current_simil = functions.compute_sim(best_bag, current_bag, usingOverlapSimilarity = usingOverlapSimilarity)
			if current_simil > best_simil :
				print("\t -> updating: new node ", current_node.name(), ", with simil:", current_simil)
				best_synset = current_node
				best_bag = current_bag
				best_simil = current_simil
				# iterate
				for hypo in current_node.hyponyms():
					#print("\t- hypo:", hypo)
					frontier.append(hypo)
			#else: # prune it and ALL descendants
			#	print("discarded", current_node.name(), "with simil", current_simil)
			
			#for hypo in current_node.hyponyms():
			#	print("\t- hypo:", hypo)
			#	frontier.append(hypo)
	return best_synset, best_simil


def colum_csv_to_bag_of_words( column ):
	bag = set()
	for c in column:
		if len(c) > 0:
			bags_of_c = functions.filter_and_lemmatize_words_in(c)
			for w in bags_of_c:
				bag.add(w)
	return bag


#------------
#------------
# ------------------ main ------------------
#------------
#------------


def main():
	#print(get_synsets("play"))
	print("start :D\n\n")

	cols = read_csv()
	#print("\n\nCOLONNEEEEEE")
	#deep_array_printer(cols)

	similarities_to_test = 2

	while similarities_to_test > 0:
		similarities_to_test -= 1
		useOverlap = similarities_to_test > 0

		print("\n\n\n\n")
		for i in range(0,5):
			print("-------------------------------------------------")
		print(" testing similarity", "overlap" if useOverlap else "jaccard")

		for column in cols:
			print("\n\n\n elaborating th given column:")
			utils.deep_array_printer(column)		
			bag_for_col = colum_csv_to_bag_of_words(column)
			best_synset, best_simil = searchBestApproximatingSynset(bag_for_col, addsAntinomies = True, usingOverlapSimilarity = useOverlap)
			print("found: ", best_synset, ", with similarity of:", best_simil)

	#beware of entries in columns having len(field) == 0 ....

	print("\n\n\n fine")

main()


'''
should produce:
- tolerance, peace, calm
- greed, avarice, parsimony
- law
- alimento .. food, dietary, edible, grocerie
- radiator, heater, boiler
- mezzo di trasporto .. container, wehicle
- bolt
e poi si confonde malamente a causa di bug, che mischiano nella stessa colonna definizioni di altre cose


'''

'''
syns_to_search = ["dog", "dalmata", "copper"]
for w in syns_to_search:
	print("\n\n\n\n syn for", w, ":", end= "")
	ss = get_synsets(w)
	if len(ss) > 0:
		syn_dog = ss[0]
		print(syn_dog)
		syn_similar_to_dog = searchBestApproximatingSynset(syn_dog, cache_synsets_bag)
		print( "\n\nfound:", syn_similar_to_dog )
	else:
		print("not found")
'''