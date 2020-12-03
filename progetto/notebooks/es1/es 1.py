import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from notebooks.utilities.functions import compute_sim

'''
from nltk import word_tokenize
from nltk import pos_tag
import string
import pandas
import numpy
'''

regex_punctuation = r'[^\w\s]'
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print("start of all\n\n\n-----\n\n")


#------------


def read_load_csv():
	with open('definitions.csv', 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=';')
		cols = [ [], [], [], [] ]
		can_use = False #la prima riga deve essere salata
		for cat in csv_reader:
			if can_use:
				'''
				In ordine, le colonne sono:
				0) concreto_generico_building
				1) concreto_specifico_molecule
				2) astratto_generico_freedom
				3) astratto_specifico_compassion

				'''
				cols[0].append(cat[0])
				cols[1].append(cat[1])
				cols[2].append(cat[2])
				cols[3].append(cat[3])
			else:
				print("filtering out:")
				print(cat)
				print("\n.............\n")
				can_use = True
		print(cols)
	return cols


#------------


def prepare_words_set(definitions):
	clean_list = [re.sub(regex_punctuation, '', defs.lower()) for defs in definitions]
	return [set(lemmatization(clean_list[i])) for i in range(len(clean_list))]


def lemmatization(sentence):
	words = sentence.split()
	words = list(filter(lambda x: x not in stop_words, words))
	return [lemmatizer.lemmatize(w) for w in words] 


#------------



#------------


#------------


#------------


def main():
	print("start")
	
	lettore = read_load_csv()
	num_columns = len(lettore)
	definizioni = [None]*num_columns
	#pre-processing step
	for i in range(0, num_columns):
		definizioni[i] = prepare_words_set(lettore[i])
		#print("definizioni: ")
		#print(definizioni[i])

	similarita = [[] for i in range(0, num_columns)]
	lel = len(definizioni[0])
	# per ogni coppia ..
	for x in range(0, lel):
		for y in range(0, lel):
			if x is not y: #non confrontiamo gli elementi con se' stessi ...
				for i in range(0, num_columns):
					ddd = definizioni[i]
					similarita[i].append(compute_sim(ddd[x], ddd[y]))

	defs_name = ["concreto-generico", "concreto-specifico", "astratto-generico", "astratto-specifico"]
	filters_and_descr = [
		(lambda sim: sim > 0.5, "restrittivo (maggiore strettamente di 0.5)"),
		(lambda sim: sim >= 0.5, "morbido (maggiore o uguale a 0.5)")
	]
	print("\n\nstart tests:")
	for fd in filters_and_descr:
		print("\n\nUsando un filtro più", fd[1], ", abbiamo:")
		sim_filter = fd[0]
		for i in range(0, num_columns):
			filtered = list(filter(sim_filter, similarita[i]))
			print(defs_name[i], " -> ", (len(filtered)*100.0)/len(similarita[i]),"%" )
	
	print("\n\n.................\nend")


main()