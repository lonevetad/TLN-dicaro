# Program to measure the similarity between
# two sentences using cosine similarity.
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functions import *

# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
X ="The Northern Lights are the visible result of solar particles entering the earth's magnetic field and ionizing high in the atmosphere."
Y ="Their intensity depends on the activity of the sun and the acceleration speed of these particles."
Z = "They appear as dancing lights high in the sky and vary in color. The lights usually appear green, but occasionally also purple, red, pink, orange, and blue."
A = "Their colors depend on the elements being ionized."


print("Lesk")
print(lesk(X.split(), 'northern'))
print("Post")


# tokenization
X_list = word_tokenize(X)
Y_list = word_tokenize(Y)

# sw contains the list of stopwords
sw = stopwords.words('english')
l1 =[];l2 =[]

# remove stop words from the string
X_set = {w for w in X_list if not w in sw}
Y_set = {w for w in Y_list if not w in sw}

# form a set containing keywords of both strings
rvector = X_set.union(Y_set)
for w in rvector:
	if w in X_set: l1.append(1) # create a vector
	else: l1.append(0)
	if w in Y_set: l2.append(1)
	else: l2.append(0)
c = 0

# cosine formula
for i in range(len(rvector)):
		c+= l1[i]*l2[i]
cosine = c / float((sum(l1)*sum(l2))**0.5)      #radice quadrata
#print("similarity: ", cosine)

#print(len(wn.synsets("Lights")))
#for i in range(0, len(wn.synsets("Lights"))):
	#print(wn.synsets("Lights")[i].definition())
#print(word.hypernyms()[0].definition())
X = preprocessing(X)
syn = []

for w in X.split():
	#print("Parola", w)
	syn.append(wn.synsets(w))

#print(syn)
for i in range(0, len(syn)):
	for j in range(0, len(syn[i])):
		print(syn[i][j], syn[i][j].definition(), syn[i][j].hypernyms(), syn[i][j].hyponyms())

