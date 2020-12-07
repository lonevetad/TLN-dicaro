import nltk
import string
from nltk import word_tokenize
import csv
import pandas
import numpy

from V1.functions import *

if __name__ == "__main__":

    definizioni_concreto_generico = []
    definizioni_concreto_specifico = []
    definizioni_astratto_generico = []
    definizioni_astratto_specifico = []

    lettore = read_load_csv()

    definizioni_concreto_generico = preprocessing(lettore[0])
    definizioni_concreto_specifico = preprocessing(lettore[1])
    definizioni_astratto_generico = preprocessing(lettore[2])
    definizioni_astratto_specifico = preprocessing(lettore[3])

    similarita_concreto_generico = []
    similarita_concreto_specifico = []
    similarita_astratto_generico = []
    similarita_astratto_specifico = []

    cont = 0
    for x in range(0, 19):
        for y in range(x + 1, 19):
            cont += 1
            similarita_concreto_generico.append(compute_sim(definizioni_concreto_generico[x],
                                                            definizioni_concreto_generico[y]))
            similarita_concreto_specifico.append(compute_sim(definizioni_concreto_specifico[x],
                                                             definizioni_concreto_specifico[y]))
            similarita_astratto_generico.append(compute_sim(definizioni_astratto_generico[x],
                                                            definizioni_astratto_generico[y]))
            similarita_astratto_specifico.append(compute_sim(definizioni_astratto_specifico[x],
                                                             definizioni_astratto_specifico[y]))

    #print(cont)
    filtered = list(filter(lambda sim: sim > 0.5, similarita_concreto_generico))
    print("CG % --> ", (len(filtered) / cont)*100)

    filtered = list(filter(lambda sim: sim > 0.5, similarita_concreto_specifico))
    print("CS % --> ", (len(filtered) / cont) * 100)

    filtered = list(filter(lambda sim: sim > 0.5, similarita_astratto_generico))
    print("AG % --> ", (len(filtered) / cont) * 100)

    filtered = list(filter(lambda sim: sim > 0.5, similarita_astratto_specifico))
    print("AS % --> ", (len(filtered) / cont) * 100)



