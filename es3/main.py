from collections import Counter
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from functs import *
from nltk.corpus import wordnet as wn

if __name__ == '__main__':

    menu()
    file_name = ''
    verb = int(input())
    if verb != 1 and verb != 2:
        print("Wrong value. Insert 1 or 2")
    elif verb == 1:
        file_name = "Corpus\\tobuild_corpus.txt"
    else:
        file_name = "Corpus\\tocook_corpus.txt"

    frasi = read_sentences(file_name)
    #Lettura Corpus OK
    sents = preprocessing(frasi)

    subj_obj = []
    disambigued = []
    super_senses = []
    slot1_ss = []
    slot2_ss = []
    index = 0

    for i in range(0, len(sents)):
        index += 1
        subj_obj.append(parse_find_subj_obj(sents[i], index))          #Riempiamo gli slot con i filler
        temp = sents[i]
        temp_subj = subj_obj[i][0]
        temp_obj = subj_obj[i][1]
        sent = temp.split()
        s, o = wsd(sent, temp_subj, temp_obj)
        if s is not None and o is not None:
            so = (s, o)
            disambigued.append(so)     #Soggetti e oggetti non nulli
            sup = super_sense(so[0], so[1])
            super_senses.append(sup)      #Supersensi OK
            if sup[0] is not None:
                slot1_ss.append(sup[0].split('.')[1])    #Supersensi dei soggetti
            else:
                slot1_ss.append(sup[0])

            if sup[1] is not None:
                slot2_ss.append(sup[1].split('.')[1])    #Supersensi degli oggetti
            else:
                slot2_ss.append(sup[1])
            tot_slot1 = len(slot1_ss)
            tot_slot2 = len(slot2_ss)
            tot_ss = len(super_senses)


    #Risultati finali
    ss_count = Counter(super_senses)

    occ = ss_count.most_common(10)
    for pair, occ in occ:
        print("Semantic Type:  ", pair, round((occ/tot_ss)*100, 2), " %")
    #print("Occorrenze delle combinazioni dei Filler negli slot 1 e 2 \n", ss_count.most_common(10))
    #print("Totale supesensi ", tot_ss)                              #OK

    slot1_count = Counter(slot1_ss)

    occ_slot1 = slot1_count.most_common(10)
    for pair, occ in occ_slot1:
        print("Filler slot 1:  ", pair, round((occ/tot_slot1) * 100, 2), " %")
    #print("Occorrenze supersensi dei soggetti\n", slot1_count.most_common(10))
    #print("Totale slot 1 ", tot_slot1)

    slot2_count = Counter(slot2_ss)

    occ_slot2 = slot2_count.most_common(10)
    for pair, occ in occ_slot2:
        print("Filler slot 2:  ", pair, round((occ/tot_slot2) * 100, 2), " %")
    #print("Occorrenze supersensi degli oggetti\n", slot2_count.most_common(10))
    #print("Totale slot 2 ", tot_slot2)

    slots = [slot1_ss, slot2_ss]
    valenza = 2

    for i in range(0, valenza):
        generate_word_cloud(slots[i])
        print("Word Cloud", i+1, "generata")




