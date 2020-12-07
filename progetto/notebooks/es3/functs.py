import string
import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from wordcloud import WordCloud
import matplotlib.pyplot as plt


nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def read_sentences(file_name):
    sents = []

    with open(file_name,'r',encoding="utf-8") as lines:
        for line in lines :
            if len(line.split('<s>')) > 1:
                sents.append(line.split('<s>')[1].replace('</s>','').strip())
    return sents


def preprocessing(sentence):
    #print("Lowering")
    sentence = [s.lower() for s in sentence]    #OK
    #print("Removing punctuation")
    sentence = [''.join(c for c in s if c not in string.punctuation) for s in sentence]

    return sentence


p_subj = {'subj', 'nsubjpass', 'nsubj'}
p_obj = {'pobj', 'dobj', 'obj', 'iobj'}


def parse_find_subj_obj(sent, i):
    o = None
    s = None

    sent = nlp(sent)
    #print(elem.text, elem.lemma_, elem.pos_, elem.tag_, elem.dep_, elem.shape_, elem.is_alpha, elem.is_stop)

    for elem in sent:
            if elem.dep_ in p_subj:
                if elem.lemma_ != "-PRON-":
                    s = elem.lemma_
                else:
                    s = elem.text
            if elem.dep_ in p_obj:
                if elem.lemma_ != "-PRON-":
                    o = elem.lemma_
                else:
                    o = elem.text

    parsed_sent(sent)

    #print("It ", i, "subj", s, " obj ", o)
    return s, o

def parsed_sent(sent):
    psd_sent = []
    psd_sent.append(sent)
    #print(psd_sent)

def wsd(sent, subj, obj):
    possible_subj = ["i", 'you', 'he', "she", "it", "we", "they"]
    if subj in possible_subj:
        ris = wn.synsets('people')[0]               #Prende solo il primo synset
    elif subj is not None:
        ris = lesk(sent, subj)
    else:
        ris = None
    if obj is not None:
        ris1 = lesk(sent, obj)
    else:
        ris1 = None
    #super_sense(ris, ris1)
    #print(sent)
    #print("Ris ", ris, "Ris1 ", ris1)
    return ris, ris1


def super_sense(ris, ris1):
    #Prendiamo i supersensi
    if ris is not None or ris1 is not None:
        #print("Supersense/Tipo semantico per il filler")
        if ris is not None:
            ss1 = ris.lexname()
        else:
            ss1 = None
        if ris1 is not None:
            ss2 = ris1.lexname()
        else:
            ss2 = None
        #print("Subj supersense", ris.lexname())
        #print("Obj supersense", ris1.lexname(), "\n")
    else:
        #print("\nDisambiguazione = None\n")
        ss1 = None
        ss2 = None

    return ss1, ss2


def generate_word_cloud(slot):
    proc = list_to_string(slot)
    #print("Stringatizzato per wc\n", proc)

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(proc)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def list_to_string(lista):
    trs = " ".join(lista)

    return trs

def menu():
    print("Scegli il verbo che vuoi analizzare")
    print("1 - To build;")
    print("2 - To cook;")