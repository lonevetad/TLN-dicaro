import nltk
import string
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv



def read_load_csv():
    with open('..\input\definitions.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        col1 = []
        col2 = []
        col3 = []
        col4 = []

        #Controllo "eliminazione" prima riga
        frs = 1
        for cat in csv_reader:
            if frs != 1:
                col1.append(cat[0])
                col2.append(cat[1])
                col3.append(cat[2])
                col4.append(cat[3])
            else:
                frs = 0

    return col1, col2, col3, col4


def preprocessing(definitions):
    clean_list = []
    lem_sents = []

    for defs in definitions:
        defs = defs.lower()
        defs = "".join([val for val in defs if val not in string.punctuation])
        clean_list.append(defs)

    for i in range(len(clean_list)):
        sentence = clean_list[i]
        lem_sents.append(lemmatization(sentence))

    return lem_sents


def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    lemmatized_sentences = []
    sentence = sentence.split()

    #Rimozione stopword
    sentence = list(filter(lambda x: x not in stop_words, sentence))

    #Lemmatizzazione
    for i in range(len(sentence)):
        lemmatized_sentences.append(lemmatizer.lemmatize(sentence[i]))

    return lemmatized_sentences


def compute_sim(sent1, sent2):
    sent1 = set(sent1)
    sent2 = set(sent2)

    int_card = len(sent1.intersection(sent2))

    #Verificare la lunghezza minima delle 2 frasi
    len_sent1 = len(sent1)
    len_sent2 = len(sent2)
    #if len_sent1 != 0 and len_sent2 !=0:
    if len_sent1 == 0 or len_sent2 == 0:
        return 1 if len_sent1 == 0 and len_sent2 == 0 else 0
    else:
        min_len = min(len_sent1, len_sent2)            #Prende il massimo valore tra i 2 in caso uno fosse = 0

    similarity = int_card/min_len                      #Normalizza l'intersezione tra 2 definizioni
                                                       #sulla lunghezza minima tra le 2
    return similarity






