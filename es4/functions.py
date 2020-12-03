import string
import nltk
import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.wsd import lesk


nlp = spacy.load("en_core_web_sm")

def read_file():

    with open("input\iceland.txt", "r", encoding="utf-8") as f:
        content = f.read()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #content = [x.strip() for x in content]
    f.close()
    content = content.replace("\n", "")
    #content = [x.strip() for x in content]

    return content


def preprocessing(sent):
    sent = sent.lower()
    stop_word = set(stopwords.words('english'))
    punctuation = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    space = " "
    for el in sent:
        if el in punctuation:
            sent = sent.replace(el, space)
    sent = sent.split()
    numbers = "0123456789"
    sent = list(filter(lambda x: x not in stop_word and x not in numbers, sent))  #Stop words rimosse
    text = space.join(sent)

    return text


def pipeline(sent):

    sent = sent.lower()
    ris = set()
    dis = set()
    syn = []
    sent_tokens = nlp(sent)
    for token in sent_tokens:
        if token.pos_ != "PUNCT" and not token.is_stop:
            if token.lemma_ != "-PRON-":
                ris.add(token.lemma_)
            else:
                ris.add(token.text)
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
            #print(sent.split(), token.lemma_, token.pos_, token.tag_)
            syn.append(lesk(sent.split(), token.lemma_))

    hyp = []
    for s in syn:
        hyp.append(s.hypernyms())

    print(syn, "\n", hyp)

    return ris


def wsd(sent, lemma, pos):
    dis = lesk(sent, lemma, pos)

def window_analysis(text):
    initial_index = 6
    final_length = len(text)

    stop_words = set(stopwords.words('english'))

    punctuation = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)


    #print()









