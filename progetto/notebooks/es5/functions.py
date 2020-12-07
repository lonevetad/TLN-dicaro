from itertools import cycle

import spacy

nlp = spacy.load("en_core_web_sm")


def read_file():
    with open("characters.txt", "r", encoding="utf-8") as file_in:
        sentences = []
        for line in file_in:
            line_stripped = line.strip()
            if len(line_stripped) > 0:
                sentences.append(line_stripped)

    return sentences

#print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

p_subj = {'subj', 'nsubjpass', 'nsubj'}
p_obj = {'pobj', 'dobj', 'obj', 'iobj'}
p_vb = {"V", "VP", "verb", "VBZ", "VBN", "VBD", "VERB", "aux", "auxpass"}

class OieTriple(object):
    def __init__(self):
        self.subj = None
        self.rel = None
        self.obj = None

    def isCompound(self, word, dep):
        return dep == "compound"

    def isObj(self, word, dep):
        return dep in p_obj or self.isCompound(word, dep)

    def isSubj(self, word, dep):
        return dep in p_subj or self.isCompound(word, dep)

    def isVerb(self, word, pos, tag):
        return (pos in p_vb) or (tag in p_vb)


def pipeline(sent, i):
    processed = nlp(sent)

    ot = OieTriple()
    cont = 0
    temp_com = None                 #Prima parte di nome composto
    for word in processed:
        if (word.dep_ == 'compound' and word.pos_ == 'PROPN') or (word.tag_ in p_vb and (word.dep_ == 'aux' or word.dep_ == 'auxpass')):
            if temp_com is None:
                temp_com = word.text
            else:
                temp_com = temp_com + " " + word.text
        else:
            if ot.isSubj(word, word.dep_):
                if ot.subj is None:
                    if temp_com is not None:
                        ot.subj = temp_com + " " + word.text
                        temp_com = None
                    else:
                        ot.subj = word.text
            elif ot.isObj(word, word.dep_):
                if ot.obj is None:
                    if temp_com is not None:
                        ot.obj = temp_com + " " + word.text
                        temp_com = None
                    else:
                        ot.obj = word.text
            elif ot.isVerb(word, word.pos_, word.tag_):
                if ot.rel is None:
                    if temp_com is not None:
                        ot.rel = temp_com + " " +word.text
                        temp_com = None
                    else:
                        ot.rel = word.text
        #print(word.text, word.lemma_, word.pos_, word.tag_, word.dep_, word.shape_, word.is_alpha, word.is_stop)
    print("Tripla ", i, " - (", ot.subj, ", ", ot.rel, ", ", ot.obj, ")")














