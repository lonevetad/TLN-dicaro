from functions import *

if __name__ == '__main__':
    sents = read_file()
    #print(sents)
    i= 0

    for sent in sents:
        i += 1
        #print("Frase", i, "- ", sent )
        pipeline(sent, i)


