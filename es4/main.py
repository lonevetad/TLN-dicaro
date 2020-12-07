import collections
from collections import Counter, OrderedDict
import numpy as np
from functions import *


if __name__ == '__main__':
    text = read_file()
    #text = text.strip()

    text = text.split('.')
    processed = []

    for i in range(0, 5):
        text[i] = text[i].strip()       #testo pulito
        if len(text[i]) > 0:
            processed.append(pipeline(text[i]))            #preprocessing OK

    flat_list = []
    #print(processed)
    window_analysis(text)

    dictionary = {}

    for line in processed:
        for word in line:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1

    #print(dictionary)
    sorted = sorted(dictionary.items(), key = lambda x: x[1], reverse = True)
    #print(sorted)






