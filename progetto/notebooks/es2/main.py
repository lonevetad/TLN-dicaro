import csv

from notebooks.es2 import es2
from notebooks.utilities import utils, functions


def read_csv():
    with open('Esperimento content-to-form - Foglio1.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        can_use = False  # la prima riga deve essere salata
        cols = None
        i = 0
        for row in csv_reader: # "row" e' una stringa rozza e grezza
            if len(row) > 0:
                if can_use:
                    row_string = row[0]
                    i = 0
                    cols_in_this_row = utils.split_csv_row_string(row_string)
                    length = len(cols_in_this_row) - 1
                    while i < length:
                        cols[i].append(cols_in_this_row[i + 1])  # because the first column is reserved for indexes
                        i += 1
                else:
                    cols_names = row[0].split(",")
                    cols = [[] for i in range(len(cols_names) - 1)]  # because the first column is reserved for indexes
                    can_use = True
            i += 1
        return cols
    return None


def colum_csv_to_bag_of_words(column):
    bag = set()
    for c in column:
        if len(c) > 0:
            bags_of_c = functions.filter_and_lemmatize_words_in(c)
            for w in bags_of_c:
                bag.add(w)
    return bag


def main():
    # print(get_synsets("play"))
    print("start :D\n\n")

    cols = read_csv()
    # print("\n\nCOLONNEEEEEE")
    # deep_array_printer(cols)

    similarities_to_test = 2

    while similarities_to_test > 0:
        similarities_to_test -= 1
        useOverlap = similarities_to_test > 0

        print("\n\n\n\n")
        for i in range(0, 5):
            print("-------------------------------------------------")
        print(" testing similarity", "overlap" if useOverlap else "jaccard")

        i = 0
        for column in cols:
            print("\n\n\n elaborating th given column:", i)
            #utils.deep_array_printer(column)
            bag_for_col = colum_csv_to_bag_of_words(column) # sono le informazioni fornite dalla "colonna di definizioni"
            best_synset, best_simil = es2.searchBestApproximatingSynset(bag_for_col, addsAntinomies=True, usingOverlapSimilarity=useOverlap)
            print("found: ", best_synset, ", with similarity of:", best_simil)
            i += 1

    # beware of entries in columns having len(field) == 0 ....

    print("\n\n\n fine")


main()
