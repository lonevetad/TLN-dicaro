import collections

# inizio cose per stampare liste multidimensionali

def print_tab(level, end_line="\n"):
    if level > 0:
        print("\t" * level, end=end_line)


def deep_array_printer(x, level=0):
    if isinstance(x, list):
        print('[')
        new_level = level + 1
        i = 0
        for y in x:
            print_tab(new_level, "")
            print(i, '\t -> ', end="")
            deep_array_printer(y, new_level)
            i += 1
        print_tab(level, "")
        print(']')
    else:
        print(x)

# fine cose per stampare liste multidimensionali

#

# inizio ALTROH

#

def split_csv_row_string(rs):
    '''
	A CSV file if made of rows, whose columns are delimited by a single comma.
	The rows containing commas are wrapped inside double quotes.
	This method scan a string-converted row
	(i.e. 'Column 1, Col. 2, "Some text, as example, on col 3", col 4,, col6,,,col9')
	and returns a list of strings, one for each column.
	Empty columns results in empty strings.
	'''
    start = 0
    length = len(rs)
    parts = []
    if rs[0] == '\"':
        i = 1
        start = 1
        delimiter = '\"'
    else:
        i = 0
        delimiter = ','
    while i < length:
        if rs[i] is delimiter:
            if i != start:
                parts.append(rs[start:i])
            else:  # record the empty string
                parts.append("")
            # move to the next chunk
            if delimiter == '\"':
                i += 2
            else:
                i += 1
            start = i
            if start < length:
                if rs[start] == '\"':
                    delimiter = '\"'
                    i += 1
                    start = i
                else:
                    delimiter = ','
            else:
                if rs[length - 1] == ',':
                    parts.append("")
        else:
            i += 1
    if i > length:
        i = length
    if start < i:  # add the last part that I forgot
        parts.append(rs[start:i])
    return parts
