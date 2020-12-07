import math
import collections
from typing import Dict, Iterable
from sortedcontainers import SortedDict

from notebooks.utilities import synsetInfoExtraction
from notebooks.utilities.cacheVarie import CacheSynsets
from notebooks.utilities.synsetInfoExtraction import DEFAULT_SYNSET_EXTRACTION_OPTIONS, FilteredWeightedWordsInSentence


class WordWeighted(object):
    def __init__(self, word):
        self.word = word
        self.cumulativeWeight = 0
        self.countPresenceInDocument = 0
        self.cacheTotalValue = -1  # invalidates cache

    def recalculateTotalValue(self):
        # the additional 1 inside the second factor is provided to still consider the word's weight,
        # even if it's not present in the document
        return int(math.floor(self.cumulativeWeight * (1 + math.log2(1 + self.countPresenceInDocument))))

    def getTotalWeigth(self):
        if self.cacheTotalValue < 0:
            self.cacheTotalValue = self.recalculateTotalValue()
        return self.cacheTotalValue

    def addWeight(self, weight, isPresentInDocument=False):
        self.cumulativeWeight += weight
        self.cacheTotalValue = -1  # invalidates cache
        if isPresentInDocument:
            self.countPresenceInDocument += 1


class CachePairwiseSentenceSimilarity:
    def __init__(self, map_words_weights, list_word_bags_from_sentences, sentence_similarity_function):
        """
        :param map_words_weights: map <string, int> representing the words' weights
        :param list_word_bags_from_sentences: a list of set of string, i.e. a list of bag of words
        each extracted from a respective sentence
        :param sentence_similarity_function: a function that accepts a map <string, int> (words' weights),
        and two strings (sentences) and returns a float (the similarity between those)
        """
        self.map_words_weights = map_words_weights
        self.list_word_bags_from_sentences = list_word_bags_from_sentences
        self.sentence_similarity_function = sentence_similarity_function
        self.cache_simil = [None] * len(list_word_bags_from_sentences)

    def get_similarity_by_sentence_indexes(self, index_sentence1, index_sentence2):
        minind = 0
        maxind = 0
        if index_sentence1 > index_sentence2:
            minind = index_sentence2
            maxind = index_sentence1
        elif index_sentence1 == index_sentence2:
            raise ValueError(
                "Shouldn't calculate the similarity of the same sentence (index: " + str(index_sentence1) + ")")
        else:
            minind = index_sentence1
            maxind = index_sentence2
        m = self.cache_simil[minind]
        if m is None:
            m = {}
            self.cache_simil[minind] = m
        if maxind in m:
            return m[maxind]
        else:
            sim = self.sentence_similarity_function(self.map_words_weights,
                                                    self.list_word_bags_from_sentences[index_sentence1],
                                                    self.list_word_bags_from_sentences[index_sentence2])
            m[maxind] = sim
            return sim




#
#
# start Paragraph
#
#

class Paragraph(object):
    def __init__(self, documentSegmentator):
        if not (isinstance(documentSegmentator, DocumentSegmentator)):
            raise ValueError("The first constuctor parameter must be a DocumentSegmentator")

        self.documentSegmentator = documentSegmentator
        self.score = -1
        self.lowest_index_sentence = 0
        self.highest_index_sentence = -1  # ESTREMI INCLUSI
        self.previous_paragraph = None
        self.next_paragraph = None
        self.words_cooccurrence_in_bags_counter: Dict[str, int] = {}

    def is_empty(self):
        # return len(self.map_sentence_by_index) == 0
        return self.lowest_index_sentence > self.highest_index_sentence

    def size(self):
        return 0 if self.is_empty() else 1 + (self.highest_index_sentence - self.lowest_index_sentence)

    def __adjust_indexes__(self):
        # fix wrong data
        if self.lowest_index_sentence < 0:
            self.lowest_index_sentence = 0
        if self.highest_index_sentence >= len(self.documentSegmentator.list_of_sentences):
            self.highest_index_sentence = len(self.documentSegmentator.list_of_sentences) - 1

    def raiseNonContiguousError(self, par):
        raise ValueError("Non contiguous paragraphs: self:(" + str(self.lowest_index_sentence) + ";" +
                         str(self.highest_index_sentence) + "), given:(" +
                         str(par.lowest_index_sentence) + ";" + str(par.highest_index_sentence) + ")")

    def merge_paragraph(self, par):
        """
        Merge the "lowest" (in term of starting index) paragraph
        into the "highest".
        Will rise an exception if they are not contiguous.
        :param par: a given paragraph
        :return: the remaining paragraph, or None in case of non-Paragraph parameter
        """
        if not isinstance(par, Paragraph):
            return None
        if self.lowest_index_sentence > par.lowest_index_sentence:
            return par.merge_paragraph(self)
        # I'm the lowest
        if ((
                    self.highest_index_sentence + 1) != par.lowest_index_sentence) or self.next_paragraph != par or par.previous_paragraph != self:
            self.raiseNonContiguousError(par)
        self.highest_index_sentence = par.highest_index_sentence
        # merge links
        if par.next_paragraph is not None:
            par.next_paragraph.previous_paragraph = self
        # if self.previous_paragraph is not None:
        #    self.previous_paragraph.next_paragraph
        self.next_paragraph = par.next_paragraph
        par.next_paragraph = None
        par.previous_paragraph = None
        # update score (cohesion) caches
        par.score = -1
        for i in range(par.lowest_index_sentence, par.highest_index_sentence + 1):
            self.__add_sentence_bag_to_counter_by_index__(i)
        return self

    def add_sentence(self, sentence, i, is_start_of_paragraph=True):
        self.score = -1
        # self.map_sentence_by_index[i] = sentence
        if self.is_empty():
            self.lowest_index_sentence = i
            self.highest_index_sentence = i
        else:
            if is_start_of_paragraph:
                self.lowest_index_sentence = i
            else:
                self.highest_index_sentence = i
        self.__adjust_indexes__()
        self.__add_sentence_bag_to_counter_by_index__(i)

    def remove_sentence(self, i):

        # self.map_sentence_by_index.pop(i)
        removed = False
        if i == self.lowest_index_sentence:
            self.lowest_index_sentence += 1
            removed = True
        elif i == self.highest_index_sentence:
            self.highest_index_sentence -= 1
            removed = True
        if removed:
            self.score = -1
            self.__adjust_indexes__()
            self.__remove_sentence_bag_to_counter_by_index__(i)

    def __add_sentence_bag_to_counter_by_index__(self, i):
        bag_sent = self.documentSegmentator.get_bag_of_word_of_sentence_by_index(i)
        if len(bag_sent) > 0:
            self.score = -1  # invalidates score(cohesion) cache
        for w in bag_sent:
            if w in self.words_cooccurrence_in_bags_counter:
                self.words_cooccurrence_in_bags_counter[w] += 1
            else:
                self.words_cooccurrence_in_bags_counter[w] = 1

    def __remove_sentence_bag_to_counter_by_index__(self, i):
        bag_sent = self.documentSegmentator.get_bag_of_word_of_sentence_by_index(i)
        for w in bag_sent:
            if w in self.words_cooccurrence_in_bags_counter:
                self.score = -1  # invalidates score(cohesion) cache
                c = self.words_cooccurrence_in_bags_counter[w]
                if c == 1:
                    del self.words_cooccurrence_in_bags_counter[w]
                else:
                    self.words_cooccurrence_in_bags_counter[w] -= 1

    '''
    SVILUPPI FUTURI:
    si potrebbe ottimizzare il calcolo della coesione memorizzando non il singolo "score",
    ma solo il numeratore (il denominatore è la cardinalità dell'insieme di frasi, ossia un
    dato facilmente reperibile).
    Aggiungendo una frase (come sopra), si potrebbero sommare allo "score" tutti i pesi delle
    parole che sono già presenti in "words_cooccurrence_in_bags_counter" MA con c == 1.
    Analogamente, in fase di rimozione di una frase si rimuovono i pesi delle parole con
    esattamente c == 2 (con c == 1 la parola verrebbe già rimossa dal codice soprastante).
    La "cohesion" sarebbe, quindi, la divisione dello score con la dimensione del paragrafo
    (ossia, non si fa cache: è solo una divisione, d'altronde).
    '''

    def get_cohesion(self):
        if self.is_empty():
            return 0
        if self.score >= 0:
            return self.score
        self.score = 0
        for w, c in self.words_cooccurrence_in_bags_counter.items():
            if c > 1:
                self.score += self.documentSegmentator.words_weight[w].getTotalWeigth()  # it's not multiplied by "c"
        self.score /= self.size()
        return self.score

    def get_first_sentence_index(self):
        """
        The extremes are included.
        :return: the index of the first sentence held by this paragraph
        """
        if self.is_empty():
            return -1
        return self.lowest_index_sentence

    def get_first_sentence(self):
        if self.is_empty():
            return None
        # return self.map_sentence_by_index.peekitem(0)
        return self.documentSegmentator.list_of_sentences[self.lowest_index_sentence]

    def get_last_sentence_index(self):
        """
        The extremes are included.
        :return: the index of the last sentence held by this paragraph
        """
        if self.is_empty():
            return -1
        return self.highest_index_sentence

    def get_last_sentence(self):
        if self.is_empty():
            return None
        # return self.map_sentence_by_index.peekitem(0)
        return self.documentSegmentator.list_of_sentences[self.highest_index_sentence]

    def words_cooccurrence_counter_to_words_set(self):
        return set(w for w, c in self.words_cooccurrence_in_bags_counter.items())

    def toString(self):
        sss = "P[(" + str(self.lowest_index_sentence) + "; " + str(self.highest_index_sentence) + "), size: " + str(
            self.size())
        if self.previous_paragraph is None:
            sss += ", no prev"
        else:
            sss += ", prev-" + str(self.previous_paragraph.lowest_index_sentence)
        if self.next_paragraph is None:
            sss += ", no next"
        else:
            sss += ", next-" + str(self.next_paragraph.lowest_index_sentence)
        return sss + "]"


#
#
# end Paragraph
#
#

#
#
# start DocumentSegmentator
#
#

class DocumentSegmentator(object):
    def __init__(self, list_of_sentences, cache_synset_and_bag=None, options=None):
        """
        :param list_of_sentences: list of sentences
        :param cache_synset_and_bag:  a CacheSynsets object or None, used to cache synsets to speed up and reduce the use of Internet
        (at the cost of more memory usage)
        :param options:  a SynsetInfoExtractionOptions object or None, used to specify what information are required to be
        collected from synsets
        """
        if cache_synset_and_bag is None:
            cache_synset_and_bag = CacheSynsets()
        if options is None:
            options = DEFAULT_SYNSET_EXTRACTION_OPTIONS
        self.list_of_sentences = list_of_sentences
        self.cache_synset_and_bag = cache_synset_and_bag
        self.options = options
        self.map_sentence_to_bag = {}
        self.bag_from_sentence_list = []
        self.cache_bag_sentence_similarity = None
        self.words_weight = {}

    # start document pre-processing

    def get_sentence_by_index(self, i):
        return self.list_of_sentences[i]

    def get_bag_of_word_of_sentence_by_sentence(self, sentence):
        return self.map_sentence_to_bag[sentence]

    def get_bag_of_word_of_sentence_by_index(self, i):
        return self.get_bag_of_word_of_sentence_by_sentence(self.list_of_sentences[i])

    def get_weighted_word_map_for_sentence(self, sentence):
        """
        :param sentence:  an English sentence in string variable
        :return:  a map <string, float> mapping each non-stop word in the given sentence
        into a float positive value: a weight indicating how much that word helps in
        describing the argument of the sentence
        """
        h = synsetInfoExtraction.weighted_bag_for_sentence(sentence)
        if not isinstance(h, FilteredWeightedWordsInSentence):
            raise ValueError("get_weighted_word_map_for_sentence returned a non- FilteredWeightedWordsInSentence "
                             "instance:\n" + str(h))
        self.map_sentence_to_bag[sentence] = h.filtered_words
        self.bag_from_sentence_list.append(h.filtered_words)
        return h.word_to_weight_mapping

    #
    #
    # end pre-processing del documento
    #
    #

    #
    #
    # start processing the whole document
    #
    #

    def weighted_intersection(self, word_weight_map, string_set1, string_set2):
        """
        :param word_weight_map: a mapping <string,int> representing the weights
        :param string_set1: a set of words
        :param string_set2: a set of words
        :return:
        """
        if len(string_set2) < len(string_set1):
            return self.weighted_intersection(word_weight_map, string_set2, string_set1)
        # consider the first as the smaller set
        summ = 0
        for w in string_set1:
            if w in string_set2:
                summ += word_weight_map[w].getTotalWeigth()
        return summ

    def weighted_overlap(self, word_weight_map, string_set1, string_set2):
        wi_sum = self.weighted_intersection(word_weight_map, string_set1, string_set2)
        minlen = min(len(string_set1), len(string_set2))
        if minlen == 0:
            return 0
        return wi_sum / minlen

    def similarity(self, word_weight_map, string_set1, string_set2):
        """
        similarity function
        :param word_weight_map: a map <string, int> representing the words' weights
        :param string_set1: a set of words, extracted from a sentence
        :param string_set2: as the previous parameter
        :return: a float, indicating how much those sentences are similar
        """
        return self.weighted_overlap(word_weight_map, string_set1, string_set2)

    #

    def doc_tiling(self, windows_count, max_iterations=8):
        """
        :param windows_count: the amount of paragraph to find. It's greater by 1 than the length of
        the returned list
        :param max_iterations: the inner algorithm improves iteratively the tiling; this parameter
        sets an upper bound of iterations
        :return: a list of breakpoints: indexes (between one sentence and the next) where one paragraph ends and
        the next starts. The length is equal to the parameter "windows_count", so the last index is the last sentence.
        So, the indexes are to be considered as "inclusive".
        """
        '''
        all'inizio:
        - si genera un Paragraph per ogni frase
        - per ogni Paragraph (tranne primo ed ultimo, che è scontato)
            si calcola quale frase tra la precedente e la successiva
            è la migliore candidata per la fusione dei paragrafi
        - (convertire i backpointer per comodita', vedere dopo)
        - fondere ogni paragrafo con quello puntato
        
        POI
        ...
        '''
        sentences = self.list_of_sentences
        sentences_amount = len(sentences)
        paragraphs_by_starting_index = SortedDict()

        # maps < a paragraph's lowest index -> the par.'s lowest index wish to merge into
        preferences = SortedDict()

        # inizializzazione
        i = 0
        prevParagraph = None
        while i < sentences_amount:  # creo i paragrafi
            par = Paragraph(self)
            par.add_sentence(sentences[i], i, is_start_of_paragraph=False)
            if prevParagraph is not None:
                prevParagraph.next_paragraph = par
                par.previous_paragraph = prevParagraph
            paragraphs_by_starting_index[i] = par
            prevParagraph = par
            i += 1

        # cerco le preferenze iniziali
        for j, par in paragraphs_by_starting_index.items():
            if j == 0:
                # scelta obbligata
                preferences[0] = 1
            elif j == (sentences_amount - 1):
                # scelta obbligata
                preferences[j] = j - 1  # il penultimo
            else:
                simil_prev = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(j - 1, j)
                simil_next = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(j, j + 1)
                if simil_next >= simil_prev:
                    preferences[j] = j + 1
                else:
                    preferences[j] = j - 1
                    '''
                    V2
                    a ragion veduta, si puo' risparmiare la conversione dei backpointer
                    mettendola qui, dato che le "back-chain" vengono costruite iterativamente
                    quando si cade, consecutivamente, in questo ramo
                    
                    pref_of_prev = preferences[j - 1]
                    if j > 1 and pref_of_prev < (j - 1):
                        preferences[j - 1] = j  # conversione del backpointer
                        
                    ma questo codice ha prodotto dei problemi, quindi non viene
                    '''

        # conversione di tutti i backpointers
        # perche' tanto andranno a finire nello stesso paragrafo
        i = sentences_amount - 1
        while i > 0:
            pref = preferences[i]
            if pref < i:
                # search for the start of a "back sequence": convert all other stuffs
                j = pref  # j == i-1 per costruzione
                # a do-while ...
                even_prev = preferences[j]
                while even_prev < j and 0 < j:
                    preferences[j] = j + 1
                    j = even_prev
                    even_prev = preferences[j]
                # j holds the last of the backward chain (i.e., the first to be redirected

                # qualora non si entrasse nel ciclo (il paragrafo ha 2 frasi), semplicemente si riconfermera' la precedenza
                preferences[j] = j + 1
                i = j - 1
            else:
                i -= 1

        # ora i backpointers possono essere trattati come "terminatori di paragrafo"
        # merge delle preferenze:
        start = 0
        end = 0
        # V2
        # si procede a ritroso per semplicita
        end = sentences_amount - 1
        start = end
        # per ogni paragrafo
        while start > 0:
            start = end - 1
            # almeno due elementi nel paragrafo, per costruzione
            par_end = paragraphs_by_starting_index[end]
            paragraphs_by_starting_index[start] = paragraphs_by_starting_index[start].merge_paragraph(par_end)
            paragraphs_by_starting_index.pop(end)
            start -= 1  # jump to the next (previous, tbh) sentence
            pref = preferences[start]
            if start < pref:
                # sequence not ended: the current paragraph (end-1) could be merged into start
                while 0 <= start < pref:
                    paragraphs_by_starting_index[start] = paragraphs_by_starting_index[start].merge_paragraph(
                        paragraphs_by_starting_index[pref])
                    paragraphs_by_starting_index.pop(pref)
                    start -= 1
                    if 0 <= start:
                        pref = preferences[start]
                end = start
            else:
                # the paragraph has ended: start is pointing backward
                end = start

        # WELL, INITIALIZATION HAS ENDED
        # now make the paragraphs-bubble boiling

        # START MAIN ALGORITHM - V2
        is_up = True
        for iteration in range(0, max_iterations):
            is_up = (iteration % 2) == 1

            paragraphs_to_be_processed = []
            amount_paragr_pairs = len(paragraphs_by_starting_index) - 1
            if is_up:
                stack_paragraphs = collections.deque()
                for ind, par in paragraphs_by_starting_index.items():
                    if ind != 0:
                        stack_paragraphs.appendleft(par)
                for par in stack_paragraphs:
                    prev_cohesion_current = par.get_cohesion()
                    prev_cohesion_prev = par.previous_paragraph.get_cohesion()
                    index_first = par.get_first_sentence_index()
                    par.remove_sentence(index_first)  # __remove_sentence_bag_to_counter_by_index__(index_first)
                    par.previous_paragraph.add_sentence("will be ignored :D", index_first,
                                                        is_start_of_paragraph=False)

                    # __add_sentence_bag_to_counter_by_index__(index_first)
                    modified_cohesion_current = par.get_cohesion()
                    modified_cohesion_prev = par.previous_paragraph.get_cohesion()
                    sum_previous = prev_cohesion_current + prev_cohesion_prev
                    sum_modified = modified_cohesion_current + modified_cohesion_prev
                    # restore previous situation
                    par.add_sentence("will be ignored :D",
                                     index_first)  # __add_sentence_bag_to_counter_by_index__(index_first)
                    par.score = prev_cohesion_current
                    par.previous_paragraph.remove_sentence(
                        index_first)  # __remove_sentence_bag_to_counter_by_index__(index_first)
                    par.previous_paragraph.score = prev_cohesion_prev
                    # check modifications
                    if sum_previous < sum_modified:
                        paragraphs_to_be_processed.append((par, modified_cohesion_current, modified_cohesion_prev))
                # print("\n\non going UP: ", len(paragraphs_to_be_processed), "paragraphs to process:")
                for tupla in paragraphs_to_be_processed:
                    par = tupla[0]
                    # print("processing -", par.toString())
                    par_index = par.lowest_index_sentence  # che sarebbe la sua "chiave"
                    par.previous_paragraph.add_sentence("will be ignored :D", par_index, is_start_of_paragraph=False)
                    par.remove_sentence(par_index)
                    par.score = tupla[1]
                    par.previous_paragraph.score = tupla[2]
                    # adjust indexes
                    paragraphs_by_starting_index.pop(par_index)
                    if not par.is_empty():
                        paragraphs_by_starting_index[par_index + 1] = par
                        # ora vale: par_index+1 == par.lowest_index_sentence
            else:  # we're going dooown, dooown, dooooown, we're going dooown, dooOOWWN, DOOOOWN [cit. Bruce Springsteen]
                # scorro tutte le coppie di paragrafi, ossia tutti i par. tranne l'ultimo
                i = 0
                for ind, par in paragraphs_by_starting_index.items():
                    if i != amount_paragr_pairs:
                        prev_cohesion_current = par.get_cohesion()
                        prev_cohesion_next = par.next_paragraph.get_cohesion()
                        index_last = par.get_last_sentence_index()
                        par.remove_sentence(index_last)
                        par.next_paragraph.add_sentence("will be ignored :D", index_last, is_start_of_paragraph=True)
                        modified_cohesion_current = par.get_cohesion()
                        modified_cohesion_next = par.next_paragraph.get_cohesion()
                        sum_previous = prev_cohesion_current + prev_cohesion_next
                        sum_modified = modified_cohesion_current + modified_cohesion_next
                        # restore previous situation
                        par.add_sentence("will be ignored :D", index_last, is_start_of_paragraph=False)
                        par.score = prev_cohesion_current
                        par.next_paragraph.remove_sentence(index_last)
                        par.next_paragraph.score = prev_cohesion_next
                        # check modifications
                        if sum_previous < sum_modified:
                            paragraphs_to_be_processed.append((par, modified_cohesion_current, modified_cohesion_next))
                    i += 1
                for tupla in paragraphs_to_be_processed:
                    par = tupla[0]
                    par_next_index = par.next_paragraph.lowest_index_sentence
                    # aggiungo la frase al paragrafo dopo
                    index_to_add = par.get_last_sentence_index()
                    par.next_paragraph.add_sentence("will be ignored :D", index_to_add, is_start_of_paragraph=True)
                    # la rimuovo da quello attuale
                    par.remove_sentence(index_to_add)
                    par.score = tupla[1]
                    par.next_paragraph.score = tupla[2]
                    paragraphs_by_starting_index.pop(par_next_index)
                    paragraphs_by_starting_index[index_to_add] = par.next_paragraph
                    if par.is_empty():
                        paragraphs_by_starting_index.pop(par.lowest_index_sentence)

        # Union of smaller paragraph (those with 1 or 2 sentences)
        for par in [ppp for ind, ppp in paragraphs_by_starting_index.items() if ppp.size() <= 2]:
            par_context = par.words_cooccurrence_counter_to_words_set()
            prev_score = 0
            if par.previous_paragraph is not None:
                prev_score = self.weighted_overlap(self.words_weight, par_context,
                                                   par.previous_paragraph.words_cooccurrence_counter_to_words_set())
            next_score = 0
            if par.next_paragraph is not None:
                next_score = self.weighted_overlap(self.words_weight, par_context,
                                                   par.next_paragraph.words_cooccurrence_counter_to_words_set())
            if ((par.previous_paragraph is not None) and (prev_score >= next_score)) or (par.next_paragraph is None):
                if par.next_paragraph is None:
                    print("POSSIBLE BUG: the paragraph", par.toString(), "should not be merged into the previous [i.e.",
                          par.previous_paragraph.toString(), "], but the next is None.")
                    print("\nThe weighted overlap with previous paragraph is:", prev_score)
                par.previous_paragraph.merge_paragraph(par)
                del paragraphs_by_starting_index[par.get_first_sentence_index()]
            else:
                paragraphs_by_starting_index[par.lowest_index_sentence] = par
                index_of_next = par.next_paragraph.lowest_index_sentence
                par.merge_paragraph(par.next_paragraph)
                del paragraphs_by_starting_index[index_of_next]

            # END MAIN ALGORITHM - V2

            # conversione in array di indici
        return [par.highest_index_sentence for i, par in paragraphs_by_starting_index.items()]

    def document_segmentation(self, desiredParagraphAmount=0) -> Iterable[Iterable[str]] or list:
        if desiredParagraphAmount < 2:
            desiredParagraphAmount = 2
        words_mapped_each_sentences = [self.get_weighted_word_map_for_sentence(sentence) for sentence in
                                       self.list_of_sentences]
        w_w = self.words_weight  # contiene i pesi finali delle parole
        # aggreghiamo i pesi delle parole:
        # ispirandosi all term-frequency, il peso finale è
        # la somma di tutti i pesi moltiplicta per floor(1+log(numero occorrenze nel documento))
        i = 0
        for map_for_a_sent in words_mapped_each_sentences:
            for word, weight in map_for_a_sent.items():
                bag_of_word_of_sentence = self.map_sentence_to_bag[self.list_of_sentences[i]]
                is_in_sentence = word in bag_of_word_of_sentence
                if word in w_w:
                    w_w[word].addWeight(weight, isPresentInDocument=is_in_sentence)
                else:
                    w = WordWeighted(word)
                    w.addWeight(weight, isPresentInDocument=is_in_sentence)
                    w_w[word] = w
            i += 1

        self.cache_bag_sentence_similarity = CachePairwiseSentenceSimilarity(
            map_words_weights=w_w,
            list_word_bags_from_sentences=self.bag_from_sentence_list,
            sentence_similarity_function=self.similarity
        )

        # now segment
        breakpoint_indexes = self.doc_tiling(desiredParagraphAmount)
        print("list of breakpoints (the division will happen between the shown indexes and their respective successors):")
        print(breakpoint_indexes)
        desiredParagraphAmount = len(breakpoint_indexes)  # forzo il fatto di mantenere i paragrafi
        i = 0
        start = 0
        subdivision = [None] * desiredParagraphAmount
        while i < desiredParagraphAmount:
            subdivision[i] = self.list_of_sentences[start: breakpoint_indexes[i] + 1]
            start = breakpoint_indexes[i] + 1
            i += 1
        return subdivision


'''
SVILUPPI FUTURI:
Si potrebbe modificare "doc_tiling" in modo da tenere conto del numero di paragrafi
richiesti (ossia il parametro "desiredParagraphAmount") utilizzando tale valore
(che viene assegnato al parametro "windows_count) nel modo seguente:

- se il numero di paragrafi prodotti fosse maggiore di quello desiderato:
    ispirandosi all'algoritmo implementato da riga 579 in poi (marcato col commento "Union of smaller paragraph"),
    si possono iterativamente cercare i paragrafi che hanno maggiore weighted overlap con un qualche paragrafo
    adiacente e fonderli, iterando fino a quando non si raggiunge il numero desiderato di paragrafi
- se fosse minore, invece:
    un ciclo in cui, fintanto che il numero di paragrafi è inferiore a quello desiderato:
        cerca il paragrafo più grande
        cerca la frase con minore weighted overlap rispetto al contesto generato da ogni altra frase (del paragrafo) messa assieme
            iterare la ricerca per la seconda frase, o terza, etc, qualora le frase trovata generasse paragrafi troppo piccoli (da 1 o 2 frasi)
        split su quella frase (escludendola da ambo i paragrafi così prodotti)
        aggiunta di tale frase al paragrafo (tra i due) con cui ha maggiore weighted overlap
'''



'''
Codice deprecato


# on class Paragraph


    def get_score_OLD(self):
        """
        DEPRECATED
        :return:
        """
        if self.is_empty():
            return 0
        if self.score >= 0:
            return self.score
        self.score = 0
        # for i, sent1 in self.map_sentence_by_index.items():
        #    for j, sent2 in self.map_sentence_by_index.items():
        for i in range(self.lowest_index_sentence, self.highest_index_sentence + 1):
            for j in range(self.lowest_index_sentence, self.highest_index_sentence + 1):
                if i != j:
                    # self.score += self.cache_pairwise_sentence_similarity.get_similarity_by_sentence_indexes(
                    self.score += self.documentSegmentator.cache_bag_sentence_similarity \
                        .get_similarity_by_sentence_indexes(
                        # self.documentSegmentator.get_bag_of_word_of_sentence_by_index(i),
                        # self.documentSegmentator.get_bag_of_word_of_sentence_by_index(j)
                        i, j
                    )
        self.score /= float(len(self.map_sentence_by_index) * (len(self.map_sentence_by_index) - 1))
        return self.score





#on class DocumentSegmentator
    
    
    def compute_similarity_lists_subsequent_sentences(self, bags_sentences, word_weight_map):
        """
        DEPRECATED
        :param bags_sentences: list of bags of words, extracter from the sentences
        (whose are the parameter of the function "document_segmentation") during the computation
        of the function "get_weighted_synset_map"
        :param word_weight_map: map <string, WordWeighted> that maps the weight of each words
        :return: a list, with length equal to "len(document_segmentation)-1", containing
        holding the similarity score between two subsequent sentences
        (the similarity between sentences in index 0 and 1 is stored in index 0).
        """
        i = 0
        leng = len(bags_sentences) - 1
        simils = [0] * leng
        while i < leng:
            simils[i] = self.cache_bag_sentence_similarity.get_similarity_by_sentence_indexes(bags_sentences[i],
                                                                                              bags_sentences[i + 1])
            i += 1
        return simils


    ... and, in the function "document_segmentation" where it was useful, this code before invoking that function..
'''