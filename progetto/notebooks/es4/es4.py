import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import collections

from notebooks.utilities.cacheVarie import CacheSynsetsBag
from notebooks.utilities.functions import preprocessing, SynsetToBagOptions

'''
import sys
#sys.path.append(".")
#sys.path.append(".\\..\\..\\utilities")

sys.path.append(".\\..\\utilities")
print("what the file:")
print(__file__)
print("let's go ....")
'''

'''
sys.path.append(".\\..\\aaa")
import bbb
bbb.ccc()

if __name__ == '__main__':
    print("main")
    bbb.ccc()
    print("FINE main")
'''

# sys.path.append(".\\..\\")
# from utilities import *
# import utilities
# from utilities import *
# from utilities import cacheVarie
# from ..utilities import *

# import cacheVarie

# from ..utilities import cacheVarie

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))

print("\n\nlet's do it")


def newCache():
    return CacheSynsetsBag()


# ------------
# ------------
# ------------------ func ------------------
# ------------
# ------------


def get_preprocessed_words(text):
    return preprocessing(text)


# return preprocessing(text)


class WeightsSynsetsInfo(object):
    def __init__(self, high=4, medium=2, low=1):
        self.high = high
        self.medium = medium
        self.low = low


WEIGHTS_SYNSET_INFO = WeightsSynsetsInfo()


def get_weighet_synset_map(synset_name, cache=None):
    if cache is None:
        cache = newCache()
    synsets = cache.get_synsets(synset_name)
    if synsets is None:
        return None

    # weight high
    mapped_weights = {synset_name: WEIGHTS_SYNSET_INFO.high}
    for synset in synsets:
        # synonyms
        for synonym in synset.lemmas():
            mapped_weights[synonym.name()] = WEIGHTS_SYNSET_INFO.high

        # medium
        defin = synset.definition()
        defin_refined = preprocessing(defin)
        for def_word in defin_refined:
            mapped_weights[def_word] = WEIGHTS_SYNSET_INFO.medium
        defin_refined = None
        defin = None
        for holon in synset.member_holonyms():  # what about part and substance holonyms? and meronyms?
            mapped_weights[holon.name()] = WEIGHTS_SYNSET_INFO.medium
        for exampl in synset.examples():
            ex_refined = preprocessing(exampl)
            for ex_word in ex_refined:
                mapped_weights[ex_word] = WEIGHTS_SYNSET_INFO.medium

        # low
        for coll in [synset.hypernyms(), synset.hyponyms()]:
            for s in coll:
                mapped_weights[s.name()] = WEIGHTS_SYNSET_INFO.low
    return mapped_weights


def get_weigthed_word_map_for_sentence(sentence, cache=None):
    """
    :param sentence: an English sentence in string variable
    :param cache: a CacheSynsetsBag object or None
    :return: a map <string, float> mapping each non-stop word in the given sentence into a float positive value
    """
    if cache is None:
        cache = newCache()
    sent_filtered = preprocessing(sentence)
    all_weigted_maps = [get_weighet_synset_map(w, cache=cache) for w in sent_filtered]
    sent_filtered = None
    final_weighted_map = {}
    '''
    prima si collezionano tutti i pesi per una data parola della frase
    (differenti parole potrebbero aver ri-trovato la stessa parola in momenti
    diversi, ergo assegnando pesi diversi). poi si calcola una sorta
    di media e la si assegna a quella parola nella mappatura finale
    '''
    weights_collectors = {}
    for wm in all_weigted_maps:
        for wordd, weight in wm.items():
            if wordd in weights_collectors:
                weights_collectors[wordd].append(weight)
            else:
                weights_collectors[wordd] = [weight]
    all_weigted_maps = None
    # some sort of averaging ... like "arithmetic" ones
    for wordd, weights in weights_collectors.items():
        final_weighted_map[wordd] = float(sum(weights) / len(weights))
    return final_weighted_map


def document_segmentation(list_of_sentences, cache=None):
    if cache is None:
        cache = newCache()
    if not (isinstance(list_of_sentences, list)):
        return None
    words_each_sentences = [get_preprocessed_words(sentence) for sentence in list_of_sentences]
    words_counts_and_bags = {}  # set()
    option = SynsetToBagOptions(bag=set())
    # option = SynsetToBagOptions(bag=set())
    for words in words_each_sentences:
        for w in words:
            # all_words.add(w)
            if w in words_counts_and_bags:
                words_counts_and_bags[w][0] += 1
            else:
                words_counts_and_bags[w] = [1, (
                    w, cache.get_extended_bag_for_words(option=option))]  # bag_of_word_expanded

    # TODO: ora usare le frasi e questi bags per computarne le similarità ed eseguire il text tiling
    return words_counts_and_bags


sentences = [
    "I love to pet my cat while reading fantasy books.",
    "Reading books makes my fantasy fly over everyday problems.",
    "One problem of those is my cat vomiting on my pants."
]

'''
wcabs = document_segmentation(sentences)
for word, cab in wcabs.items():
    print(word, " -> (", cab[0], ";;;", cab[1])
'''

print("\n\n now the last made function: get_weigthed_word_map_for_sentence")
local_cache = newCache()
for sent in sentences:
    print("\n\n\n\ngiven the sentence:\n\t--", sentences, "--")
    print(get_weigthed_word_map_for_sentence(sent, cache=local_cache))


'''
result:


 now the last made function: get_weigthed_word_map_for_sentence

given the sentence:
	-- ['I love to pet my cat while reading fantasy books.', 'Reading books makes my fantasy fly over everyday problems.', 'One problem of those is my cat vomiting on my pants.'] --
{'fantasy': 2.0, 'phantasy': 4.0, 'reality': 2.0, 'unrestricte': 2.0, 'imagination': 2.0, 'schoolgirl': 2.0, 'imagination.n.01': 1.0, 'fantasy_life.n.01': 1.0, 'fantasy_world.n.01': 1.0, 'pipe_dream.n.01': 1.0, 'large': 2.0, 'fiction': 2.0, 'write': 2.0, 'romantic': 2.0, 'money': 2.0, 'lot': 2.0, 'fiction.n.01': 1.0, 'science_fiction.n.01': 1.0, 'illusion': 2.0, 'fancy': 4.0, 'believe': 2.0, 'false': 2.0, 'people': 2.0, 'wealthy': 2.0, 'misconception.n.01': 1.0, 'bubble.n.03': 1.0, "will-o'-the-wisp.n.02": 1.0, 'wishful_thinking.n.01': 1.0, 'fantasize': 2.0, 'fantasise': 4.0, 'indulge': 2.0, 'start': 2.0, 'plan': 2.0, 'company': 2.0, 'say': 3.0, 'imagine.v.01': 1.0, 'book': 4.0, 'print': 2.0, 'work': 2.0, 'publish': 2.0, 'composition': 2.0, 'bind': 2.0, 'page': 2.0, 'good': 2.0, 'read': 2.0, 'economic': 2.0, 'publication.n.01': 1.0, 'appointment_book.n.01': 1.0, 'authority.n.07': 1.0, 'bestiary.n.01': 1.0, 'booklet.n.01': 1.0, 'catalog.n.01': 1.0, 'catechism.n.02': 1.0, 'copybook.n.01': 1.0, 'curiosa.n.01': 1.0, 'formulary.n.01': 1.0, 'phrase_book.n.01': 1.0, 'playbook.n.02': 1.0, 'pop-up_book.n.01': 1.0, 'prayer_book.n.01': 1.0, 'reference_book.n.01': 1.0, 'review_copy.n.01': 1.0, 'songbook.n.01': 1.0, 'storybook.n.01': 1.0, 'textbook.n.01': 1.0, 'tome.n.01': 1.0, 'trade_book.n.01': 1.0, 'workbook.n.01': 1.0, 'yearbook.n.01': 1.0, 'volume': 4.0, 'object': 2.0, 'physical': 2.0, 'consist': 2.0, 'number': 2.0, 'doorstop': 2.0, 'product.n.02': 1.0, 'album.n.02': 1.0, 'coffee-table_book.n.01': 1.0, 'folio.n.03': 1.0, 'hardback.n.01': 1.0, 'journal.n.04': 1.0, 'notebook.n.01': 1.0, 'novel.n.02': 1.0, 'order_book.n.02': 1.0, 'paperback_book.n.01': 1.0, 'picture_book.n.01': 1.0, 'sketchbook.n.01': 1.0, 'record': 3.0, 'record_book': 4.0, 'know': 2.0, 'fact': 2.0, 'compilation': 2.0, 'Smith': 2.0, 'Al': 2.0, 'look': 2.0, 'let': 2.0, 'fact.n.02': 1.0, 'card.n.08': 1.0, 'logbook.n.01': 1.0, 'won-lost_record.n.01': 1.0, 'script': 4.0, 'playscript': 4.0, 'performance': 2.0, 'version': 2.0, 'play': 2.0, 'dramatic': 2.0, 'prepare': 2.0, 'dramatic_composition.n.01': 1.0, 'continuity.n.02': 1.0, 'dialogue.n.02': 1.0, 'libretto.n.01': 1.0, 'promptbook.n.01': 1.0, 'scenario.n.01': 1.0, 'screenplay.n.01': 1.0, 'shooting_script.n.01': 1.0, 'ledger': 4.0, 'leger': 4.0, 'account_book': 4.0, 'book_of_account': 4.0, 'account': 2.0, 'commercial': 2.0, 'examine': 2.0, 'get': 2.0, 'subpoena': 2.0, 'record.n.07': 1.0, 'cost_ledger.n.01': 1.0, 'daybook.n.01': 1.0, 'general_ledger.n.01': 1.0, 'subsidiary_ledger.n.01': 1.0, 'rule': 2.0, 'game': 2.0, 'card': 2.0, 'satisfy': 2.0, 'collection': 2.0, 'collection.n.01': 1.0, 'rule_book': 4.0, 'basis': 2.0, 'prescribe': 2.0, 'standard': 2.0, 'decision': 2.0, 'run': 2.0, 'thing': 2.0, 'Koran': 4.0, 'Quran': 4.0, "al-Qur'an": 4.0, 'Book': 4.0, 'Muhammad': 2.0, 'writing': 2.0, 'sacred': 2.0, 'prophet': 2.0, 'life': 2.0, 'God': 2.0, 'reveal': 2.0, 'Mecca': 2.0, 'Medina': 2.0, 'Islam': 2.0, 'Bible': 4.0, 'Christian_Bible': 4.0, 'Good_Book': 4.0, 'Holy_Scripture': 4.0, 'Holy_Writ': 4.0, 'Scripture': 4.0, 'Word_of_God': 4.0, 'Word': 2.0, 'religion': 2.0, 'christian': 2.0, 'heathen': 2.0, 'carry': 2.0, 'go': 2.0, 'sacred_text.n.01': 1.0, 'family_bible.n.01': 1.0, 'major': 2.0, 'division': 2.0, 'long': 2.0, 'Isaiah': 2.0, 'section.n.01': 1.0, 'epistle.n.02': 1.0, 'etc': 2.0, 'ticket': 2.0, 'sheet': 2.0, 'edge': 2.0, 'stamp': 2.0, 'buy': 2.0, 'engage': 2.0, 'concert': 2.0, 'Tokyo': 2.0, 'agent': 2.0, 'schedule.v.01': 1.0, 'reserve': 2.0, 'hold': 2.0, 'arrange': 2.0, 'advance': 2.0, 'seat': 2.0, 'flight': 2.0, 'family': 2.0, 'table': 2.0, 'Maxim': 2.0, 'request.v.01': 1.0, 'keep_open.v.01': 1.0, 'charge': 2.0, 'register': 3.0, 'police': 2.0, 'policeman': 2.0, 'man': 2.0, 'solicit': 2.0, 'try': 2.0, 'record.v.01': 1.0, 'ticket.v.01': 1.0, 'booker': 2.0, 'hotel': 2.0, 'register.v.01': 1.0, 'article': 2.0, 'interpret': 4.0, 'advertisement': 2.0, 'Salman': 2.0, 'Rushdie': 2.0, 'interpret.v.01': 1.0, 'anagram.v.01': 1.0, 'decipher.v.02': 1.0, 'dip_into.v.01': 1.0, 'lipread.v.01': 1.0, 'reread.v.01': 1.0, 'skim.v.07': 1.0, 'wording': 2.0, 'certain': 2.0, 'contain': 2.0, 'form': 2.0, 'follow': 2.0, 'passage': 2.0, 'law': 2.0, 'have.v.02': 1.0, 'loud': 2.0, 'proclamation': 2.0, 'noon': 2.0, 'King': 2.0, 'talk.v.02': 1.0, 'call.v.08': 1.0, 'dictate.v.02': 1.0, 'numerate.v.02': 1.0, 'scan': 3.0, 'obtain': 2.0, 'magnetic': 2.0, 'tape': 2.0, 'datum': 2.0, 'computer': 2.0, 'dictionary': 2.0, 'misread.v.01': 1.0, 'leave': 2.0, 'palm': 2.0, 'human': 2.0, 'intestine': 2.0, 'behavior': 2.0, 'sky': 2.0, 'tea': 2.0, 'significance': 2.0, 'rain': 2.0, 'predict': 2.0, 'strange': 2.0, 'fortune': 2.0, 'ball': 2.0, 'fate': 2.0, 'teller': 2.0, 'crystal': 2.0, 'predict.v.01': 1.0, 'scry.v.01': 1.0, 'take': 4.0, 'impression': 2.0, 'meaning': 2.0, 'convey': 2.0, 'way': 2.0, 'particular': 2.0, 'satire': 2.0, 'address': 2.0, 'message': 2.0, 'credit': 2.0, 'misread.v.02': 1.0, 'learn': 4.0, 'study': 4.0, 'subject': 2.0, 'student': 2.0, 'exam': 2.0, 'bar': 2.0, 'audit.v.02': 1.0, 'drill.v.03': 1.0, 'train.v.02': 1.0, 'show': 2.0, 'instrument': 2.0, 'reading': 2.0, 'gauge': 2.0, 'indicate': 2.0, 'degree': 2.0, 'thirteen': 2.0, 'zero': 2.0, 'thermometer': 2.0, 'indicate.v.03': 1.0, 'say.v.11': 1.0, 'show.v.10': 1.0, 'strike.v.05': 1.0, 'audition': 2.0, 'role': 2.0, 'stage': 2.0, 'part': 2.0, 'Caesar': 2.0, 'Stratford': 2.0, 'year': 2.0, 'Julius': 2.0, 'audition.v.01': 1.0, 'understand': 2.0, 'hear': 2.0, 'clear': 2.0, 'understand.v.01': 1.0, 'translate': 4.0, 'language': 2.0, 'sense': 2.0, 'French': 2.0, 'Greek': 2.0, 'pet': 4.0, 'companionship': 2.0, 'domesticate': 2.0, 'amusement': 2.0, 'animal': 2.0, 'keep': 2.0, 'animal.n.01': 1.0, 'darling': 4.0, 'favorite': 4.0, 'favourite': 4.0, 'dearie': 4.0, 'deary': 4.0, 'ducky': 4.0, 'love': 3.0, 'special': 2.0, 'lover.n.01': 1.0, 'chosen.n.01': 1.0, 'macushla.n.01': 1.0, 'mollycoddle.n.01': 1.0, "teacher's_pet.n.01": 1.0, 'especially': 2.0, 'feel': 2.0, 'slight': 2.0, 'sulkiness': 2.0, 'petulance': 2.0, 'fit': 2.0, 'irritability.n.01': 1.0, 'positron_emission_tomography': 4.0, 'PET': 4.0, 'technique': 2.0, 'tissue': 2.0, 'brain': 2.0, 'metabolic': 2.0, 'computerized': 2.0, 'radiographic': 2.0, 'activity': 2.0, 'imaging.n.02': 1.0, 'caress': 2.0, 'stroke': 2.0, 'gently': 2.0, 'lamb': 2.0, 'caress.v.01': 1.0, 'canoodle.v.01': 1.0, 'gentle.v.03': 1.0, 'neck.v.01': 1.0, 'lovemaking': 2.0, 'erotic': 2.0, 'manner': 2.0, 'favored': 4.0, 'best-loved': 4.0, 'preferred': 4.0, 'preferent': 4.0, 'treat': 2.0, 'partiality': 2.0, 'prefer': 2.0, 'child': 2.0, 'favor': 2.0, 'positive': 2.0, 'emotion': 2.0, 'affection': 2.0, 'regard': 2.0, 'strong': 2.0, 'need': 2.0, 'emotion.n.01': 1.0, 'agape.n.01': 1.0, 'agape.n.02': 1.0, 'amorousness.n.01': 1.0, 'ardor.n.02': 1.0, 'benevolence.n.01': 1.0, 'devotion.n.01': 1.0, 'filial_love.n.01': 1.0, 'heartstrings.n.01': 1.0, 'lovingness.n.01': 1.0, 'loyalty.n.02': 1.0, 'puppy_love.n.01': 1.0, 'worship.n.02': 1.0, 'passion': 2.0, 'warm': 2.0, 'devotion': 2.0, 'theater': 2.0, 'cock': 2.0, 'fighting': 2.0, 'object.n.04': 1.0, 'beloved': 2.0, 'dear': 4.0, 'dearest': 4.0, 'honey': 4.0, 'term': 2.0, 'person': 2.0, 'endearment': 2.0, 'sexual_love': 4.0, 'erotic_love': 4.0, 'deep': 2.0, 'sexual': 2.0, 'desire': 2.0, 'attraction': 2.0, 'feeling': 2.0, 'indifferent': 2.0, 'surrounding': 2.0, 'sexual_desire.n.01': 1.0, 'squash': 2.0, 'tennis': 2.0, 'score': 2.0, '40': 2.0, 'score.n.03': 1.0, 'making_love': 4.0, 'love_life': 4.0, 'include': 2.0, 'intercourse': 2.0, 'disgust': 2.0, 'month': 2.0, 'complicated': 2.0, 'sexual_activity.n.01': 1.0, 'great': 2.0, 'like': 2.0, 'french': 2.0, 'food': 2.0, 'hard': 2.0, 'boss': 2.0, 'adore.v.01': 1.0, 'care_for.v.02': 1.0, 'dote.v.02': 1.0, 'love.v.03': 1.0, 'enjoy': 4.0, 'pleasure': 2.0, 'cooking': 2.0, 'like.v.02': 1.0, 'get_off.v.06': 1.0, 'enamor': 2.0, 'deeply': 2.0, 'husband': 2.0, 'love.v.01': 1.0, 'romance.v.02': 1.0, 'sleep_together': 4.0, 'roll_in_the_hay': 4.0, 'make_out': 4.0, 'make_love': 4.0, 'sleep_with': 4.0, 'get_laid': 4.0, 'have_sex': 4.0, 'do_it': 4.0, 'be_intimate': 4.0, 'have_intercourse': 4.0, 'have_it_away': 4.0, 'have_it_off': 4.0, 'screw': 4.0, 'fuck': 4.0, 'jazz': 4.0, 'eff': 4.0, 'hump': 4.0, 'lie_with': 4.0, 'bed': 4.0, 'have_a_go_at_it': 4.0, 'bang': 4.0, 'get_it_on': 4.0, 'bonk': 4.0, 'sleep': 2.0, 'dorm': 2.0, 'Adam': 2.0, 'Eve': 2.0, 'intimate': 2.0, 'copulate.v.01': 1.0, 'fornicate.v.01': 1.0, 'take.v.35': 1.0, 'cat': 4.0, 'true_cat': 4.0, 'have': 2.0, 'soft': 2.0, 'fur': 2.0, 'mammal': 2.0, 'usually': 2.0, 'feline': 2.0, 'wildcat': 2.0, 'ability': 2.0, 'roar': 2.0, 'thick': 2.0, 'domestic': 2.0, 'feline.n.01': 1.0, 'domestic_cat.n.01': 1.0, 'wildcat.n.03': 1.0, 'guy': 2.0, 'hombre': 4.0, 'bozo': 4.0, 'informal': 2.0, 'youth': 2.0, 'nice': 2.0, 'doll': 2.0, 'man.n.01': 1.0, 'sod.n.04': 1.0, 'woman': 2.0, 'spiteful': 2.0, 'gossip': 2.0, 'gossip.n.03': 1.0, 'woman.n.01': 1.0, 'kat': 2.0, 'khat': 4.0, 'qat': 4.0, 'quat': 4.0, 'Arabian_tea': 4.0, 'African_tea': 4.0, 'effect': 2.0, 'stimulant': 2.0, 'shrub': 2.0, 'tobacco': 2.0, 'eduli': 2.0, 'euphoric': 2.0, 'Catha': 2.0, 'chew': 2.0, 'adult': 2.0, '%': 2.0, '85': 2.0, 'Yemen': 2.0, 'daily': 2.0, 'stimulant.n.02': 1.0, "cat-o'-nine-tails": 4.0, 'knotted': 2.0, 'whip': 2.0, 'cord': 2.0, 'fear': 2.0, 'sailor': 2.0, 'british': 2.0, 'whip.n.01': 1.0, 'Caterpillar': 4.0, 'construction': 2.0, 'track': 2.0, 'frequently': 2.0, 'propel': 2.0, 'move': 2.0, 'vehicle': 2.0, 'endless': 2.0, 'belt': 2.0, 'earth': 2.0, 'metal': 2.0, 'farm': 2.0, 'tracked_vehicle.n.01': 1.0, 'big_cat': 4.0, 'wild': 2.0, 'live': 2.0, 'able': 2.0, 'typically': 2.0, 'felidae.n.01': 2.0, 'cheetah.n.01': 1.0, 'jaguar.n.01': 1.0, 'leopard.n.02': 1.0, 'liger.n.01': 1.0, 'lion.n.01': 1.0, 'saber-toothed_tiger.n.01': 1.0, 'snow_leopard.n.01': 1.0, 'tiger.n.02': 1.0, 'tiglon.n.01': 1.0, 'computerized_tomography': 4.0, 'computed_tomography': 4.0, 'CT': 4.0, 'computerized_axial_tomography': 4.0, 'computed_axial_tomography': 4.0, 'CAT': 4.0, '-': 2.0, 'ray': 2.0, 'method': 2.0, 'axis': 2.0, 'organ': 2.0, 'cross': 2.0, 'body': 2.0, 'series': 2.0, 'construct': 2.0, 'single': 2.0, 'sectional': 2.0, 'x': 2.0, 'x-raying.n.01': 1.0, "o'-nine": 2.0, 'beat': 2.0, 'tail': 2.0, 'flog.v.01': 1.0, 'vomit': 2.0, 'vomit_up': 4.0, 'purge': 2.0, 'cast': 4.0, 'sick': 4.0, 'be_sick': 4.0, 'disgorge': 4.0, 'regorge': 4.0, 'retch': 4.0, 'puke': 4.0, 'barf': 4.0, 'spew': 4.0, 'spue': 4.0, 'chuck': 4.0, 'upchuck': 4.0, 'honk': 4.0, 'regurgitate': 2.0, 'throw_up': 4.0, 'stomach': 2.0, 'eject': 2.0, 'content': 2.0, 'mouth': 2.0, 'drink': 2.0, 'continuously': 2.0, 'give': 2.0, 'night': 2.0, 'patient': 2.0, 'excrete.v.01': 1.0}




given the sentence:
	-- ['I love to pet my cat while reading fantasy books.', 'Reading books makes my fantasy fly over everyday problems.', 'One problem of those is my cat vomiting on my pants.'] --
{'everyday': 2.0, 'mundane': 4.0, 'quotidian': 2.0, 'routine': 2.0, 'unremarkable': 4.0, 'workaday': 4.0, 'Anita': 2.0, 'course': 2.0, 'diamant': 2.0, 'event': 2.0, 'ordinary': 2.0, 'find': 2.0, 'scene': 2.0, 'placid': 2.0, 'day': 2.0, 'color': 2.0, 'add': 2.0, 'commute': 2.0, 'real': 2.0, 'like': 2.0, 'conductor': 2.0, 'train': 2.0, 'casual': 2.0, 'daily': 4.0, 'appropriate': 2.0, 'occasion': 2.0, 'clothe': 2.0, 'commonplace': 2.0, 'world': 2.0, 'familiar': 2.0, 'fantasy': 2.0, 'phantasy': 4.0, 'reality': 2.0, 'unrestricte': 2.0, 'imagination': 2.0, 'schoolgirl': 2.0, 'imagination.n.01': 1.0, 'fantasy_life.n.01': 1.0, 'fantasy_world.n.01': 1.0, 'pipe_dream.n.01': 1.0, 'large': 2.0, 'fiction': 2.0, 'write': 2.0, 'romantic': 2.0, 'money': 2.0, 'lot': 2.0, 'fiction.n.01': 1.0, 'science_fiction.n.01': 1.0, 'illusion': 2.0, 'fancy': 4.0, 'believe': 2.0, 'false': 2.0, 'people': 2.0, 'wealthy': 2.0, 'misconception.n.01': 1.0, 'bubble.n.03': 1.0, "will-o'-the-wisp.n.02": 1.0, 'wishful_thinking.n.01': 1.0, 'fantasize': 2.0, 'fantasise': 4.0, 'indulge': 2.0, 'start': 2.0, 'plan': 2.0, 'company': 2.0, 'say': 2.6666666666666665, 'imagine.v.01': 1.0, 'book': 4.0, 'print': 2.0, 'work': 2.0, 'publish': 2.0, 'composition': 2.0, 'bind': 2.0, 'page': 2.0, 'good': 2.0, 'read': 2.0, 'economic': 2.0, 'publication.n.01': 1.0, 'appointment_book.n.01': 1.0, 'authority.n.07': 1.0, 'bestiary.n.01': 1.0, 'booklet.n.01': 1.0, 'catalog.n.01': 1.0, 'catechism.n.02': 1.0, 'copybook.n.01': 1.0, 'curiosa.n.01': 1.0, 'formulary.n.01': 1.0, 'phrase_book.n.01': 1.0, 'playbook.n.02': 1.0, 'pop-up_book.n.01': 1.0, 'prayer_book.n.01': 1.0, 'reference_book.n.01': 1.0, 'review_copy.n.01': 1.0, 'songbook.n.01': 1.0, 'storybook.n.01': 1.0, 'textbook.n.01': 1.0, 'tome.n.01': 1.0, 'trade_book.n.01': 1.0, 'workbook.n.01': 1.0, 'yearbook.n.01': 1.0, 'volume': 4.0, 'object': 2.0, 'physical': 2.0, 'consist': 2.0, 'number': 2.0, 'doorstop': 2.0, 'product.n.02': 1.0, 'album.n.02': 1.0, 'coffee-table_book.n.01': 1.0, 'folio.n.03': 1.0, 'hardback.n.01': 1.0, 'journal.n.04': 1.0, 'notebook.n.01': 1.0, 'novel.n.02': 1.0, 'order_book.n.02': 1.0, 'paperback_book.n.01': 1.0, 'picture_book.n.01': 1.0, 'sketchbook.n.01': 1.0, 'record': 3.0, 'record_book': 4.0, 'know': 2.0, 'fact': 2.0, 'compilation': 2.0, 'Smith': 2.0, 'Al': 2.0, 'look': 2.0, 'let': 2.0, 'fact.n.02': 1.0, 'card.n.08': 1.0, 'logbook.n.01': 1.0, 'won-lost_record.n.01': 1.0, 'script': 4.0, 'playscript': 4.0, 'performance': 2.0, 'version': 2.0, 'play': 2.0, 'dramatic': 2.0, 'prepare': 2.0, 'dramatic_composition.n.01': 1.0, 'continuity.n.02': 1.0, 'dialogue.n.02': 1.0, 'libretto.n.01': 1.0, 'promptbook.n.01': 1.0, 'scenario.n.01': 1.0, 'screenplay.n.01': 1.0, 'shooting_script.n.01': 1.0, 'ledger': 4.0, 'leger': 4.0, 'account_book': 4.0, 'book_of_account': 4.0, 'account': 2.0, 'commercial': 2.0, 'examine': 2.0, 'get': 2.0, 'subpoena': 2.0, 'record.n.07': 1.0, 'cost_ledger.n.01': 1.0, 'daybook.n.01': 1.0, 'general_ledger.n.01': 1.0, 'subsidiary_ledger.n.01': 1.0, 'rule': 2.0, 'game': 2.0, 'card': 2.0, 'satisfy': 2.0, 'collection': 2.0, 'collection.n.01': 1.0, 'rule_book': 4.0, 'basis': 2.0, 'prescribe': 2.0, 'standard': 2.0, 'decision': 2.0, 'run': 2.0, 'thing': 2.0, 'Koran': 4.0, 'Quran': 4.0, "al-Qur'an": 4.0, 'Book': 4.0, 'Muhammad': 2.0, 'writing': 2.0, 'sacred': 2.0, 'prophet': 2.0, 'life': 2.0, 'God': 2.0, 'reveal': 2.0, 'Mecca': 2.0, 'Medina': 2.0, 'Islam': 2.0, 'Bible': 4.0, 'Christian_Bible': 4.0, 'Good_Book': 4.0, 'Holy_Scripture': 4.0, 'Holy_Writ': 4.0, 'Scripture': 4.0, 'Word_of_God': 4.0, 'Word': 2.0, 'religion': 2.0, 'christian': 2.0, 'heathen': 2.0, 'carry': 2.0, 'go': 2.0, 'sacred_text.n.01': 1.0, 'family_bible.n.01': 1.0, 'major': 2.0, 'division': 2.0, 'long': 2.0, 'Isaiah': 2.0, 'section.n.01': 1.0, 'epistle.n.02': 1.0, 'etc': 2.0, 'ticket': 2.0, 'sheet': 2.0, 'edge': 2.0, 'stamp': 2.0, 'buy': 2.0, 'engage': 2.0, 'concert': 2.0, 'Tokyo': 2.0, 'agent': 2.0, 'schedule.v.01': 1.0, 'reserve': 2.0, 'hold': 2.0, 'arrange': 2.0, 'advance': 2.0, 'seat': 2.0, 'flight': 2.0, 'family': 2.0, 'table': 2.0, 'Maxim': 2.0, 'request.v.01': 1.0, 'keep_open.v.01': 1.0, 'charge': 2.0, 'register': 3.0, 'police': 2.0, 'policeman': 2.0, 'man': 2.0, 'solicit': 2.0, 'try': 2.0, 'record.v.01': 1.0, 'ticket.v.01': 1.0, 'booker': 2.0, 'hotel': 2.0, 'register.v.01': 1.0, 'article': 2.0, 'interpret': 4.0, 'advertisement': 2.0, 'Salman': 2.0, 'Rushdie': 2.0, 'interpret.v.01': 1.0, 'anagram.v.01': 1.0, 'decipher.v.02': 1.0, 'dip_into.v.01': 1.0, 'lipread.v.01': 1.0, 'reread.v.01': 1.0, 'skim.v.07': 1.0, 'wording': 2.0, 'certain': 2.0, 'contain': 2.0, 'form': 2.0, 'follow': 2.0, 'passage': 2.0, 'law': 2.0, 'have.v.02': 1.0, 'loud': 2.0, 'proclamation': 2.0, 'noon': 2.0, 'King': 2.0, 'talk.v.02': 1.0, 'call.v.08': 1.0, 'dictate.v.02': 1.0, 'numerate.v.02': 1.0, 'scan': 4.0, 'obtain': 2.0, 'magnetic': 2.0, 'tape': 2.0, 'datum': 2.0, 'computer': 2.0, 'dictionary': 2.0, 'misread.v.01': 1.0, 'leave': 2.0, 'palm': 2.0, 'human': 2.0, 'intestine': 2.0, 'behavior': 2.0, 'sky': 2.0, 'tea': 2.0, 'significance': 2.0, 'rain': 2.0, 'predict': 2.0, 'strange': 2.0, 'fortune': 2.0, 'ball': 2.0, 'fate': 2.0, 'teller': 2.0, 'crystal': 2.0, 'predict.v.01': 1.0, 'scry.v.01': 1.0, 'take': 3.0, 'impression': 2.0, 'meaning': 2.0, 'convey': 2.0, 'way': 2.0, 'particular': 2.0, 'satire': 2.0, 'address': 2.0, 'message': 2.0, 'credit': 2.0, 'misread.v.02': 1.0, 'learn': 4.0, 'study': 4.0, 'subject': 2.0, 'student': 2.0, 'exam': 2.0, 'bar': 2.0, 'audit.v.02': 1.0, 'drill.v.03': 1.0, 'train.v.02': 1.0, 'show': 2.0, 'instrument': 2.0, 'reading': 2.0, 'gauge': 2.0, 'indicate': 2.0, 'degree': 2.0, 'thirteen': 2.0, 'zero': 2.0, 'thermometer': 2.0, 'indicate.v.03': 1.0, 'say.v.11': 1.0, 'show.v.10': 1.0, 'strike.v.05': 1.0, 'audition': 2.0, 'role': 2.0, 'stage': 2.0, 'part': 2.0, 'Caesar': 2.0, 'Stratford': 2.0, 'year': 2.0, 'Julius': 2.0, 'audition.v.01': 1.0, 'understand': 2.0, 'hear': 2.0, 'clear': 2.0, 'understand.v.01': 1.0, 'translate': 4.0, 'language': 2.0, 'sense': 2.0, 'French': 2.0, 'Greek': 2.0, 'make': 4.0, 'brand': 2.0, 'recognizable': 2.0, 'kind': 2.0, 'hero': 2.0, 'new': 2.0, 'movie': 2.0, 'car': 2.0, 'kind.n.01': 1.0, 'shuffle': 4.0, 'shuffling': 4.0, 'haphazardly': 2.0, 'mix': 2.0, 'act': 2.0, 'reordering.n.01': 1.0, 'reshuffle.n.02': 1.0, 'riffle.n.02': 1.0, 'do': 4.0, 'love': 2.0, 'war': 2.0, 'effort': 2.0, 'research': 2.0, 'revolution': 2.0, 'overdo.v.01': 1.0, 'property': 2.0, 'mad': 2.0, 'silly': 2.0, 'fool': 2.0, 'meeting': 2.0, 'deal': 2.0, 'big': 2.0, 'millionaire': 2.0, 'invention': 2.0, 'change.v.01': 1.0, 'get.v.03': 1.0, 'leave.v.03': 1.0, 'render.v.01': 1.0, 'create': 2.0, 'cause': 2.0, 'office': 2.0, 'mess': 2.0, 'furor': 2.0, 'arouse.v.01': 1.0, 'assemble.v.01': 1.0, 'bear.v.05': 1.0, 'beat.v.18': 1.0, 'beget.v.01': 1.0, 'blast.v.05': 1.0, 'bring.v.03': 1.0, 'build.v.03': 1.0, 'cause.v.01': 1.0, 'chop.v.03': 1.0, 'choreograph.v.01': 1.0, 'clear.v.02': 1.0, 'cleave.v.02': 1.0, 'compose.v.02': 1.0, 'construct.v.01': 1.0, 'copy.v.04': 1.0, 'create.v.05': 1.0, 'create_by_mental_act.v.01': 1.0, 'create_from_raw_material.v.01': 1.0, 'create_verbally.v.01': 1.0, 'cut.v.06': 1.0, 'cut.v.22': 1.0, 'derive.v.04': 1.0, 'direct.v.03': 1.0, 'distill.v.03': 1.0, 'establish.v.05': 1.0, 'film-make.v.01': 1.0, 'film.v.02': 1.0, 'form.v.01': 1.0, 'froth.v.02': 1.0, 'generate.v.01': 1.0, 'give.v.09': 1.0, 'grind.v.06': 1.0, 'incorporate.v.03': 1.0, 'institute.v.02': 1.0, 'lay_down.v.01': 1.0, 'manufacture.v.04': 1.0, 'offset.v.04': 1.0, 'originate.v.02': 1.0, 'prepare.v.03': 1.0, 'press.v.07': 1.0, 'produce.v.01': 1.0, 'produce.v.03': 1.0, 'puncture.v.02': 1.0, 'put_on.v.04': 1.0, 'raise.v.07': 1.0, 'raise.v.11': 1.0, 're-create.v.01': 1.0, 'realize.v.03': 1.0, 'recreate.v.04': 1.0, 'regenerate.v.07': 1.0, 'reproduce.v.02': 1.0, 'scrape.v.02': 1.0, 'short-circuit.v.02': 1.0, 'strike.v.13': 1.0, 'style.v.02': 1.0, 'track.v.05': 1.0, 'twine.v.03': 1.0, 'induce': 2.0, 'stimulate': 4.0, 'have': 3.0, 'manner': 2.0, 'specify': 2.0, 'vcr': 2.0, 'ad': 2.0, 'child': 2.0, 'finally': 2.0, 'sofa': 2.0, 'wife': 2.0, 'bring.v.11': 1.0, 'compel.v.01': 1.0, 'decide.v.03': 1.0, 'encourage.v.03': 1.0, 'lead.v.05': 1.0, 'let.v.02': 1.0, 'persuade.v.02': 1.0, 'prompt.v.02': 1.0, 'solicit.v.04': 1.0, 'suborn.v.03': 1.0, 'rise': 2.0, 'happen': 2.0, 'occur': 2.0, 'intentionally': 2.0, 'commotion': 2.0, 'stir': 2.0, 'accident': 2.0, 'make.v.03': 1.0, 'determine.v.02': 1.0, 'effect.v.01': 1.0, 'engender.v.01': 1.0, 'facilitate.v.03': 1.0, 'impel.v.01': 1.0, 'initiate.v.02': 1.0, 'make.v.08': 1.0, 'motivate.v.01': 1.0, 'occasion.v.01': 1.0, 'provoke.v.02': 1.0, 'produce': 2.0, 'manufacture': 2.0, 'product': 2.0, 'sell': 2.0, 'toy': 2.0, 'century': 2.0, 'bootleg.v.02': 1.0, 'breed.v.03': 1.0, 'churn_out.v.02': 1.0, 'clap_up.v.01': 1.0, 'confect.v.01': 1.0, 'custom-make.v.01': 1.0, 'cut.v.21': 1.0, 'dummy.v.01': 1.0, 'elaborate.v.02': 1.0, 'extrude.v.01': 1.0, 'fudge_together.v.01': 1.0, 'generate.v.03': 1.0, 'laminate.v.01': 1.0, 'machine.v.02': 1.0, 'output.v.01': 1.0, 'overproduce.v.02': 1.0, 'prefabricate.v.01': 1.0, 'prefabricate.v.02': 1.0, 'print.v.01': 1.0, 'proof.v.01': 1.0, 'pulse.v.02': 1.0, 'put_out.v.02': 1.0, 'remake.v.01': 1.0, 'render.v.04': 1.0, 'reproduce.v.01': 1.0, 'smelt.v.01': 1.0, 'turn_out.v.03': 1.0, 'underproduce.v.01': 1.0, 'draw': 2.0, 'mind': 2.0, 'formulate': 2.0, 'derive': 2.0, 'line': 2.0, 'conclusion': 2.0, 'parallel': 2.0, 'estimate': 2.0, 'remark': 2.0, 'compel': 2.0, 'somebody': 2.0, 'pass': 2.0, 'integrate': 2.0, 'People': 2.0, 'sweat': 2.0, 'heat': 2.0, 'drive.v.07': 1.0, 'mean': 2.0, 'artistic': 2.0, 'poem': 2.0, 'tone': 2.0, 'Schoenberg': 2.0, 'music': 2.0, 'Picasso': 2.0, 'Cubism': 2.0, 'verse': 2.0, 'Auden': 2.0, 'design.v.03': 1.0, 'design.v.05': 1.0, 'do.v.08': 1.0, 'gain': 4.0, 'take_in': 4.0, 'earn': 2.0, 'realize': 4.0, 'realise': 4.0, 'pull_in': 4.0, 'bring_in': 4.0, 'wage': 2.0, 'transaction': 2.0, 'business': 2.0, 'salary': 2.0, 'month': 2.0, 'job': 2.0, 'bring': 2.0, 'merger': 2.0, '5,000': 2.0, '$': 2.0, 'get.v.01': 1.0, 'eke_out.v.03': 1.0, 'gross.v.01': 1.0, 'profit.v.02': 1.0, 'rake_in.v.01': 1.0, 'rake_off.v.01': 1.0, 'take_home.v.01': 1.0, 'yield.v.10': 1.0, 'design': 2.0, 'room': 2.0, 'blue': 2.0, 'express': 2.0, 'wood': 2.0, 'forest': 2.0, 'piece': 2.0, 'constitute': 2.0, 'represent': 2.0, 'compose': 2.0, 'background': 2.0, 'setting': 2.0, 'wall': 2.0, 'roof': 2.0, 'branch': 2.0, 'fine': 2.0, 'introduction': 2.0, 'constitute.v.01': 1.0, 'add.v.06': 1.0, 'chelate.v.01': 1.0, 'reach': 2.0, 'get_to': 4.0, 'progress_to': 4.0, 'e.g.': 2.0, 'goal': 2.0, 'team': 2.0, 'grade': 2.0, 'achieve.v.01': 1.0, 'capable': 2.0, 'change': 2.0, 'host': 2.0, 'great': 2.0, 'father': 2.0, 'become.v.03': 1.0, 'shape': 2.0, 'constituent': 2.0, 'dress': 2.0, 'cake': 2.0, 'stone': 2.0, 'compile.v.03': 1.0, 'compose.v.04': 1.0, 'cooper.v.01': 1.0, 'fashion.v.01': 1.0, 'manufacture.v.01': 1.0, 'perform': 2.0, 'phone': 2.0, 'perform.v.01': 1.0, 'pay.v.08': 1.0, 'construct': 2.0, 'build': 4.0, 'material': 2.0, 'combine': 2.0, 'pig': 2.0, 'house': 2.0, 'straw': 2.0, 'little': 2.0, 'electric': 2.0, 'eccentric': 2.0, 'brassiere': 2.0, 'warm': 2.0, 'cantilever.v.02': 1.0, 'channelize.v.02': 1.0, 'corduroy.v.01': 1.0, 'customize.v.02': 1.0, 'dry-wall.v.01': 1.0, 'frame.v.06': 1.0, 'groin.v.01': 1.0, 'lock.v.09': 1.0, 'raise.v.09': 1.0, 'rebuild.v.01': 1.0, 'revet.v.01': 1.0, 'wattle.v.01': 1.0, 'water': 2.0, 'wine': 2.0, 'lead': 2.0, 'gold': 2.0, 'clay': 2.0, 'brick': 2.0, 'acquire': 2.0, 'friend': 2.0, 'enemy': 2.0, 'act.v.02': 1.0, 'name': 2.0, 'nominate': 4.0, 'function': 2.0, 'Head': 2.0, 'Committee': 2.0, 'president': 2.0, 'club': 2.0, 'appoint.v.02': 1.0, 'rename.v.02': 1.0, 'point': 2.0, 'achieve': 2.0, '70': 2.0, 'Nicklaus': 2.0, 'brazilian': 2.0, '4': 2.0, '29': 2.0, 'score.v.01': 1.0, 'attain': 4.0, 'hit': 2.0, 'arrive_at': 4.0, 'abstract': 2.0, 'destination': 2.0, 'Detroit': 2.0, 'doorstep': 2.0, 'barely': 2.0, 'finish': 2.0, 'MAC': 2.0, 'machine': 2.0, 'weekend': 2.0, 'access.v.02': 1.0, 'bottom_out.v.01': 1.0, 'catch_up.v.01': 1.0, 'culminate.v.04': 1.0, 'find.v.15': 1.0, 'get_through.v.03': 1.0, 'ground.v.06': 1.0, 'make.v.37': 1.0, 'scale.v.04': 1.0, 'summit.v.01': 1.0, 'top.v.06': 1.0, 'top_out.v.03': 1.0, 'lay_down': 4.0, 'establish': 2.0, 'institute': 2.0, 'enact': 2.0, 'set.v.04': 1.0, 'commit': 2.0, 'mistake': 2.0, 'faux': 2.0, 'pas': 2.0, 'perpetrate.v.01': 1.0, 'assemble': 2.0, 'individual': 2.0, 'quorum': 2.0, 'assemble.v.03': 1.0, 'throw': 2.0, 'give': 4.0, 'responsible': 2.0, 'organize': 2.0, 'reception': 2.0, 'party': 2.0, 'direct.v.04': 1.0, 'make_up': 4.0, 'order': 2.0, 'neaten': 2.0, 'bed': 2.0, 'tidy.v.01': 1.0, 'head': 2.0, 'direction': 2.0, 'convict': 2.0, 'escaped': 2.0, 'hill': 2.0, 'mountain': 2.0, 'head.v.01': 1.0, 'stool': 4.0, 'defecate': 4.0, 'shit': 4.0, 'take_a_shit': 4.0, 'take_a_crap': 4.0, 'ca-ca': 4.0, 'crap': 4.0, 'movement': 2.0, 'bowel': 2.0, 'dog': 2.0, 'flower': 2.0, 'excrete.v.01': 1.0, 'dung.v.02': 1.0, 'fabrication': 2.0, 'creation': 2.0, 'undergo': 2.0, 'nice': 2.0, 'sweater': 2.0, 'wool': 2.0, 'change.v.02': 1.0, 'suitable': 2.0, 'furniture': 2.0, 'be.v.01': 1.0, 'total.v.01': 1.0, 'difference': 2.0, 'living': 2.0, 'increase': 2.0, 'amount.v.01': 1.0, 'essence': 2.0, 'begin': 2.0, 'appear': 2.0, 'activity': 2.0, 'speak': 2.0, 'end': 2.0, 'hello': 2.0, 'look.v.02': 1.0, 'proceed': 2.0, 'path': 2.0, 'crowd': 2.0, 'pass.v.01': 1.0, 'bushwhack.v.03': 1.0, 'claw.v.01': 1.0, 'jostle.v.01': 1.0, 'time': 2.0, 'plane': 2.0, 'reach.v.01': 1.0, 'gather': 2.0, 'light': 2.0, 'fire': 2.0, 'cook': 2.0, 'fix': 2.0, 'ready': 4.0, 'apply': 2.0, 'eat': 2.0, 'dinner': 2.0, 'omelette': 2.0, 'breakfast': 2.0, 'guest': 2.0, 'concoct.v.02': 1.0, 'deglaze.v.01': 1.0, 'devil.v.02': 1.0, 'dress.v.06': 1.0, 'flambe.v.01': 1.0, 'lard.v.01': 1.0, 'precook.v.01': 1.0, 'preserve.v.04': 1.0, 'put_on.v.03': 1.0, 'scallop.v.02': 1.0, 'whip_up.v.01': 1.0, 'seduce': 2.0, 'score': 2.0, 'sex': 2.0, 'Harry': 2.0, 'Sally': 2.0, 'night': 2.0, 'success': 2.0, 'assure': 2.0, 'review': 2.0, 'critic': 2.0, 'guarantee.v.02': 1.0, 'pretend': 2.0, 'make_believe': 4.0, 'fictitiously': 2.0, 'actress': 2.0, 'act.v.03': 1.0, 'go_through_the_motions.v.01': 1.0, 'consider': 2.0, 'problem': 2.0, 'see.v.05': 1.0, 'calculate': 2.0, '100': 2.0, 'foot': 2.0, 'height': 2.0, 'estimate.v.01': 1.0, 'enjoyable': 2.0, 'pleasurable': 2.0, 'favor': 2.0, 'development': 2.0, 'practice': 2.0, 'winner': 2.0, 'develop.v.13': 1.0, 'develop': 2.0, 'splendid': 2.0, 'develop.v.14': 1.0, 'behave': 2.0, 'merry': 2.0, 'urinate': 4.0, 'piddle': 4.0, 'puddle': 4.0, 'micturate': 4.0, 'piss': 4.0, 'pee': 4.0, 'pee-pee': 4.0, 'make_water': 4.0, 'relieve_oneself': 4.0, 'take_a_leak': 4.0, 'spend_a_penny': 4.0, 'wee': 4.0, 'wee-wee': 4.0, 'pass_water': 4.0, 'urine': 2.0, 'eliminate': 2.0, 'rug': 2.0, 'expensive': 2.0, 'cat': 2.0, 'stale.v.01': 1.0, 'wet.v.02': 1.0, 'fly': 4.0, 'active': 2.0, 'insect': 2.0, 'characterize': 2.0, 'wing': 4.0, 'diptera.n.01': 2.0, 'dipterous_insect.n.01': 1.0, 'bee_fly.n.01': 1.0, 'blowfly.n.01': 1.0, 'flesh_fly.n.01': 1.0, 'gadfly.n.02': 1.0, 'horn_fly.n.01': 1.0, 'housefly.n.01': 1.0, 'tachina_fly.n.01': 1.0, 'tsetse_fly.n.01': 1.0, 'tent-fly': 4.0, 'rainfly': 4.0, 'fly_sheet': 4.0, 'tent_flap': 4.0, 'flap': 2.0, 'consisting': 2.0, 'provide': 2.0, 'entrance': 2.0, 'tent': 2.0, 'canvas': 2.0, 'flap.n.01': 1.0, 'fly_front': 4.0, 'button': 2.0, 'garment': 2.0, 'conceal': 2.0, 'zipper': 2.0, 'close': 2.0, 'cloth': 2.0, 'fold': 2.0, 'opening': 2.0, 'opening.n.10': 1.0, 'fly_ball': 4.0, 'air': 2.0, 'baseball': 2.0, 'hit.n.02': 1.0, 'blast.n.01': 1.0, 'flare.n.11': 1.0, 'liner.n.01': 1.0, 'pop_fly.n.01': 1.0, 'texas_leaguer.n.01': 1.0, 'lure': 2.0, 'fishhook': 2.0, 'fisherman': 2.0, 'decorate': 2.0, "fisherman's_lure.n.01": 1.0, 'dry_fly.n.01': 1.0, 'streamer_fly.n.01': 1.0, 'wet_fly.n.01': 1.0, 'airborne': 2.0, 'travel': 2.0, 'travel.v.01': 1.0, 'buzz.v.02': 1.0, 'flight.v.02': 1.0, 'fly_on.v.01': 1.0, 'hover.v.03': 1.0, 'rack.v.06': 1.0, 'soar.v.03': 1.0, 'suddenly': 2.0, 'quickly': 2.0, 'place': 2.0, 'move.v.03': 1.0, 'aviate': 4.0, 'pilot': 2.0, 'airplane': 2.0, 'operate': 2.0, 'Cuba': 2.0, 'operate.v.03': 1.0, 'balloon.v.01': 1.0, 'fly_blind.v.01': 1.0, 'fly_contact.v.01': 1.0, 'glide.v.02': 1.0, 'hang_glide.v.01': 1.0, 'hedgehop.v.01': 1.0, 'hydroplane.v.01': 1.0, 'jet.v.02': 1.0, 'solo.v.01': 1.0, 'test_fly.v.01': 1.0, 'aeroplane': 2.0, 'transport': 2.0, 'Caribbean': 2.0, 'America': 2.0, 'North': 2.0, 'transport.v.02': 1.0, 'airlift.v.01': 1.0, 'float': 2.0, 'kite': 2.0, 'kite.v.04': 1.0, 'disseminate': 2.0, 'disperse': 2.0, 'rumor': 2.0, 'accusation': 2.0, 'emotional': 2.0, 'state': 2.0, 'rage': 2.0, 'fell': 4.0, 'vanish': 2.0, 'rapidly': 2.0, 'away': 2.0, 'arrow': 2.0, 'beneath': 2.0, 'flee': 2.0, 'elapse.v.01': 1.0, 'tonight': 2.0, 'Cincinnati': 2.0, 'drive': 2.0, 'travel.v.05': 1.0, 'red-eye.v.01': 1.0, 'display': 2.0, 'U.N.': 2.0, 'flag': 2.0, 'nation': 2.0, 'show.v.04': 1.0, 'take_flight': 4.0, 'gun': 2.0, 'scat.v.01': 1.0, 'abscond.v.01': 1.0, 'break.v.19': 1.0, 'defect.v.01': 1.0, 'elope.v.01': 1.0, 'escape.v.01': 1.0, 'high-tail.v.01': 1.0, 'stampede.v.04': 1.0, 'land': 2.0, 'aircraft': 2.0, 'sea': 2.0, 'area': 2.0, 'Lindbergh': 2.0, 'Atlantic': 2.0, 'travel.v.04': 1.0, 'hit.v.01': 1.0, 'vaporize': 2.0, 'disappear': 2.0, 'decrease': 2.0, 'Vegas': 2.0, 'las': 2.0, 'stock': 2.0, 'asset': 2.0, 'decrease.v.01': 1.0, 'informal': 2.0, 'hoodwink': 2.0, 'deceive': 2.0, 'british': 2.0, 'resolve': 2.0, 'need': 2.0, 'difficulty': 2.0, 'husband': 2.0, 'contact': 2.0, 'traffic': 2.0, 'smog': 2.0, 'urban': 2.0, 'congestion': 2.0, 'difficulty.n.03': 1.0, 'balance-of-payments_problem.n.01': 1.0, 'race_problem.n.01': 1.0, 'question': 2.0, 'raise': 2.0, 'consideration': 2.0, 'solution': 2.0, 'homework': 2.0, 'solve': 2.0, 'question.n.02': 1.0, 'case.n.08': 1.0, 'gordian_knot.n.01': 1.0, 'homework_problem.n.01': 1.0, 'koan.n.01': 1.0, 'pons_asinorum.n.01': 1.0, 'poser.n.03': 1.0, 'puzzle.n.01': 1.0, 'rebus.n.01': 1.0, 'riddle.n.01': 1.0, 'trouble': 2.0, 'source': 2.0, 'delay': 2.0, 'difficulty.n.02': 1.0, 'can_of_worms.n.01': 1.0, 'deep_water.n.01': 1.0, 'growing_pains.n.03': 1.0, 'hydra.n.03': 1.0, 'matter.n.04': 1.0, 'pressure_point.n.02': 1.0}




given the sentence:
	-- ['I love to pet my cat while reading fantasy books.', 'Reading books makes my fantasy fly over everyday problems.', 'One problem of those is my cat vomiting on my pants.'] --
{'pant': 2.0, 'engine': 2.0, 'short': 2.0, 'noise': 2.0, 'steam': 2.0, 'puff': 4.0, 'noise.n.01': 1.0, 'trouser': 2.0, 'ankle': 2.0, 'leg': 2.0, 'waist': 2.0, 'garment': 2.0, 'usually': 2.0, 'extend': 2.0, 'knee': 2.0, 'plural': 2.0, 'cover': 2.0, 'separately': 2.0, 'sharp': 2.0, 'crease': 2.0, 'garment.n.01': 1.0, 'bellbottom_trousers.n.01': 1.0, 'breeches.n.01': 1.0, 'chino.n.01': 1.0, 'churidars.n.01': 1.0, 'cords.n.01': 1.0, 'flannel.n.03': 1.0, 'jean.n.01': 1.0, 'jodhpurs.n.01': 1.0, 'long_trousers.n.01': 1.0, 'pajama.n.01': 1.0, 'pantaloon.n.03': 1.0, 'pedal_pusher.n.01': 1.0, 'salwar.n.01': 1.0, 'short_pants.n.01': 1.0, 'slacks.n.01': 1.0, 'stretch_pants.n.01': 1.0, 'sweat_pants.n.01': 1.0, 'trews.n.01': 1.0, 'gasp': 4.0, 'breath': 2.0, 'mouth': 2.0, 'open': 2.0, 'labored': 2.0, 'intake': 2.0, 'give': 2.0, 'faint': 2.0, 'inhalation.n.01': 1.0, 'heave': 4.0, 'noisily': 2.0, 'breathe': 2.0, 'exhausted': 2.0, 'finish': 2.0, 'line': 2.0, 'runner': 2.0, 'reach': 2.0, 'heavily': 2.0, 'blow.v.01': 1.0, 'utter': 2.0, 'utter.v.02': 1.0, 'problem': 2.0, 'job': 2.0, 'resolve': 2.0, 'need': 2.0, 'state': 2.0, 'difficulty': 2.0, 'husband': 2.0, 'have': 2.0, 'contact': 2.0, 'traffic': 2.0, 'smog': 2.0, 'urban': 2.0, 'congestion': 2.0, 'difficulty.n.03': 1.0, 'balance-of-payments_problem.n.01': 1.0, 'race_problem.n.01': 1.0, 'question': 2.0, 'raise': 2.0, 'consideration': 2.0, 'solution': 2.0, 'homework': 2.0, 'solve': 2.0, 'consist': 2.0, 'question.n.02': 1.0, 'case.n.08': 1.0, 'gordian_knot.n.01': 1.0, 'homework_problem.n.01': 1.0, 'koan.n.01': 1.0, 'pons_asinorum.n.01': 1.0, 'poser.n.03': 1.0, 'puzzle.n.01': 1.0, 'rebus.n.01': 1.0, 'riddle.n.01': 1.0, 'trouble': 2.0, 'source': 2.0, 'delay': 2.0, 'difficulty.n.02': 1.0, 'can_of_worms.n.01': 1.0, 'deep_water.n.01': 1.0, 'growing_pains.n.03': 1.0, 'hydra.n.03': 1.0, 'matter.n.04': 1.0, 'pressure_point.n.02': 1.0, 'vomit': 2.0, 'vomitus': 4.0, 'puke': 4.0, 'barf': 4.0, 'vomiting': 4.0, 'eject': 2.0, 'matter': 2.0, 'body_waste.n.01': 1.0, 'emetic': 4.0, 'vomitive': 4.0, 'nauseant': 4.0, 'medicine': 2.0, 'nausea': 2.0, 'induce': 2.0, 'remedy.n.02': 1.0, 'ipecac.n.01': 1.0, 'powdered_mustard.n.01': 1.0, 'emesis': 4.0, 'regurgitation': 4.0, 'disgorgement': 4.0, 'puking': 4.0, 'reflex': 2.0, 'stomach': 2.0, 'content': 2.0, 'act': 2.0, 'expulsion.n.03': 1.0, 'reflex.n.01': 1.0, 'hematemesis.n.01': 1.0, 'hyperemesis.n.01': 1.0, 'rumination.n.03': 1.0, 'vomit_up': 4.0, 'purge': 2.0, 'cast': 4.0, 'sick': 4.0, 'cat': 4.0, 'be_sick': 4.0, 'disgorge': 4.0, 'regorge': 4.0, 'retch': 4.0, 'spew': 4.0, 'spue': 4.0, 'chuck': 4.0, 'upchuck': 4.0, 'honk': 4.0, 'regurgitate': 2.0, 'throw_up': 4.0, 'student': 2.0, 'drink': 2.0, 'continuously': 2.0, 'food': 2.0, 'night': 2.0, 'patient': 2.0, 'excrete.v.01': 1.0, 'true_cat': 4.0, 'soft': 2.0, 'fur': 2.0, 'mammal': 2.0, 'feline': 2.0, 'wildcat': 2.0, 'ability': 2.0, 'roar': 2.0, 'thick': 2.0, 'domestic': 2.0, 'feline.n.01': 1.0, 'domestic_cat.n.01': 1.0, 'wildcat.n.03': 1.0, 'guy': 2.0, 'hombre': 4.0, 'bozo': 4.0, 'informal': 2.0, 'youth': 2.0, 'term': 2.0, 'man': 2.0, 'nice': 2.0, 'doll': 2.0, 'man.n.01': 1.0, 'sod.n.04': 1.0, 'woman': 2.0, 'spiteful': 2.0, 'gossip': 2.0, 'gossip.n.03': 1.0, 'woman.n.01': 1.0, 'kat': 2.0, 'khat': 4.0, 'qat': 4.0, 'quat': 4.0, 'Arabian_tea': 4.0, 'African_tea': 4.0, 'effect': 2.0, 'stimulant': 2.0, 'leave': 2.0, 'shrub': 2.0, 'tobacco': 2.0, 'eduli': 2.0, 'euphoric': 2.0, 'Catha': 2.0, 'tea': 2.0, 'chew': 2.0, 'like': 2.0, 'adult': 2.0, '%': 2.0, '85': 2.0, 'Yemen': 2.0, 'daily': 2.0, 'stimulant.n.02': 1.0, "cat-o'-nine-tails": 4.0, 'knotted': 2.0, 'whip': 2.0, 'cord': 2.0, 'fear': 2.0, 'sailor': 2.0, 'british': 2.0, 'whip.n.01': 1.0, 'Caterpillar': 4.0, 'construction': 2.0, 'track': 2.0, 'frequently': 2.0, 'work': 2.0, 'propel': 2.0, 'move': 2.0, 'vehicle': 2.0, 'endless': 2.0, 'belt': 2.0, 'earth': 2.0, 'large': 2.0, 'metal': 2.0, 'farm': 2.0, 'tracked_vehicle.n.01': 1.0, 'big_cat': 4.0, 'wild': 2.0, 'live': 2.0, 'able': 2.0, 'typically': 2.0, 'felidae.n.01': 2.0, 'cheetah.n.01': 1.0, 'jaguar.n.01': 1.0, 'leopard.n.02': 1.0, 'liger.n.01': 1.0, 'lion.n.01': 1.0, 'saber-toothed_tiger.n.01': 1.0, 'snow_leopard.n.01': 1.0, 'tiger.n.02': 1.0, 'tiglon.n.01': 1.0, 'computerized_tomography': 4.0, 'computed_tomography': 4.0, 'CT': 4.0, 'computerized_axial_tomography': 4.0, 'computed_axial_tomography': 4.0, 'CAT': 4.0, '-': 2.0, 'ray': 2.0, 'method': 2.0, 'axis': 2.0, 'organ': 2.0, 'cross': 2.0, 'scan': 2.0, 'body': 2.0, 'series': 2.0, 'construct': 2.0, 'single': 2.0, 'examine': 2.0, 'computer': 2.0, 'sectional': 2.0, 'x': 2.0, 'x-raying.n.01': 1.0, "o'-nine": 2.0, 'beat': 2.0, 'tail': 2.0, 'flog.v.01': 1.0}
'''

print("\n\n\n\n end")
