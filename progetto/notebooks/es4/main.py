from notebooks.es4 import es4
from notebooks.es4.es4 import DocumentSegmentator
from notebooks.utilities.cacheVarie import CacheSynsetsBag

sentences_mocked = [
    "I love to pet my cat while reading fantasy books.",
    "The fur of my cat is so fluffy that chills me.",
    "It entertain me, its tail sometimes goes over my pages and meows, also showing me the weird pattern in its chest's fur.",
    "Sometimes, it plays with me but hurts me with its claws and once it scratched me so bad the one drop of blood felt over my favourite book's pages.",
    "Even so, its agile and soft body at least warms me and appease me with its cute fur, meow and purr, so I alwayse forgive it and those little drops of damage.",
    # "Even so, its agile and soft body warms me with its cute fur, meow and purr, so I've forgave it.",
    "Every time a cat is put aside of a character, i cannot prevent remembering when I found it malnourished and lacking in fur.",

    "There are tons of books I like, from literature, romance, fantasy and sci-fi.",
    "When i hold a book, the stress flushes out and I start reading the whole life wit an external and more critical mindset.",
    "Sometimes, some author or topic can help to better understand the world, as just by giving a meaning",
    "Mostly, I grab a book to just distract the mind in some other reality.",
    "Fantasy is my best genre because let my imagination to run freely.",
    "Reading books makes my fantasy fly over everyday problems.",

    "One problem of those is my cat vomiting on my pants.",
    "But, dealing with people is way more problematic and causes me stomach issues."
    "I find difficult to deal with people, they are usually focused in their egoistic desires and purposes.",
    "Most of them just treat others as resources to achieve their objectives and gets upset if You don't fulfill their expectations.",
    "Sometimes they also undervaluate your own problem, like they are nothing compared to theirs.",
    "Sometimes they even stop listening as you start talking about your own problems, like they are annoyed.",
    "It's a no surprise if happens that I found a better dialogue with a book or my cat."
]

local_cache_synset = CacheSynsetsBag()
document_sentences = sentences_mocked
filename = "brexit"  # "cambridge_analytica"
with open(filename + ".txt") as file_in:
    # noinspection PyRedeclaration
    document_sentences = []
    for line in file_in:
        line_stripped = line.strip()
        if len(line_stripped) > 0:
            document_sentences.append(line_stripped)

print("start analysys with " + str(len(document_sentences)) + " sentences")
ds = DocumentSegmentator(document_sentences, cache_synset_and_bag=local_cache_synset)
paragraphs = ds.document_segmentation(5)
for p in paragraphs:
    print("\n\n paragraph:")
    for s in p:
        print("----", s)

print("\n\n\n\n end")
