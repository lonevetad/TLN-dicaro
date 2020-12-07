from notebooks.es4 import es4
from notebooks.es4.es4 import DocumentSegmentator
from notebooks.utilities.cacheVarie import CacheSynsets


local_cache_synset = CacheSynsets()

#"cat_book",
#"iceland",
files = [
    "cat_book",
    "iceland",
    "cambridge_analytica",
    "brexit"
]
for filename in files:
    document_sentences = []
    print("\n\nusando il file", filename, "come input")
    with open(".\\Input\\"+filename + ".txt") as file_in:
        for line in file_in:
            line_stripped = line.strip()
            if len(line_stripped) > 0:
                document_sentences.append(line_stripped)
    print("start analysys with " + str(len(document_sentences)) + " sentences")
    ds = DocumentSegmentator(document_sentences, cache_synset_and_bag=local_cache_synset)
    paragraphs = ds.document_segmentation(5)
    for p in paragraphs:
        print("\nparagraph:")
        for s in p:
            print("----", s)

print("\n\n\n\n end")
