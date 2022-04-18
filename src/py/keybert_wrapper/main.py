from keybert import KeyBERT

class KeybertWrapper:
    def __init__(self, docs):
        self.docs = docs
        self.kw_model = KeyBERT()

    def find_keywords(self):
        keywords = self.kw_model.extract_keywords(self.docs, keyphrase_ngram_range=(1,5), top_n=10, stop_words='english')
        return keywords

    def write_keywords_to_disk(self, keywords):
        pass