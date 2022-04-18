from keybert import KeyBERT
import pickle as p 

class KeybertWrapper:
    def __init__(self, docs):
        self.docs = docs
        self.kw_model = KeyBERT()

    def find_keywords(self):
        keywords = self.kw_model.extract_keywords(self.docs)
        return keywords

    def write_keywords_to_disk(self):
        pass