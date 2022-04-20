from keybert import KeyBERT
import ast

class KeybertWrapper:
    def __init__(self, docs):
        self.docs = docs
        self.kw_model = KeyBERT()

    def find_keywords(self):
        keywords = []
        content_dict = ast.literal_eval(str(self.docs))

        for key, value in content_dict.items():
            keywords.append(str(key) + ": " + str(self.run_keybert(value)))

        return keywords

    def run_keybert(self, content):
        keywords = self.kw_model.extract_keywords(content, keyphrase_ngram_range = (1,5), top_n=10, stop_words='english')
        return keywords

    def write_keywords_to_disk(self, keywords, output_file_directory):
        output = open(output_file_directory + "keybert_keywords_results" + ".txt", "a+")
        output.write(str(keywords))
        output.close()