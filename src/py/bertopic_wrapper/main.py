import os
import sys
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class BertopicTraining():
    def __init__(self, absolute_in_file_path, out_directory_path, out_filename, search_term):
        self.out_directory_path = out_directory_path
        self.out_filename = out_filename
        self.search_term = search_term
        self.topic_model = BERTopic()
        
        # Create data frame and store as object
        # df = pd.read_csv(out_dir + "/" + out_file + ".csv")
        df = pd.read_json(os.path.abspath(absolute_in_file_path), lines=True, encoding="utf-8-sig")
        self.data = df["content"]

        print(self.data)

    def trainModel(self):
        topics = None
        probs = None

        try:
            topics, probs = self.topic_model.fit_transform(self.data)
        except RuntimeError as e:
            print("\n=== Out of GPU memory in order for cuda to work properly ===\n")

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
        self.topic_model.update_topics(self.data, topics, vectorizer_model=vectorizer_model)
    
        self.write_training_data_to_disk(self.topic_model,
                                     self.topic_model.get_topic_info(), 
                                     self.topic_model.find_topics(self.search_term),
                                     self.topic_model.get_topics(),
                                     self.topic_model.get_representative_docs(), 
                                     self.topic_model.get_topic_freq())

        self.write_visualization_data_to_disk(self.topic_model)

        print(self.topic_model.get_topic_info())

    def write_training_data_to_disk(self, topic_model, topicInfo, findTopics, allTopicInfo, repDoc, topicFrequency):
        try:
            self.ml_data_path = "/" + self.out_directory_path + "/ml_data"
        
            try:
                os.mkdir(self.ml_data_path)
            except FileExistsError as e:
                print("[WARNING]: ml_data folder already exists, writing to previous folder")

            topic_info_dir = os.path.join(self.ml_data_path, self.out_filename + "_TOPIC_INFO" + ".csv")
            all_topic_info_dir = os.path.join(self.ml_data_path, self.out_filename + "_ALL_TOPIC_INFO" + ".csv")
            find_topics_dir = os.path.join(self.ml_data_path, self.out_filename + "_FOUND_TOPICS" + ".csv")
            rep_doc_dir = os.path.join(self.ml_data_path, self.out_filename + "_REPERSENTITIVE_DOCS" + ".csv")
            topic_frequency_dir = os.path.join(self.ml_data_path, self.out_filename + "_TOPIC_FREQUENCY" + ".csv")
            topic_model_dir = os.path.join(self.ml_data_path, self.out_filename + "_TOPIC_MODEL" + ".bin")
            formatted_found_topics_dir = os.path.join(self.ml_data_path, self.out_filename + "_FORMATTED_FOUND_TOPICS.csv")
            formatted_all_topics_dir = os.path.join(self.ml_data_path, self.out_filename + "_FORMATTED_ALL_TOPICS.csv")


            TIF = open(topic_info_dir, "w")
            TIF.write(topicInfo.to_string())
            TIF.close()

            AIF = open(all_topic_info_dir, "w")
            AIF.write(str(allTopicInfo))
            AIF.close()

            FTF = open(find_topics_dir, "w")
            FTF.write(str(findTopics))
            FTF.close()

            RDF = open(rep_doc_dir, "w")
            print(repDoc, file=RDF)
            RDF.close()

            TFF = open(topic_frequency_dir, "w")
            TFF.write(topicFrequency.to_string())
            TFF.close()

            self.format_found_topics(formatted_found_topics_dir)
            
            self.format_all_topics(formatted_all_topics_dir)

            topic_model.save(topic_model_dir)
    
        except ValueError as e: 
            print("Error: " + str(e))
            print("!=== This probably has to do with bertopic not being provided enough data ===!")

    def write_visualization_data_to_disk(self, topic_model):
        try:
            path = "/" + self.out_directory_path + "/visualizations/"

            try:
                os.mkdir(path)
            except FileExistsError:
                print("[WARNING]: Visualizations folder already exists, writing to previously created folder.")

            vt = topic_model.visualize_topics()
            vt.write_html(path + "topics_visual.html")

            vhi = topic_model.visualize_hierarchy()
            vhi.write_html(path + "hierarchy_visual.html")

            vb = topic_model.visualize_barchart()
            vb.write_html(path + "barchart_visual.html")

            vhe = topic_model.visualize_heatmap()
            vhe.write_html(path + "heatmap_visual.html")
        except ValueError as e: 
            print("Error: " + str(e))
            print("!=== This probably has to do with bertopic not being provided enough data ===!")

    def format_found_topics(self, out_filename):
        found_topics = []

        index = self.topic_model.find_topics(self.search_term, top_n=15)[0]
        salience = self.topic_model.find_topics(self.search_term, top_n=15)[1]

        for x in index:
            found_topics.append(self.topic_model.get_topic(x))

        data = [salience, found_topics]
        df = pd.DataFrame(data=data, columns=index, index = ["Salience", "Terms"])
        df = df.explode(index)
        df.to_csv(out_filename)

    def format_all_topics(self, out_filename):
        topic_name = self.topic_model.get_topic_info()['Name']
        topic_number = self.topic_model.get_topic_info()['Topic']
        topics_by_topic_number = []

        for x in topic_number:
            topics_by_topic_number.append(self.topic_model.get_topic(x))

        df = pd.DataFrame(data=topics_by_topic_number, index=topic_name)
        df.to_csv(out_filename)

    def get_rep_docs(self):
        return self.topic_model.get_representative_docs()