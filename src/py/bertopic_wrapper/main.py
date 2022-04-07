import os
import sys
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class BertopicTraining():
    def __init__(self, absolute_in_file_path, out_directory_path, out_filename):
        self.out_directory_path = out_directory_path
        self.out_filename = out_filename
        
        # Create data frame and store as object
        # df = pd.read_csv(out_dir + "/" + out_file + ".csv")
        df = pd.read_json(os.path.abspath(absolute_in_file_path), lines=True)
        self.data = df["content"]

        print(self.data)

    def trainModel(self):
        topic_model = BERTopic()
        topics = None
        probs = None

        try:
            topics, probs = topic_model.fit_transform(self.data)
        except RuntimeError as e:
            print("\n=== Out of GPU memory in order for cuda to work properly ===\n")

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
        topic_model.update_topics(self.data, topics, vectorizer_model=vectorizer_model)
    
        self.write_training_data_to_disk(topic_model,
                                     topic_model.get_topic_info(), 
                                     topic_model.get_topics(),
                                     topic_model.get_representative_docs(), 
                                     topic_model.get_topic_freq())

        self.write_visualization_data_to_disk(topic_model)

        print(topic_model.get_topic_info())

    def write_training_data_to_disk(self, topic_model, topicInfo, allTopicInfo, repDoc, topicFrequency):
        try:
            ml_data_path = self.out_directory_path + "/ml_data"
            
            try:
                os.mkdir(ml_data_path)
            
                topic_info_dir = os.path.join(ml_data_path, self.out_filename + "_TOPIC_INFO" + ".txt")
                all_topic_info_dir = os.path.join(ml_data_path, self.out_filename + "_ALL_TOPIC_INFO" + ".txt")
                rep_doc_dir = os.path.join(ml_data_path, self.out_filename + "_REPERSENTITIVE_DOCS" + ".txt")
                topic_frequency_dir = os.path.join(ml_data_path, self.out_filename + "_TOPIC_FREQUENCY" + ".txt")
                topic_model_dir = os.path.join(ml_data_path, self.out_filename + "_TOPIC_MODEL" + ".bin")

                TIF = open(topic_info_dir, "w")
                TIF.write(topicInfo.to_string())
                TIF.close()

                AIF = open(all_topic_info_dir, "w")
                AIF.write(str(allTopicInfo))
                AIF.close()

                RDF = open(rep_doc_dir, "w")
                print(repDoc, file=RDF)
                RDF.close()

                TFF = open(topic_frequency_dir, "w")
                TFF.write(topicFrequency.to_string())
                TFF.close()

                topic_model.save(topic_model_dir)
        
            except FileExistsError:
                print("ml_data folder already exists, writing to previously created folder.")

        except ValueError as e: 
            print("Error: " + str(e))
            print("!=== This probably has to do with bertopic not being provided enough data ===!")

    def write_visualization_data_to_disk(self, topic_model):
        try:
            path = self.out_directory_path + "/visualizations/"

            try:
                os.mkdir(path)
            except FileExistsError:
                print("Visualizations folder already exists, writing to previously created folder.")

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
