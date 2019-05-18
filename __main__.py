"""
   GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import os

from pyspark.ml.feature import Word2VecModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from ML_Models import NB_Model as Nb
from ML_Models import Decision_Tree as DTree

import Spark_Word2Vec as W2v
import Analysis
import Pre_Process as Pp

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-oracle"
os.environ['PYSPARK_PYTHON'] = "/home/ulgen/anaconda3/envs/Keyword_Extraction/bin/python3.7"
main_data_path = "/home/ulgen/Documents/Pycharm_Workspace/Keyword_Extraction/Data/"

if __name__ == '__main__':
    conf = SparkConf().setMaster("local[*]") \
        .setAppName("Keyword_Extraction") \
        .set("spark.driver.memory", "35g") \
        .set("spark.executor.memory	", "4g") \
        .set("spark.driver.maxResultSize", "4g") \
        .set("spark.rdd.compress", "true")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    spark = SparkSession \
        .builder \
        .config(conf=conf) \
        .getOrCreate()

    # reddit_raw = spark.read.json(main_data_path + "five_class.json").select("body", "subreddit")

    # training_set = spark.read.json(main_data_path + "train_data.json")
    # category_list = ["music", "science", "politics", "gaming", "cars"]
    # reddit_cleaned = Pp.clean_raw_dataset(raw_dataset=reddit_raw, path=main_data_path)
    # keyword_set, word2vec_model = W2v.spark_doc2vec_model(dataset=training_set, keyword_list=category_list,
    #                                                      keyword_size=700, spark=spark, path=main_data_path)

    tf_keyword_set = spark.read.json(main_data_path + "tf_train.json").withColumnRenamed("subreddit", "string_label")
    embed_keyword_set = spark.read.json(main_data_path + "w2v_keywords.json")
    word2vec_model = Word2VecModel.load(main_data_path + "word2vec_model")

    NB_Freq_Model = Nb.naive_bayes_modelling(training_set=tf_keyword_set, w2v_model=word2vec_model,
                                             vectorization="Frequency_Based")

    NB_Embed_Model = Nb.naive_bayes_modelling(training_set=embed_keyword_set, w2v_model=word2vec_model,
                                              vectorization="Embed_Based")

    DTree_Freq_Model = DTree.decision_tree_model(training_set=tf_keyword_set, w2v_model=word2vec_model,
                                                 vectorization="Frequency_Based")

    DTree_Embed_Model = DTree.decision_tree_model(training_set=embed_keyword_set, w2v_model=word2vec_model,
                                                  vectorization="Embed_Based")

    reddit_test_set = spark.read.json(main_data_path + "250k_set.json")
    Analysis.total_analysis(nb_tf_model=NB_Freq_Model, nb_embed_model=NB_Embed_Model, dtree_tf_model=DTree_Freq_Model,
                            dtree_embed_model=DTree_Embed_Model, word2vec_model=word2vec_model,
                            reddit_dumb=reddit_test_set, spark=spark)

