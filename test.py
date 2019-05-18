import os
import pyspark.sql.functions as sqlf
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer, Word2VecModel, Tokenizer
from pyspark.sql import SparkSession
import pyspark.sql.functions as sqlf
from pyspark.sql.types import StructType, StructField, StringType

import Pre_Process as Pp

path = "/media/ulgen/4AF6618EF6617B59/Users/ULGEN/Desktop/RC_2015.json"
main_data_path = "/home/ulgen/Documents/Pycharm_Workspace/Keyword_Extraction/Data/"
os.environ['PYSPARK_PYTHON'] = "/home/ulgen/anaconda3/envs/Keyword_Extraction/bin/python3.7"

conf = SparkConf().setMaster("local[*]") \
    .setAppName("Keyword_Extraction") \
    .set("spark.driver.memory", "20g") \
    .set("spark.executor.memory	", "4g") \
    .set("spark.driver.maxResultSize", "3g")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()


def keyword_extractor():
    keyword_list = ["music", "science", "politics", "gaming", "cars"]

    word2vec_model = Word2VecModel.load(main_data_path + "word2vec")

    schema = StructType([StructField("word", StringType(), True), StructField("string_label", StringType(), False)])
    training_set = spark.createDataFrame([], schema)

    for i in range(0, len(keyword_list)):
        keyword_set = word2vec_model.findSynonyms(keyword_list[i].lower(), 1000).drop("similarity") \
            .withColumn("string_label", sqlf.lit(keyword_list[i]))

        training_set = training_set.union(keyword_set)

    tokenizer = Tokenizer(inputCol="word", outputCol="text")
    training_set = tokenizer.transform(training_set).drop("word")
    training_set.coalesce(1).write.json(main_data_path + "w2v_keywords")

    word_counts = training_set.withColumn('word', sqlf.explode(sqlf.col("text"))).count()
    print(str(word_counts))


def dataset_reducer():
    reddit_raw = spark.read.json(main_data_path + "data") \
        .where((sqlf.col("subreddit") == "Music") |
               (sqlf.col("subreddit") == "science") |
               (sqlf.col("subreddit") == "politics") |
               (sqlf.col("subreddit") == "gaming") |
               (sqlf.col("subreddit") == "cars"))

    reddit_raw.repartition(16, "text", "subreddit").coalesce(1).write.json("main_data_path/TF_Train")


def selective_data():
    reddit = spark.read.json(main_data_path + "train_data.json")

    music = reddit.select("text", "subreddit").where("subreddit == 'Music'").limit(5000)
    science = reddit.select("text", "subreddit").where("subreddit == 'science'").limit(5000)
    politics = reddit.select("text", "subreddit").where("subreddit == 'politics'").limit(5000)
    gaming = reddit.select("text", "subreddit").where("subreddit == 'gaming'").limit(5000)
    cars = reddit.select("text", "subreddit").where("subreddit == 'cars'").limit(5000)

    total_set = music.union(science).union(politics).union(gaming).union(cars)

    print(str(total_set.count()))
    total_set.coalesce(1).write.json(main_data_path + " 250k_set")


selective_data()
