import datetime

import pyspark.sql.functions as sqlf
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, StringIndexer, MinMaxScaler, HashingTF, IDF, Normalizer


def find_subreddits(dataset):
    categories = dataset.withColumn('category', sqlf.col("subreddit")). \
        groupBy('category').count().sort('count', ascending=False)
    print("the total size of the categories is =" + str(categories.count()) + "\n")
    categories.show(500, False)


def word_count(dataset, column, state):
    if state == "word_count":
        word_counts = dataset.withColumn('word', sqlf.explode(sqlf.col(column))). \
            groupBy('word').count().sort('count', ascending=True)
        return word_counts
    elif state == "total_count":
        word_counts = dataset.withColumn('word', sqlf.explode(sqlf.col(column))).count()
        return word_counts


def clean_raw_dataset(raw_dataset,path):
    raw_dataset_row_count = raw_dataset.count()
    print("Total raw dataset row size= " + str(raw_dataset_row_count))

    date1 = datetime.datetime.now()

    # This part we clean "[deleted]" and "[removed]" rows since they hold no information
    cleaned_set = raw_dataset.select("body", "subreddit").where("body !='[deleted]' AND body !='[removed]'")
    cleaned_set_count = cleaned_set.count()
    print("Clean dataset row size = " + str(cleaned_set_count))
    print("Pruned Rows size = " + str(raw_dataset_row_count - cleaned_set_count))

    # This part we use regex tokenizer to tokenize word and remove artifacts
    regex_tokenizer = RegexTokenizer(minTokenLength=2, inputCol="body", outputCol="words", pattern="\\W")
    tokenized = regex_tokenizer.transform(cleaned_set).drop("body")
    print("total word count after regex tokenizing = " +
          str(word_count(dataset=tokenized, column="words", state="total_count")))

    # This part Drops the Stop Words from dataset
    tokenized.cache()
    remover = StopWordsRemover(inputCol="words", outputCol="stop_word_less")
    stopwordremoved_set = remover.transform(tokenized).drop("words")
    tokenized.unpersist()
    print("total word count after stop words removing = " +
          str(word_count(dataset=stopwordremoved_set, column="stop_word_less", state="total_count")))

    # This part we find the rare words below given treshold
    rare_words = word_count(stopwordremoved_set, 'stop_word_less', "word_count"). \
        select("word").where("count <= 5").rdd.flatMap(lambda x: x).collect()
    print("rare word size is = " + str(len(rare_words)))

    # This Part we define a stop word instance and feed it with our rare words to remove
    stopwordremoved_set.cache()
    rare_word_remover = StopWordsRemover(inputCol="stop_word_less", outputCol="text", stopWords=rare_words)
    reduced_set = rare_word_remover.transform(stopwordremoved_set).drop("stop_word_less")
    print("rare words are deleted")
    print("Total Rows after rare words deletion = " + str(reduced_set.count()))
    print("total word count after rare words removing = " +
          str(word_count(dataset=reduced_set, column="text", state="total_count")))
    stopwordremoved_set.unpersist()
    date2 = datetime.datetime.now()
    print("Total seconds for cleaning phase is = " + str((date2 - date1).seconds))

    # This part we save the reduced set and measure the new word count
    reduced_set.cache()
    reduced_set.coalesce(1).write.json(path+"W2V_Train_Data")
    print("reduced set has saved")
    reduced_set.unpersist()

    return reduced_set


def string_to_label(training_set, inputcol):
    indexer = StringIndexer(inputCol=inputcol, outputCol="label", stringOrderType="alphabetDesc")
    indexed = indexer.fit(training_set).transform(training_set)  # .drop(inputcol)
    return indexed


def text_to_freq_vectors(dataset, state, inputcol):
    if state == "TF":
        hashing_tf = HashingTF(inputCol=inputcol, outputCol="features", numFeatures=100)
        featurized_data = hashing_tf.transform(dataset).drop(inputcol)
        normal_data = normalizer(featurized_data)
        return normal_data

    elif state == "TF-IDF":
        hashing_tf = HashingTF(inputCol=inputcol, outputCol="rawFeatures", numFeatures=10)
        featurized_data = hashing_tf.transform(dataset).drop(inputcol)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(featurized_data)
        rescaled_data = idf_model.transform(featurized_data).drop("rawFeatures")
        normal_data = normalizer(rescaled_data)
        return normal_data


def text_to_embed_vectors(word2vec_model, dataset, inputcol):
    vector = word2vec_model.transform(dataset).drop(inputcol)
    scalar_vector = min_max_scalar(vector)
    return scalar_vector


def min_max_scalar(vector):
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(vector)
    scaled_data = scaler_model.transform(vector).drop("features").withColumnRenamed("scaledFeatures", "features")
    return scaled_data


def normalizer(vector):
    normalizer_params = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
    l2_norm_data = normalizer_params.transform(vector).drop(
        "features").withColumnRenamed("normFeatures", "features")
    return l2_norm_data


# query = "SELECT body,subreddit FROM reddit WHERE (body !='[deleted]' AND body !='[removed]')"
# total_words = spark.sql("SELECT SUM (LENGTH(body)) AS word_count FROM reddit")
# spark.sql("SELECT * FROM reddit WHERE parent_id = 't3_675oj' ORDER BY created_utc ASC")
# clean_n_len = spark.sql("SELECT *, LENGTH(body) as body_len FROM reddit2 order by (body_len) desc ")
# return datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
# total_test_set.select("body").repartition(16, "body","subreddit").coalesce(1).write.text/json("path")
#  collected_set = stopwordremoved_set.withColumn("filtered", str_convert(stopwordremoved_set.filtered)) \
#        .withColumnRenamed(existing="filtered", new="body")
# return ' '.join(body_text)
# gaming_set = spark.sql("SELECT * FROM reddit WHERE subreddit =='"+category[0]+"' LIMIT 2500")

"""
def find_subreddits(dataset, spark):
    dataset.createOrReplaceTempView("reddit")
    categories = spark.sql(
        "SELECT subreddit, COUNT(*) FROM reddit GROUP BY subreddit ORDER BY COUNT(*) desc").distinct()
    categories.show(400, False)
    print("the total size of the categories is =" + str(categories.count()) + "\n")
    spark.catalog.dropTempView("reddit")
    
    This way of measure is so slow.
"""
