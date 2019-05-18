import datetime

from pyspark.ml.feature import Word2Vec, Tokenizer
from pyspark.sql.functions import lit
from pyspark.sql.types import StructField, StringType, StructType


def spark_doc2vec_model(dataset, keyword_list, keyword_size, spark, path):
    date1 = datetime.datetime.now()
    # dataset.persist()
    word2vec = Word2Vec(vectorSize=300, minCount=0, numPartitions=16, maxIter=8, inputCol="text",
                        outputCol="features", windowSize=5)
    model = word2vec.fit(dataset)
    # dataset.unpersist()
    model.save(path + "word2vec_model")
    print("Total Unique Dictionary Size = " + str(model.getVectors().select("word").count()))

    # we create empty dataframe to union all keywords sets in to one
    schema = StructType([StructField("word", StringType(), True), StructField("string_label", StringType(), False)])
    training_set = spark.createDataFrame([], schema)

    for i in range(0, len(keyword_list)):
        keyword_set = model.findSynonyms(keyword_list[i].lower(), keyword_size).drop("similarity") \
            .withColumn("string_label", lit(keyword_list[i]))

        training_set = training_set.union(keyword_set)

    tokenizer = Tokenizer(inputCol="word", outputCol="text")
    training_set = tokenizer.transform(training_set).drop("word")
    training_set.coalesce(1).write.json(path + "w2v_keywords")

    date2 = datetime.datetime.now()
    print("Total seconds for word2vec train phase is = " + str((date2 - date1).seconds))
    print("\n\n")

    return training_set, model


"""

# .select("word", fmt("similarity", 5).alias("similarity"))


- if your dataframe's columns cant match use this to re-cast them to same type

keyword_set = synonyms_set.withColumn("text", synonyms_set.text.cast(StringType())) \
            .withColumn("label", synonyms_set.label.cast(StringType()))
            
        
"""
