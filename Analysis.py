from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import Pre_Process as Pp
import pyspark.sql.functions as sqlf


def total_analysis(nb_tf_model, nb_embed_model, dtree_tf_model, dtree_embed_model, word2vec_model, reddit_dumb, spark):
    dumb_indexed = Pp.string_to_label(reddit_dumb, "subreddit")
    tf_dumb_vector = Pp.text_to_freq_vectors(dumb_indexed, "TF", "text")
    w2v_dumb_vector = Pp.text_to_embed_vectors(word2vec_model=word2vec_model, dataset=dumb_indexed, inputcol="text")
    evaluator = MulticlassClassificationEvaluator().setLabelCol("label"). \
        setPredictionCol("prediction").setMetricName("accuracy")

    print("TF based results")
    predictions_tf_nb = nb_tf_model.transform(tf_dumb_vector)
    detailed_results(predictions_tf_nb, "NB TF Results", spark)
    accuracy_tf_nb = evaluator.evaluate(predictions_tf_nb)
    print("NAIVE BAYES TF Accuracy =  %g" % (accuracy_tf_nb * 100))
    predictions_dtree_tf = dtree_tf_model.transform(tf_dumb_vector)
    detailed_results(predictions_dtree_tf, "DTREE TF Results", spark)
    accuracy_tf_dtree = evaluator.evaluate(predictions_dtree_tf)
    print("DECISION TREE TF Accuracy =  %g" % (accuracy_tf_dtree * 100))

    print("\n\n\nEmbed based results")
    predictions_nb_embed = nb_embed_model.transform(w2v_dumb_vector)
    detailed_results(predictions_nb_embed, "NB EMBED Results", spark)
    accuracy_embed_nb = evaluator.evaluate(predictions_nb_embed)
    print("NAIVE BAYES Embed =  %g" % (accuracy_embed_nb * 100))
    predictions_dtree_embed = dtree_embed_model.transform(w2v_dumb_vector)
    detailed_results(predictions_dtree_embed, "DTREE EMBED Results", spark)
    accuracy_embed_dtree = evaluator.evaluate(predictions_dtree_embed)
    print("DECISION TREE Embed =  %g" % (accuracy_embed_dtree * 100))


def detailed_results(dataset, identifier, spark):
    dataset.createOrReplaceTempView("reddit")
    print(identifier)
    categories = dataset.withColumn('category', sqlf.col("prediction")). \
        groupBy('category').count().sort('count', ascending=False)
    categories.show()
    spark.catalog.dropTempView("reddit")

"""   
reddit = spark.sql("SELECT subreddit, COUNT(*) FROM reddit WHERE label == prediction "
                       "GROUP BY subreddit ORDER BY COUNT(*) desc")
"""