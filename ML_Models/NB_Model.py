from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import Pre_Process as Pp


def naive_bayes_modelling(training_set, w2v_model, vectorization=None):
    training_set = Pp.string_to_label(training_set, inputcol="string_label")

    vectors = None

    if vectorization == "Frequency_Based":
        vectors = Pp.text_to_freq_vectors(training_set, state="TF", inputcol="text")

    if vectorization == "Embed_Based":
        vectors = Pp.text_to_embed_vectors(word2vec_model=w2v_model, dataset=training_set, inputcol="text")

    train_list = [0.6, 0.7, 0.8]
    test_list = [0.4, 0.3, 0.2]

    nb = NaiveBayes(smoothing=0.001, modelType="multinomial", labelCol="label", featuresCol="features")
    evaluator = MulticlassClassificationEvaluator().setLabelCol("label"). \
        setPredictionCol("prediction").setMetricName("accuracy")

    # This section calculates the classic accuracy measurement
    for i in range(len(train_list)):
        splits = vectors.randomSplit([train_list[i], test_list[i]], 1234)
        train = splits[0]
        test = splits[1]
        model = nb.fit(train)
        predictions = model.transform(test)
        # predictions.show()

        accuracy = evaluator.evaluate(predictions)
        print(
            "NAIVE BAYES " + vectorization + " train size = " + str(train_list[i]) + " test size " + str(test_list[i]) +
            " Test set accuracy = %g" % (accuracy * 100))

    num_folds = 10
    pipeline = Pipeline(stages=[nb])
    paramgrid = ParamGridBuilder().build()
    crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramgrid,
                              evaluator=evaluator, parallelism=10, numFolds=num_folds)
    cv_model = crossval.fit(vectors)
    evaluator.evaluate(cv_model.transform(vectors))

    pr = cv_model.transform(vectors)
    metric = evaluator.evaluate(pr)
    print("NAIVE BAYES " + vectorization + " Cross Validation K = 10 Fold Accuracy Metric = %g" % (metric * 100))
    print("\n\n\n")
    return cv_model.bestModel
