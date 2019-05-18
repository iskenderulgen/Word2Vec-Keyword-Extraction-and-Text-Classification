from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import Pre_Process as Pp


def decision_tree_model(training_set, w2v_model, vectorization=None):
    training_set = Pp.string_to_label(training_set, inputcol="string_label")

    vectors = None

    if vectorization == "Frequency_Based":
        vectors = Pp.text_to_freq_vectors(training_set, state="TF", inputcol="text")

    if vectorization == "Embed_Based":
        vectors = Pp.text_to_embed_vectors(word2vec_model=w2v_model, dataset=training_set, inputcol="text")

    train_list = [0.6, 0.7, 0.8]
    test_list = [0.4, 0.3, 0.2]
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=10,
                                maxBins=32, maxMemoryInMB=1024, impurity="gini")
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")

    for i in range(len(train_list)):
        splits = vectors.randomSplit([train_list[i], test_list[i]], 1234)
        train = splits[0]
        test = splits[1]
        model = dt.fit(train)
        predictions = model.transform(test)
        # predictions.show()

        accuracy = evaluator.evaluate(predictions)
        print("DECISION TREE " + vectorization + " train size = " + str(train_list[i]) + " test size " + str(
            test_list[i]) +
              " Test set accuracy = %g" % (accuracy * 100))

    num_folds = 10
    pipeline = Pipeline(stages=[dt])
    paramgrid = ParamGridBuilder().build()
    crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramgrid,
                              evaluator=evaluator, parallelism=10, numFolds=num_folds)
    cv_model = crossval.fit(vectors)
    evaluator.evaluate(cv_model.transform(vectors))

    pr = cv_model.transform(vectors)
    metric = evaluator.evaluate(pr)
    print("DECISION TREE " + vectorization + " Validation K = 10 Fold Accuracy Metric = %g" % (metric * 100))
    print("\n\n\n")
    return cv_model.bestModel
