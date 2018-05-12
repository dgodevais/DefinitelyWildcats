import os
import sys
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from scipy import sparse as sp
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, log_loss, confusion_matrix

DATAFOLDER = sys.argv[1]

# Pyspark related imports
import time
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.linalg import Matrices


spark = SparkSession.builder.appName("Python Spark SQL basic example2").getOrCreate()
# spark.conf.set("spark.executor.memory", '50g')
# spark.conf.set('spark.executor.cores', '2')
# spark.conf.set('spark.cores.max', '2')
# spark.conf.set("spark.driver.memory",'50g')
sc = spark.sparkContext
sqlCtx = SQLContext(spark)

# Load the sparse matrices containing the image feature data
sp_face_features = None
first = True
for filename in os.listdir(os.path.join(DATAFOLDER, 'sparse-images/')):
    fn_path = os.path.join(DATAFOLDER, 'sparse-images/' + filename)
    b = np.load(fn_path)
    data = b['data']
    m_format = b['format']
    shape = b['shape']
    row = b['row']
    col = b['col']
    tmp = sp.csr_matrix( (data,(row,col)), shape=shape )
    if first:
        sp_face_features = sp.vstack((tmp,sp_face_features), format="csr")
    else:
        sp_face_features = tmp
        first = False
print(sp_face_features.shape)

def get_spark_gender_dataframe_from_image_matrix(image_matrix):
    """
    Process the sparse scipy matrix with image features and return a spark dataframe with sparse vectors
    """
    VECTOR_LENGTH = 90000
    spark_rows_formatted = []
    skip_count = 0
    for i, row in enumerate(image_matrix):
        active_cols = row.nonzero()[1]
        if active_cols[0] == 0:
            active_cols = active_cols[1:-2]
        else:
            active_cols = active_cols[:-2]
        indexes = list(map(lambda x: (x, 1), active_cols))
        try:
            gender = int(image_matrix[i,90002])
            spark_rows_formatted.append( (gender, indexes) )
        except ValueError:
            skip_count += 1
    print("Note that {} images were skipped due to nan label.".format(str(skip_count)))
    mapped_f = map(lambda x: (x[0], Vectors.sparse(VECTOR_LENGTH, x[1][1:])), spark_rows_formatted)
    df_gender_analysis = spark.createDataFrame(mapped_f, schema=["label", "features"])
    return df_gender_analysis


df_gender_analysis = get_spark_gender_dataframe_from_image_matrix(sp_face_features)
df_gender_analysis.show(5)

# Prepare the training and test data
splits = df_gender_analysis.randomSplit([0.75, 0.25])
data_train = splits[0]
data_test = splits[1]
print("The training data has {} instances.".format(data_train.count()))
print("The test data has {} instances.".format(data_test.count()))

lr = LogisticRegression(maxIter=10, regParam=0.3)

# Fit the model
lrModel = lr.fit(data_train)
trainingSummary = lrModel.summary
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

predictions = lrModel.transform(data_test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)
evaluator.getMetricName()

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [90000, 10, 10, 2]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(data_train)

# compute accuracy on the test set
result = model.transform(data_test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
