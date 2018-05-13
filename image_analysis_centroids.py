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
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.linalg import Matrices

NUM_FEATURES = 90000
GENDER_INDEX = NUM_FEATURES + 2
AGE_INDEX = NUM_FEATURES + 1
NUM_F_SETS = 9
COLS_PER_SET = 10000
PIC_DIM = 100
FIRST_COL_OFFSET = 1
NOSE_BRIDGE_INDEX = 3

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
for filename in os.listdir(DATAFOLDER):
    fn_path = os.path.join(DATAFOLDER, filename)
    b = np.load(fn_path)
    data = b['data']
    m_format = b['format']
    shape = b['shape']
    row = b['row']
    col = b['col']
    tmp = sp.csr_matrix((data, (row, col)), shape=shape)
    if first:
        sp_face_features = sp.vstack((tmp, sp_face_features), format="csr")
    else:
        sp_face_features = tmp
        first = False
print(sp_face_features.shape)


def get_x_y_coord(z):
    z = z - FIRST_COL_OFFSET
    x = (z - ((z // COLS_PER_SET) * COLS_PER_SET)) % PIC_DIM
    y = (z - ((z // COLS_PER_SET) * COLS_PER_SET)) // PIC_DIM
    feature_set = z // COLS_PER_SET
    return (x, y, feature_set)


def get_distance_features(active_cols):
    coords = list(map(lambda z: get_x_y_coord(z), active_cols))
    np_coords = np.array(coords)
    distances = []
    nose_feature_set = np_coords[np_coords[:, 2] == NOSE_BRIDGE_INDEX]
    for i in range(0, NUM_F_SETS):
        feature_set = np_coords[np_coords[:, 2] == i]
        if i != 2:
            if len(nose_feature_set) == 0 or len(feature_set) == 0:
                distances.append((NUM_FEATURES + i, 0))
            else:
                nose_centroid = np.mean(nose_feature_set, axis=0)[0:2]
                feature_centroid = np.mean(feature_set, axis=0)[0:2]
                dist = np.linalg.norm(nose_centroid - feature_centroid)
                distances.append((NUM_FEATURES + i, dist))
    return distances


def get_spark_gender_dataframe_from_image_matrix(image_matrix, label_index):
    """
    Process the sparse scipy matrix with image features and return a spark dataframe with sparse vectors
    """
    spark_rows_formatted = []
    skip_count = 0
    for i, row in enumerate(image_matrix):
        active_cols = row.nonzero()[1]
        # Remove first column if index col and remove last two label columns
        if active_cols[0] == 0:
            active_cols = active_cols[1:-2]
        else:
            active_cols = active_cols[:-2]
        indexes = list(map(lambda z: (z, 1), active_cols))
        indexes += get_distance_features(active_cols)
        try:
            label = int(image_matrix[i, label_index])
            spark_rows_formatted.append((label, indexes))
        except ValueError:
            skip_count += 1
    print("Note that {} images were skipped due to nan label.".format(str(skip_count)))
    mapped_f = map(lambda x: (x[0], Vectors.sparse(NUM_FEATURES + NUM_F_SETS, x[1][1:])),
                   spark_rows_formatted)
    df_analysis = spark.createDataFrame(mapped_f, schema=["label", "features"])
    return df_analysis


df_gender_analysis = get_spark_gender_dataframe_from_image_matrix(sp_face_features, GENDER_INDEX)
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


# Evaluate our faces
fn_path = os.path.join(DATAFOLDER, '1_our_faces_sparse.npz')
b = np.load(fn_path)
data = b['data']
m_format = b['format']
shape = b['shape']
row = b['row']
col = b['col']
sp_face_features_me = sp.csr_matrix((data, (row, col)), shape=shape)
df_age_analysis_me = get_spark_gender_dataframe_from_image_matrix(sp_face_features_me,
                                                                  GENDER_INDEX)
df_age_analysis_me.show(5)

predictions = lrModel.transform(df_age_analysis_me)
predictions.show()
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)


# df_age_analysis = get_spark_gender_dataframe_from_image_matrix(sp_face_features, AGE_INDEX)
# df_age_analysis.show(5)
#
# # Prepare the training and test data
# splits = df_age_analysis.randomSplit([0.75, 0.25])
# data_train = splits[0]
# data_test = splits[1]
# print("The training data has {} instances.".format(data_train.count()))
# print("The test data has {} instances.".format(data_test.count()))
#
# lr = LinearRegression(maxIter=10, regParam=0.3)
#
# # Fit the model
# lrModel = lr.fit(data_train)
# trainingSummary = lrModel.summary
# trainingSummary.roc.show()
# print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
#
# predictions = lrModel.transform(data_test)
# evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
# evaluator.evaluate(predictions)
# evaluator.getMetricName()
