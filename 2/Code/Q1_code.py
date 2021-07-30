from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col,split,rand
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier, MultilayerPerceptronClassifier
import json
import time


spark = SparkSession.builder \
        .master("local[5]") \
        .appName("COM6012 AS2_Q1") \
        .config("spark.local.dir","/fastdata/acp20srm") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

#higgs = spark.read.text("../Data/HIGGS.csv.gz").cache()        
higgs = spark.read.text("/mnt/fastdata/acp20srm/ML/HIGGS.csv.gz")
ncolumns = len(higgs.columns)

split_col = split(higgs['value'], ',')
higgs.drop('value')
data = higgs.withColumn('labels', split_col.getItem(0)) \
       .withColumn('lepton_pT', split_col.getItem(1)) \
       .withColumn('lepton_eta', split_col.getItem(2)) \
       .withColumn('lepton_phi', split_col.getItem(3)) \
       .withColumn('missing_energy_magnitude', split_col.getItem(4)) \
       .withColumn('missing_energy_phi', split_col.getItem(5)) \
       .withColumn('jet_1_pt', split_col.getItem(6)) \
       .withColumn('jet_1_eta', split_col.getItem(7)) \
       .withColumn('jet_1_phi', split_col.getItem(8)) \
       .withColumn('jet_1_b-tag', split_col.getItem(9)) \
       .withColumn('jet_2_pt', split_col.getItem(10)) \
       .withColumn('jet_2_eta', split_col.getItem(11)) \
       .withColumn('jet_2_phi', split_col.getItem(12)) \
       .withColumn('jet_2_b-tag', split_col.getItem(13)) \
       .withColumn('jet_3_pt', split_col.getItem(14)) \
       .withColumn('jet_3_eta', split_col.getItem(15)) \
       .withColumn('jet_3_phi', split_col.getItem(16)) \
       .withColumn('jet_3_b-tag', split_col.getItem(17)) \
       .withColumn('jet_4_pt', split_col.getItem(18)) \
       .withColumn('jet_4_eta', split_col.getItem(19)) \
       .withColumn('jet_4_phi', split_col.getItem(20)) \
       .withColumn('jet_4_b-tag', split_col.getItem(21)) \
       .withColumn('m_jj', split_col.getItem(22)) \
       .withColumn('m_jjj', split_col.getItem(23)) \
       .withColumn('m_lv', split_col.getItem(24)) \
       .withColumn('m_jlv', split_col.getItem(25)) \
       .withColumn('m_bb', split_col.getItem(26)) \
       .withColumn('m_wbb', split_col.getItem(27)) \
       .withColumn('m_wwbb', split_col.getItem(28)) 
#data.printSchema()

StringColumns = [x.name for x in data.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    data = data.withColumn(c, col(c).cast("double"))


random_data = data.sample(withReplacement=False, fraction=0.01)
(trainingSmall, testSmall) = random_data.randomSplit([0.7, 0.3], 42)
trainingSmall.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/higgs_trainingSmall.parquet')
testSmall.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/higgs_testSmall.parquet')


(trainingData, testData) = data.randomSplit([0.7, 0.3], 42)
trainingData.write.mode("overwrite").parquet('Data/higgs_training.parquet')
testData.write.mode("overwrite").parquet('Data/higgs_test.parquet')


trainingSmall = spark.read.parquet('/mnt/fastdata/acp20srm/ML/higgs_trainingSmall.parquet')
testSmall = spark.read.parquet('/mnt/fastdata/acp20srm/ML/higgs_testSmall.parquet')

trainingData = spark.read.parquet('/mnt/fastdata/acp20srm/ML/higgs_training.parquet')
testData = spark.read.parquet('/mnt/fastdata/acp20srm/ML/higgs_test.parquet')


input_columns = ['lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 
'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 
'jet_4_phi', 'jet_4_b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

vecAssembler = VectorAssembler(inputCols = input_columns, outputCol = 'features')

evaluator = BinaryClassificationEvaluator\
      (labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC") 

evaluator2 = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

print('*******************************************************************************************************')
print("Subset of data")
print('*******************************************************************************************************')

#RandomForestClassification 
rfc_sub = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=42)
stages_rfc_sub = [vecAssembler, rfc_sub]
pipeline_rfc_sub = Pipeline(stages=stages_rfc_sub)

# Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
paramGrid_rfc = ParamGridBuilder() \
    .addGrid(rfc_sub.maxDepth, [1, 5, 10]) \
    .addGrid(rfc_sub.numTrees, [1, 5, 10]) \
    .addGrid(rfc_sub.featureSubsetStrategy, ['all','sqrt', 'log2']) \
    .build()

# Make Crossvalidator object, we use the same evaluator as for the previous exercise
crossval = CrossValidator(estimator=pipeline_rfc_sub,
                          estimatorParamMaps=paramGrid_rfc,
                          evaluator=evaluator,
                          numFolds=5)

cvModel_rfc = crossval.fit(trainingSmall)
predictions = cvModel_rfc.transform(testSmall)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for Random Forrest = %g " % accuracy)
print("AUC for Random Forrest = %g " % auc)

paramDict_rfc = {param[0].name: param[1] for param in cvModel_rfc.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_rfc, indent = 4))

print('*******************************************************************************************************')
#GradientBoostingClassification 
gbc_sub = GBTClassifier(labelCol="labels", featuresCol="features", seed=42)
stages_gbc_sub = [vecAssembler, gbc_sub]
pipeline_gbc_sub = Pipeline(stages=stages_gbc_sub)

# Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
paramGrid_gbc = ParamGridBuilder() \
    .addGrid(gbc_sub.maxDepth, [1, 5, 10]) \
    .addGrid(gbc_sub.maxIter, [3, 5, 10]) \
    .addGrid(gbc_sub.maxBins, [5, 10, 20]) \
    .build()

# Make Crossvalidator object, we use the same evaluator as for the previous exercise
crossval = CrossValidator(estimator=pipeline_gbc_sub,
                          estimatorParamMaps=paramGrid_gbc,
                          evaluator=evaluator,
                          numFolds=5)

cvModel_gbc = crossval.fit(trainingSmall)
predictions = cvModel_gbc.transform(testSmall)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for Gradient Boosting = %g " % accuracy)
print("AUC for Gradient Boosting = %g " % auc)

paramDict_gbc = {param[0].name: param[1] for param in cvModel_gbc.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_gbc, indent = 4))

print('*******************************************************************************************************')
#ShallowNeuralNetwork
nn_sub =  MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", seed=42)
stages_nn_sub = [vecAssembler, nn_sub]
pipeline_nn_sub = Pipeline(stages=stages_nn_sub)

# Create the crossvalidator with the different number of layers and nodes in each layer set in the ParamGrid.
paramGrid_nn = ParamGridBuilder() \
            .addGrid(nn_sub.layers, [[len(input_columns),20,5,2],  # The first element HAS to be equal to the number of input features.
                                  [len(input_columns),40,10,2],
                                  [len(input_columns),40,20,2],
                                  [len(input_columns),80,20,2],
                                  [len(input_columns),80,40,20,2]])\
            .addGrid(nn_sub.maxIter,[20,50,100])\
            .addGrid(nn_sub.stepSize,[0.1,0.2,0.3])\
            .build()

# Make Crossvalidator object, we use the same evaluator as for the previous exercise
crossval = CrossValidator(estimator=pipeline_nn_sub,
                          estimatorParamMaps=paramGrid_nn,
                          evaluator=evaluator,
                          numFolds=5)

cvModel_nn = crossval.fit(trainingSmall)
predictions = cvModel_nn.transform(testSmall)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for ShallowNeuralNetwork = %g " % accuracy)
print("AUC for ShallowNeuralNetwork = %g " % auc)

paramDict_nn = {param[0].name: param[1] for param in cvModel_nn.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict_nn, indent = 4))


print('*******************************************************************************************************')
print('*******************************************************************************************************')
print('*******************************************************************************************************')
print('*******************************************************************************************************')
#RandomForestClassifier
#print('RandomForestClassifier')
#rfc = RandomForestClassifier(labelCol="labels", featuresCol="features", maxDepth=paramDict_rfc['maxDepth'],numTrees=paramDict_rfc['numTrees'], \
#                             featureSubsetStrategy=paramDict_rfc['featureSubsetStrategy'],seed=42)
rfc = RandomForestClassifier(labelCol="labels", featuresCol="features", maxDepth=10,numTrees=10, \
                             featureSubsetStrategy="all",seed=42)


stages = [vecAssembler, rfc]
pipeline = Pipeline(stages=stages)

start_time = time.time()
pipelineModel = pipeline.fit(trainingData)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
print(featureImp.sort_values(by="importance", ascending=False).head(3))


predictions = pipelineModel.transform(testData)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for Random Forrest = %g " % accuracy)
print("AUC for Random Forrest = %g " % auc)


print('*******************************************************************************************************')
#GradientBoostingClassification 
print('GradientBoostingClassification')
#gbc = GBTClassifier(labelCol="labels", featuresCol="features", maxDepth=paramDict_gbc['maxDepth'],maxIter=paramDict_gbc['maxIter'], \
#                             maxBins=paramDict_gbc['maxBins'],seed=42)
gbc = GBTClassifier(labelCol="labels", featuresCol="features", maxDepth=5,maxIter=10, \
                             maxBins=20,seed=42)


stages = [vecAssembler, gbc]
pipeline = Pipeline(stages=stages)


start_time = time.time()
pipelineModel = pipeline.fit(trainingData)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
print(featureImp.sort_values(by="importance", ascending=False).head(3))

predictions = pipelineModel.transform(testData)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for Gradient Boosting = %g " % accuracy)
print("AUC for Gradient Boosting = %g " % auc)


print('*******************************************************************************************************')
#ShallowNN
print('ShallowNN')
#nn = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers = paramDict_nn['layers'], maxIter=paramDict_nn['maxIter'],\
#                                    stepSize=paramDict_nn['stepSize'], \
#                                    tol=paramDict_nn['tol'],seed=42)

layers=[28,40,20,2]
nn = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers = layers, maxIter=50,\
                                    stepSize=0.3,seed=42)


stages = [vecAssembler, nn]
pipeline = Pipeline(stages=stages)

start_time = time.time()
pipelineModel = pipeline.fit(trainingData)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

predictions = pipelineModel.transform(testData)

accuracy = evaluator2.evaluate(predictions)
auc = evaluator.evaluate(predictions)
print("Accuracy for Shallow NN = %g " % accuracy)
print("AUC for Shallow NN = %g " % auc)


print("we're done")



