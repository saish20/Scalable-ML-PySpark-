import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StringIndexer,OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier
from pyspark.ml.regression import LinearRegression,GeneralizedLinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import numpy as np 
import time

spark = SparkSession.builder \
    .master("local[10]") \
    .appName("COM6012 AS2_Q2") \
    .config("spark.local.dir", "/fastdata/acp20srm") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

df = spark.read.load('../Data/train_set.csv',  format="csv", inferSchema="true", header="true")
#df.show()

# 1.	Preprocessing

df2=df.drop('Row_ID','Household_ID')

# Converting numeric columns to double 
df3 = df2.withColumn("Vehicle",df2["Vehicle"].cast('double'))\
                                  .withColumn("Calendar_Year",df2["Calendar_Year"].cast('double'))\
                                  .withColumn("Model_Year",df2["Model_Year"].cast('double'))

# Getting max values for each column which will be assigned to the missing values
param={}
for c in df3.columns:
    temp = df3.groupby(c).count().sort(F.col("count").desc()).take(1) 
    max_value = temp[0][0]
    param[c]=max_value

# Substituting "?" with Null to be replaced with max value from the column
df4 = df3.select([when(col(c)=="?",None).otherwise(col(c)).alias(c) for c in df3.columns])

df5 = df4.select([when(col(c).isNull(),param[c]).otherwise(col(c)).alias(c) for c in df4.columns])

# Dropping categorical columns
df5 = df5.drop('Cat1','Cat2','Cat3','Cat4','Cat5',\
               'Cat6','Cat7','Cat8','Cat10','Cat11')

# 1. b. Convert categorical values to a suitable representation
# Applying String indexer to string columns
blind_make_indexer = StringIndexer(inputCol="Blind_Make", outputCol="blindmakeindex")
blind_model_indexer = StringIndexer(inputCol="Blind_Model", outputCol="blindmodelindex")
#blind_submodel_indexer = StringIndexer(inputCol="Blind_Submodel", outputCol="blindsubmodelindex")
#cat9_indexer = StringIndexer(inputCol="Cat9", outputCol="cat9index")
cat12_indexer = StringIndexer(inputCol="Cat12", outputCol="cat12index")
ordCat_indexer = StringIndexer(inputCol="OrdCat", outputCol="ordcatIndex")
#nvcat_indexer = StringIndexer(inputCol="NVCat", outputCol="nvcatIndex")

# Applying One hot encoding to the output of string indexer
blind_make_vector = OneHotEncoder(inputCol="blindmakeindex", outputCol="blindmake_vec")
blind_model_vector = OneHotEncoder(inputCol="blindmodelindex", outputCol="blindmodel_vec")
#blind_submodel_vector = OneHotEncoder(inputCol="blindsubmodelindex", outputCol="blindsubmodel_vec")
#cat9_vector = OneHotEncoder(inputCol="cat9index", outputCol="cat9_vec")
cat12_vector = OneHotEncoder(inputCol="cat12index", outputCol="cat12_vec")
ordcat_vector = OneHotEncoder(inputCol="ordcatIndex", outputCol="ordcat_vec")
#nvcat_vector = OneHotEncoder(inputCol="nvcatIndex", outputCol="nvcat_vec")
#vehicle_vector = OneHotEncoder(inputCol="Vehicle", outputCol="vehicle_vec")
#calendar_year_vector = OneHotEncoder(inputCol="Calendar_Year", outputCol="calendaryear_vec")
#model_year_vector = OneHotEncoder(inputCol="Model_Year", outputCol="modelyear_vec")

"""
stages_preprocessing = [blind_make_indexer,blind_model_indexer,blind_submodel_indexer,cat9_indexer,cat12_indexer,ordCat_indexer,nvcat_indexer,\
blind_make_vector,blind_model_vector,blind_submodel_vector,cat9_vector,cat12_vector,ordcat_vector,nvcat_vector,vehicle_vector,\
calendar_year_vector,model_year_vector]
"""

# Pipeline for data preprocessing
stages_preprocessing = [blind_make_indexer,blind_model_indexer,cat12_indexer,ordCat_indexer,\
blind_make_vector,blind_model_vector,cat12_vector,ordcat_vector]

preprocessing_pipeline = Pipeline(stages=stages_preprocessing)

# Applying the pipeline on data
df_transformed = preprocessing_pipeline.fit(df5).transform(df5)

# Saving the preocessed data
df_transformed.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/as2.parquet')

# 1. c. Balancing the data
# Transforming column to binary
df_transformed = df_transformed.withColumn("claim_derived", when(col("Claim_Amount") == 0,0)
                .otherwise(1))

# Creating splits
(trainingData_unbalanced, testData) = df_transformed.randomSplit([0.7, 0.3], 42)
trainingData_unbalanced.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/as2_training_ub.parquet')

# Sampling the train data
trainingData = trainingData_unbalanced.sampleBy("claim_derived", fractions={0: 0.14, 1: 1}, seed=42)

# Saving the train and test data
trainingData.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/as2_training.parquet')
testData.write.mode("overwrite").parquet('/mnt/fastdata/acp20srm/ML/as2_test.parquet')

# Reading the saved train and test data
trainingData = spark.read.parquet('/mnt/fastdata/acp20srm/ML/as2_training.parquet')
testData = spark.read.parquet('/mnt/fastdata/acp20srm/ML/as2_test.parquet')


# Columns to get vectorised features
input_columns = ['Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4' ]


# 2. Linear Regression
print("Linear Regression")
trainingData_lr = trainingData.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4','Claim_Amount')

testData_lr = testData.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4','Claim_Amount')

                
vecAssembler = VectorAssembler(inputCols = input_columns, outputCol = 'features') 
lr = LinearRegression(featuresCol='features', labelCol='Claim_Amount')
stageslr = [vecAssembler, lr]
pipelinelr = Pipeline(stages=stageslr)

start_time = time.time()
pipelineModellr = pipelinelr.fit(trainingData_lr)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

predictions = pipelineModellr.transform(testData_lr)

# Performance Metrics
evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
print("MSE LR: %f" % mse)


evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print("MAE LR: %f" % mae)

# 3. Tandem Model
# GBTClassifier
print("GBTClassifier")

trainingData_gbc = trainingData.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4', 'Claim_Amount', 'claim_derived')

trainData_glm_temp = trainingData_gbc.filter(trainingData_gbc["claim_derived"] == 1)

testData_gbc = testData.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4', 'Claim_Amount', 'claim_derived')


gbc = GBTClassifier(labelCol="claim_derived", featuresCol="features", maxDepth=10,maxIter=50, \
                           maxBins=20,stepSize=0.1,validationTol=0.01,subsamplingRate=1.0,seed=42)


stagesgbc = [vecAssembler, gbc]
pipelinegbc = Pipeline(stages=stagesgbc)

start_time = time.time()
pipelineModelgbc = pipelinegbc.fit(trainingData_gbc)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

predictionsgbc = pipelineModelgbc.transform(testData_gbc)

"""
evaluator2 = MulticlassClassificationEvaluator\
      (labelCol="claim_derived", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator2.evaluate(predictionsgbc)

print("Accuracy for Random Forrest = %g " % accuracy)
"""

evaluator = RegressionEvaluator\
      (labelCol="claim_derived", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictionsgbc)
print("MSE GBC: %f" % mse)


evaluator = RegressionEvaluator\
      (labelCol="claim_derived", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictionsgbc)
print("MAE GBC: %f" % mae)

# GeneralizedLinearRegression
print("GeneralizedLinearRegression")

"""
data_glm_temp = predictionsrfc.select("features","prediction",'Claim_Amount')
testData_glm= data_glm_temp.filter(data_glm_temp["prediction"] == 1)
trainingData_glm= trainingData_rfc.filter(trainingData_rfc["claim_derived"] == 1)
"""


testData_glm_temp = predictionsgbc.filter(predictionsgbc["prediction"] == 1)
testData_glm  = testData_glm_temp.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4','Claim_Amount')

#testData_glm_temp = trainingData_rfc.filter(trainingData_rfc["claim_derived"] == 1)
trainData_glm = trainData_glm_temp.select('Vehicle','Calendar_Year','Model_Year','blindmake_vec','blindmodel_vec',\
                'cat12_vec','ordcat_vec','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4','Claim_Amount')


glm = GeneralizedLinearRegression(labelCol='Claim_Amount', featuresCol='features', predictionCol='prediction',family="gamma", link="identity", maxIter=50, regParam=0.1)

stagesglm = [vecAssembler, glm]
pipelineglm = Pipeline(stages=stagesglm)


start_time = time.time()
modelgbc = pipelineglm.fit(trainData_glm)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))

predictionsglm = modelgbc.transform(testData_glm)

# Performance Metrics
evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictionsglm)
print("MSE: %f" % mse)


evaluator = RegressionEvaluator\
      (labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictionsglm)
print("MAE: %f" % mae)


spark.stop()