from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank

spark = SparkSession.builder \
        .master("local[8]") \
        .appName("Assignment 1 Question 2") \
        .config("spark.local.dir","/fastdata/acp20srm") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")



from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# load in ratings data
ratings = spark.read.load('../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
sorted = ratings.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("timestamp")))

#sorted = ratings.sort(ratings.timestamp.asc())
sorted.show(20,False)

# loading movies.csv
movie_data = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

myseed=200206622

print("==================== Question 2 A ====================")



def run_model(training, test, als):

    model_train = als.fit(training)
    model_test = als.fit(test)
    predictions = model_train.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("1. Root-mean-square error = " + str(rmse))
    
    evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
    mse = evaluator.evaluate(predictions)
    print("2. Mean-square error = " + str(mse))
    
    evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
    mae = evaluator.evaluate(predictions)
    print("3. Mean-absolute error = " + str(mae))
    
    return model_train.userFactors,model_test.userFactors


# Split the Data

#(training, test) = sorted.randomSplit([0.5, 0.5], myseed)
#training1 = training.cache()
#test1 = test.cache()

#(training, test) = sorted.randomSplit([0.65, 0.35], myseed)
#training2 = training.cache()
#test2 = test.cache()

#(training, test) = sorted.randomSplit([0.80, 0.20], myseed)
#training3 = training.cache()
#test3 = test.cache()

training1 = sorted.where("rank <= .5").drop("rank")
test1 = sorted.where("rank > .5").drop("rank")

training2 = sorted.where("rank <= .65").drop("rank")
test2 = sorted.where("rank > .65").drop("rank")

training3 = sorted.where("rank <= .8").drop("rank")
test3 = sorted.where("rank > .8").drop("rank")



# split 1
print("Results for split 1, 50-50 ")
als1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

# ALS setting used in Lab 3
print("==================== ALS Setting 1 ====================")
userfeatures_train_a1, userfeatures_test_a1 = run_model(training1, test1, als1)

print("==================== ALS Setting 2 ====================")

#Different setting 1
als2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als2.setRank(25)
als2.setMaxIter(10)
userfeatures_train_a2, userfeatures_test_a2 = run_model(training1, test1, als2)

print("==================== ALS Setting 3 ====================")
#Different setting 2
als3 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als3.setRank(15)
als3.setMaxIter(5)
als3.setRegParam(0.001)
userfeatures_train_a3, userfeatures_test_a3  = run_model(training1, test1, als3)

# split 2
print("Results for split 1, 65-35 ")

als4 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
# ALS setting used in Lab 3
print("==================== ALS Setting 1 ====================")
userfeatures_train_b1, userfeatures_test_b1 = run_model(training2, test2, als4)

print("==================== ALS Setting 2 ====================")

#Different setting 1
als5 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als5.setRank(25)
als5.setMaxIter(10)
userfeatures_train_b2, userfeatures_test_b2 = run_model(training2, test2, als5)

print("==================== ALS Setting 3 ====================")
#Different setting 2
als6 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als6.setRank(15)
als6.setMaxIter(5)
als6.setRegParam(0.001)
userfeatures_train_b3, userfeatures_test_b3 = run_model(training2, test2, als6)

# split 3
print("Results for split 1, 80-20 ")

als7 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
# ALS setting used in Lab 3
print("==================== ALS Setting 1 ====================")
userfeatures_train_c1, userfeatures_test_c1  = run_model(training3, test3, als7)

print("==================== ALS Setting 2 ====================")

#Different setting 1
als8 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als8.setRank(25)
als8.setMaxIter(10)
userfeatures_train_c2, userfeatures_test_c2 = run_model(training3, test3, als8)

print("==================== ALS Setting 3 ====================")
#Different setting 2
als9 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
als9.setRank(15)
als9.setMaxIter(5)
als9.setRegParam(0.001)
userfeatures_train_c3, userfeatures_test_c3 = run_model(training3, test3, als9)


print("==================== Question 2 B ====================")

def predict_genres(training, test, userfeatures_train, userfeatures_test, kmeans):
    
    # Prediction on Train
    model_train = kmeans.fit(userfeatures_train.select('features'))
    predictions_train = model_train.transform(userfeatures_train)
    predictions_train.show()
    
    # Prediction on Test
    model_test = kmeans.fit(userfeatures_test.select('features'))
    predictions_test = model_test.transform(userfeatures_test)
    
    # Top clusters
    clusters_train = predictions_train.select('prediction').groupBy('prediction').count().sort('count', ascending=False).limit(3)
    clusters_test = predictions_test.select('prediction').groupBy('prediction').count().sort('count', ascending=False).limit(3)
    print("================= Question 2. B. 1) ================")
    print(f"The 3 largest training clusters are as below: ")
    print("====================================================")
    clusters_train.show()
    
    print("================= Question 2. B. 1) ================")
    print(f"The 3 largest test clusters are as below: ")
    print("====================================================")
    clusters_test.show()
    
    largest_cluster_train = clusters_train.select("prediction").first()['prediction']
    largest_cluster_test = clusters_test.select("prediction").first()['prediction']
    print("====================================================")
    print(f"The largest training cluster is {largest_cluster_train}")
    print("====================================================")
    
    print("====================================================")
    print(f"The largest test cluster is {largest_cluster_test}")
    print("====================================================")
    
    # find users in the largest cluster
    users_train = predictions_train.filter(predictions_train.prediction == largest_cluster_train)
    users_test = predictions_test.filter(predictions_test.prediction == largest_cluster_test)
    
    print(f"The users in largest train cluster are: ")
    users_train.show()
    print(f"The users in largest test cluster are: ")
    users_test.show()
    
    # fetch rating data for the users in the cluster
    user_ratings_train = users_train.join(training, users_train.id == training.userId, how = "inner")
    user_ratings_test = users_test.join(test, users_test.id == test.userId, how = "inner")
    
    # filter the movies which the users have rated 4 and above
    user_ratings_train = user_ratings_train.filter(user_ratings_train.rating >=4 )
    user_ratings_test = user_ratings_test.filter(user_ratings_test.rating >=4 )
    
    user_ratings_train = user_ratings_train.drop("id","features","timestamp","prediction","rating")
    user_ratings_test = user_ratings_test.drop("id","features","timestamp","prediction","rating")
    
    # fetch data from movies.csv
    user_ratings_movie_train = user_ratings_train.join(movie_data, user_ratings_train.movieId == movie_data.movieId, how = "inner")
    user_ratings_movie_test = user_ratings_test.join(movie_data, user_ratings_test.movieId == movie_data.movieId, how = "inner")
    
    user_ratings_movie_train = user_ratings_movie_train.drop("movieId","title")
    user_ratings_movie_test = user_ratings_movie_test.drop("movieId","title")
    
    # collect all the genres for all the movie rated(4 and above) by the user
    final_temp_train = user_ratings_movie_train.groupby("userId").agg(F.concat_ws("|", F.collect_list(user_ratings_movie_train.genres)))
    final_temp_test = user_ratings_movie_test.groupby("userId").agg(F.concat_ws("|", F.collect_list(user_ratings_movie_test.genres)))
    
    final_temp_train = final_temp_train.toDF("userId","genres")
    final_temp_test = final_temp_test.toDF("userId","genres")
    
    final_temp_train = final_temp_train.drop("userId")
    final_temp_test = final_temp_test.drop("userId")
    
    # get the top 5 genres for all the users in the cluster
    final_train = final_temp_train.withColumn("genres", explode(split(col("genres"), "[|]"))) \
            .groupBy("genres").count().sort('count', ascending=False).limit(5)
    final_test = final_temp_test.withColumn("genres", explode(split(col("genres"), "[|]"))) \
            .groupBy("genres").count().sort('count', ascending=False).limit(5)
    
    print("================= Question 2. B. 2  ) ================")
    
    print("====================================================")
    print(f"The top 5 genres for training data are :")
    print("====================================================")
    final_train.show()
    
    print("====================================================")
    print(f"The top 5 genres for test data are :")
    print("====================================================")
    final_test.show()
    

# Split 1
kmeans1 = KMeans().setK(20).setSeed(myseed) 
print("Results for split 1, 50:50 ")
predict_genres(training1, test1, userfeatures_train_a1, userfeatures_test_a1, kmeans1)

# Split 2
kmeans2 = KMeans().setK(20).setSeed(myseed)
print("Results for split 2 65:35 ")
predict_genres(training2, test2, userfeatures_train_b1, userfeatures_test_b1, kmeans2)

# Split 3
kmeans3 = KMeans().setK(20).setSeed(myseed)
print("Results for split 3, 80:20 ")
predict_genres(training2, test3, userfeatures_train_c1, userfeatures_test_c1, kmeans3)