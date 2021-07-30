from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import re
import numpy as np
from pyspark.sql.functions import split, regexp_extract
from pyspark.sql.functions import desc
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignement 1 Question 1") \
        .config("spark.local.dir","/fastdata/acp20srm") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
logFile.show(20, False)

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data.show(20,False)
data = data.drop("request","HTTP reply code","bytes in the reply")

# Preprocessing
split_col = F.split(data['timestamp'], '-')
data = data.withColumn('time', split_col.getItem(0))
split_col = F.split(data['time'], ':')
data = data.withColumn('date', split_col.getItem(0))
data = data.withColumn('hour', split_col.getItem(1))
split_col = F.split(data['date'], '/')
data = data.withColumn('day', split_col.getItem(0))
data = data.drop("timestamp","time","date")
data.show()

# Part A
print("==================== Part A ====================")


# 1 How many requests are from Japanese universities ending with ".ac.jp" ? 
qa1_out = data.filter(F.col('host').endswith('.ac.jp')).cache()
print("==================== Question 1. A. 1) ====================")
print(f"There are {qa1_out.count()} requests are from Japanese universities ending with .ac.jp .")
print("====================================================")

# 2 How many requests are from UK universities ending with ".ac.uk" ? 
qa2_out = data.filter(F.col('host').endswith('.ac.uk')).cache()
print("==================== Question 1.A. 2) ====================")
print(f"There are {qa2_out.count()} requests are from UK universities ending with .ac.uk .")
print("====================================================")

# 3 How many requests are from US universities ending with ".edu" ? 
qa3_out = data.filter(F.col('host').endswith('.edu')).cache()
print("==================== Question 1. A. 3) ====================")
print(f"There are {qa3_out.count()} requests are from US universities ending with .edu .")
print("====================================================")

# plot results
results = [qa1_out.count(),qa2_out.count(),qa3_out.count()]
plt.figure(1)
fig, ax = plt.subplots()

rects = ax.bar(['Japan', 'UK', 'US'], results, label = "Universities")

ax.set_ylabel('Requests')
ax.set_title("Requests per University")
ax.yaxis.set_data_interval(min(results), max(results)+10000, True)
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, -12), textcoords="offset points", ha='center', va='bottom')

plt.savefig("../Output/Q1_figA.png", bbox_inches="tight")


# Part B
print("==================== Part B ====================")

print("==================== Question 1. B. 1) ====================")
# top 9 most frequent universities according to the host domain
# Japan
uni_japan = qa1_out.withColumn('uni', F.split(qa1_out.host, '\.')) \
  .select(F.concat_ws('.',F.element_at(F.col('uni'), -3),F.element_at(F.col('uni'), -2),F.element_at(F.col('uni'), -1))\
  .alias('university'),'hour','day')
top_japan = uni_japan.groupBy("university").count().sort('count', ascending=False)
print(f"top 9 most frequent universities according to the host domain in Japan:")
top_japan.show(9,False)

# UK
uni_uk = qa2_out.withColumn('uni', F.split(qa2_out.host, '\.')) \
  .select(F.concat_ws('.',F.element_at(F.col('uni'), -3),F.element_at(F.col('uni'), -2),F.element_at(F.col('uni'), -1))\
  .alias('university'),'hour','day')
top_uk = uni_uk.groupBy("university").count().sort('count', ascending=False)
print(f"top 9 most frequent universities according to the host domain in the UK:")
top_uk.show(9,False)

# US
uni_us = qa3_out.withColumn('uni', F.split(qa3_out.host, '\.')) \
  .select(F.concat_ws('.',F.element_at(F.col('uni'), -2),F.element_at(F.col('uni'), -1))\
  .alias('university'),'hour','day')
top_us = uni_us.groupBy("university").count().sort('count', ascending=False)
print(f"top 9 most frequent universities according to the host domain in the US:")
top_us.show(9,False)


print("==================== Question 1. B. 2)  ====================")

# Create data to plot pie chart

# Japan 
jp_uni = top_japan.select(F.collect_list('university')).first()[0][0:9]
jp_count = top_japan.select(F.collect_list('count')).first()[0][0:9]
jp_uni_other = np.sum(np.array(top_japan.select(F.collect_list('count')).first()[0][9:]))
jp_uni.append('Others')
jp_count.append(jp_uni_other)

# UK
uk_uni = top_uk.select(F.collect_list('university')).first()[0][0:9]
uk_count = top_uk.select(F.collect_list('count')).first()[0][0:9]
uk_uni_other = np.sum(np.array(top_uk.select(F.collect_list('count')).first()[0][9:]))
uk_uni.append('Others')
uk_count.append(uk_uni_other)

# US
us_uni = top_us.select(F.collect_list('university')).first()[0][0:9]
us_count = top_us.select(F.collect_list('count')).first()[0][0:9]
us_uni_other = np.sum(np.array(top_us.select(F.collect_list('count')).first()[0][9:]))
us_uni.append('Others')
us_count.append(us_uni_other)

# Plot pie chart

# Japan
plt.figure(3)
title = plt.title('Japan')
title.set_ha("center")
plt.gca().axis("equal")
pie = plt.pie(jp_count, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
plt.legend(pie[0],jp_uni, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
plt.savefig("../Output/Q1_figB2_JAPAN.png")

# UK
plt.figure(4)
title = plt.title('UK')
title.set_ha("center")
plt.gca().axis("equal")
pie = plt.pie(uk_count, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
plt.legend(pie[0],uk_uni, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
plt.savefig("../Output/Q1_figB2_Uk.png")

# US
plt.figure(5)
title = plt.title('US')
title.set_ha("center")
plt.gca().axis("equal")
pie = plt.pie(us_count, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
plt.legend(pie[0],us_uni, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.45)
plt.savefig("../Output/Q1_figB2_US.png")

print("Pie Charts created for all 3 regions")


print("==================== Part C ====================")

print("==================== Question 1. C ====================")

# Plot Heatmap

# Japan
jp_filtered = uni_japan.filter(uni_japan.university.isin(jp_uni[0]))
jp_data = jp_filtered.groupby('day', 'hour').count()
end_date = int(jp_data.agg({"day": "max"}).collect()[0][0])
start_date = int(jp_data.agg({"day": "min"}).collect()[0][0])
days = np.arange(start_date, end_date+1)
values = np.zeros((24, end_date+1))
for rows in jp_data.collect():
    day, hour, count = rows
    values[int(hour), int(day)] = count

hours = np.arange(0, 24)

plt.figure(6)
plt.xlabel("Days")
plt.ylabel("Hours")
plt.xticks(ticks=days,labels=days,rotation=90)
plt.yticks(ticks=hours,labels=hours)
heat_map_jp = plt.imshow(values, cmap='Blues',interpolation="nearest")
plt.colorbar(heat_map_jp)
plt.savefig("../Output/Q1_figC_JAPAN.png")

# UK
uk_filtered = uni_uk.filter(uni_uk.university.isin(uk_uni[0]))
uk_data = uk_filtered.groupby('day', 'hour').count()
end_date = int(uk_data.agg({"day": "max"}).collect()[0][0])
start_date = int(uk_data.agg({"day": "min"}).collect()[0][0])
days = np.arange(start_date, end_date+1)
values = np.zeros((24, end_date+1))
for rows in uk_data.collect():
    day, hour, count = rows
    values[int(hour), int(day)] = count


plt.figure(7)
plt.xlabel("Days")
plt.ylabel("Hours")
plt.xticks(ticks=days,labels=days,rotation=90)
plt.yticks(ticks=hours,labels=hours)
heat_map_uk = plt.imshow(values, cmap='Reds',interpolation="nearest")
plt.colorbar(heat_map_uk)
plt.savefig("../Output/Q1_figC_UK.png")

# US
us_filtered = uni_us.filter(uni_us.university.isin(us_uni[0]))
us_data = us_filtered.groupby('day', 'hour').count()
end_date = int(us_data.agg({"day": "max"}).collect()[0][0])
start_date = int(us_data.agg({"day": "min"}).collect()[0][0])
days = np.arange(start_date, end_date+1)
values = np.zeros((24, end_date+1))
for rows in us_data.collect():
    day, hour, count = rows
    values[int(hour), int(day)] = count


plt.figure(8)
plt.xlabel("Days")
plt.ylabel("Hours")
plt.xticks(ticks=days,labels=days,rotation=90)
plt.yticks(ticks=hours,labels=hours)
heat_map_us = plt.imshow(values, cmap='Oranges',interpolation="nearest")
plt.colorbar(heat_map_us)
plt.savefig("../Output/Q1_figC_US.png")
print("Heat maps created for all 3 regions")
