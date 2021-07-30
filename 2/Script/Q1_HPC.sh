#!/bin/bash
#$ -l h_rt=12:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=40G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o ../Output/Q1_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M srmayekar1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

#spark-submit ../Code/lab6.py
#spark-submit  --executor-memory 32g --num-executors 20 --driver-memory 4g --executor-cores 3 --master local[10] --conf spark.driver.maxResultSize=3g --conf spark.memory.offHeap.enabled=true --conf spark.memory.offHeap.size=40g --conf spark.driver.extraJavaOptions=-Xss800M ../Code/tp.py
spark-submit --driver-memory 40G ../Code/Q1_code.py