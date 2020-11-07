package org.apache.spark.examples.ml

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.PipelineModel


object ModelTraning {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("ModelTraning")
      .master("local[2]")
      .getOrCreate()

    // Load data from HDFS 
    // Use these commands to start HDFS-server
    // $HADOOP_HOME/bin/hdfs --daemon start namenode
    // $HADOOP_HOME/bin/hdfs --daemon start datanode
    // hds dfs -ls /kth

    val rawData = spark.read.format("csv").option("header", "true").load("hdfs://localhost:9000/kth/creditcard.csv")

    // selected columns to use when traning
    val columns = Array("Class","Amount", "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15", // "Amount" "V6","V7","V8","V9","V10","V11","V12","V13","V14","V15",
                        "V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28")

    
    /******************** TEST HOW MANY FEATURES TO USE ***********************/
    /*
    This is done by testing different number of features (min 3). We train the models with x number of features.
    Then evaluate the models by running 10 test and caluculate the mean value error and then store the value into a matrix. 
    After several number of features are tested the number of features with lowest error is chosen. 
    Do not necessary need to test both models but now we do, but we only choose the bbest value for DT, maybe 
    we can change so the models are trained with their respectivly best number of features. 
    */
    
    var x = 3
    var matrix = Array.ofDim[Double](columns.length-3,2) // matrix to save the mean value of each combination
    while(x < columns.length){ 
      val numOfCols = columns.take(x)
      val numOfColsPred = numOfCols.drop(1) // without Class

      // prepocessing of data, extract selected columns and cast to double 
      val data = rawData.select(numOfCols.map(c => col(c).cast("double")): _*)
      val splits = data.randomSplit(Array(0.7, 0.3))
      val (trainingData, testData) = (splits(0), splits(1))

      // train models with x number of features 
      // DT
      val assembler = new VectorAssembler().setInputCols(numOfColsPred).setOutputCol("features")
      val labelIndexer = new StringIndexer().setInputCol("Class").setOutputCol("label")
      val dt = new DecisionTreeClassifier()
      val pipelineDT = new Pipeline().setStages(Array(assembler, labelIndexer, dt))
      val modelDT = pipelineDT.fit(trainingData)

      // RF
      val rf = new RandomForestClassifier()
      val pipelineRF = new Pipeline().setStages(Array(assembler, labelIndexer, rf))
      val modelRF = pipelineRF.fit(trainingData)
      
      // evaluator DT
      val evaluatorDT = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      // evaluator RF
      val evaluatorRF = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
        
   
      // initialize variables to calculate mean
      var y = 0
      var meanDT = 0.0
      var sumDT = 0.0
      var meanRF = 0.0
      var sumRF = 0.0
      // run 10 test and find mean value for x number of features
      while(y < 10){ 
        val predictionsDT = modelDT.transform(testData)
        var accuracyDT = evaluatorDT.evaluate(predictionsDT)
        val predictionsRF = modelRF.transform(testData)
        val accuracyRF = evaluatorRF.evaluate(predictionsRF)
        sumDT = sumDT + (1.0 - accuracyDT)
        sumRF = sumRF + (1.0 - accuracyRF)
        y = y + 1
      }
      meanDT = sumDT/10.0 // divide with 10 to get mean, 10 tests
      matrix(x-3)(0) = meanDT // add the mean to the matrix
      meanRF = sumRF/10.0
      matrix(x-3)(1) = meanRF // add the mean to the matrix

      x = x + 1
    }

    // find min value for DT and RT and corresponding number of features 
    var minValDT = 1.0
    var minValRF = 1.0
    var numberOfFeaturesDT = 0
    var numberOfFeaturesRF = 0
    for( a <- 0 to columns.length-4){
      if(matrix(a)(0)< minValDT){
        minValDT = matrix(a)(0)
        numberOfFeaturesDT = a + 3
      }
      if(matrix(a)(1)< minValRF){
        minValRF = matrix(a)(1)
        numberOfFeaturesRF = a + 3
      }
    } 

    
    /******************* TRAIN THE MODELS WITH MOST OPTIMAL NUMBER OF FEATURES FOUND ABOVE ***********************/


    val numOfCols = columns.take(numberOfFeaturesDT) // now we choose what is best for DT and use it for both DT and RF
    val numOfColsPred = numOfCols.drop(1) // without Class

    // prepocessing of data, extract selected columns and cast to double 
    val data = rawData.select(numOfCols.map(c => col(c).cast("double")): _*)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // train models with x number of features 
    val assembler = new VectorAssembler().setInputCols(numOfColsPred).setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("Class").setOutputCol("label")
    // DT
    val dt = new DecisionTreeClassifier()
    val pipelineDT = new Pipeline().setStages(Array(assembler, labelIndexer, dt))
    val modelDT = pipelineDT.fit(trainingData)
    // RF
    val rf = new RandomForestClassifier()
    val pipelineRF = new Pipeline().setStages(Array(assembler, labelIndexer, rf))
    val modelRF = pipelineRF.fit(trainingData)
    
    // write models to hdfs
    modelDT.write.overwrite().save("hdfs://localhost:9000/kth/modelDT")
    modelRF.write.overwrite().save("hdfs://localhost:9000/kth/modelRF")
    // write testdata to hdfs also to know what features to use when testing 
    testData.write.format("csv").option("header","true").mode("overwrite").csv("hdfs://localhost:9000/kth/testData.csv") 

    println("##########################################################")
    println(numberOfFeaturesDT) 
    println(numberOfFeaturesRF) 

    spark.stop(); // spark --> SparkSession
    
   
  }
}
// scalastyle:on println