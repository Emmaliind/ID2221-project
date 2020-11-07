package org.apache.spark.examples.ml

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.mllib.evaluation.MulticlassMetrics


    /********************************************
    Load data from HDFS 
    Use these commands to start HDFS-server
    hdfs --daemon start namenode
    hdfs --daemon start datanode
    dfs -ls /kth 
    **********************************************/

object MainProgram {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("MainProgram")
      .master("local[2]")
      .getOrCreate()

    val testData = spark.read.format("csv").option("header", "true").load("hdfs://localhost:9000/kth/testData.csv")

    

    val columns = Array("Class","Amount", "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13", 
                        "V14","V15","V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27")
     
    val dataImport = testData.select(columns.map(c => col(c).cast("double")): _*)
    val data = dataImport.withColumn("row", monotonically_increasing_id)


    val modelDT = PipelineModel.load("hdfs://localhost:9000/kth/modelDT")
    val modelRF = PipelineModel.load("hdfs://localhost:9000/kth/modelRF")

    // make predictions, we need to have Class column to be able to calculate the error later
    val predictionsDT = modelDT.transform(data)
    val predictionsRF = modelRF.transform(data)


    // select label and prediction and compute test error for DT
    val evaluatorDT = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracyDT = evaluatorDT.evaluate(predictionsDT)
    println(s"Test Error for DT = ${(1.0 - accuracyDT)}") 

    //predictionsDT.select("prediction", "label", "probability").show(10)
    val predictionsDTClean = predictionsDT.select("prediction", "label", "V1", "V2")

     // select label and prediction and compute test error for RF
    val evaluatorRF = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracyRF = evaluatorRF.evaluate(predictionsRF)
    println(s"Test Error for RF = ${(1.0 - accuracyRF)}") 

    predictionsRF.select("prediction", "label", "probability").show(10)
    val predictionsRFClean = predictionsRF.select("prediction", "label", "V1", "V2")

   // write to hdfs
    predictionsDTClean.coalesce(1).write.mode("overwrite").csv("hdfs://localhost:9000/kth/resultsDT.csv")
    predictionsRFClean.coalesce(1).write.mode("overwrite").csv("hdfs://localhost:9000/kth/resultsRF.csv")


  // Print results

   
   def print(): Unit = {
    
   
    val frauds = predictionsDT.select("row", "prediction", "label").filter(predictionsDT("prediction") === "1.0")
    frauds.show(500)

    println("###############################################")
    println("THESE ARE THE DETECTED FRAUDS")
    println()

    println(s"Accuracy for DT was = ${(1.0 - accuracyDT)}") 
    println(s"Accuracy for RF was = ${(1.0 - accuracyRF)}") 
    println()

   }


  print()

  spark.stop(); // spark --> SparkSession
    
   
  }
  
}

