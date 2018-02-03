package kaggletoxic

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

/**
  * Created by DT2(M.Kumar) on 1/29/2018.
  */
object KaggleToxicCountWordsFeatureExtraction {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("KaggleToxic")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    val df = spark.read.option("header", "true").option("escape", "\"")
      .csv("file:\\C:\\Users\\m.kumar\\Documents\\Machine-Learning\\Kaggle\\jigsaw-toxic-comment\\train.csv")

    val commentsDf = df.select("comment_text").as[String]

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("value")
      .setOutputCol("filtered")

    val linesToWords = commentsDf.map(_.split("\\s"))

    val usefulWords = stopWordsRemover.transform(linesToWords)
      .withColumn("words",concat_ws(" ",$"filtered"))
      .select("words")
      .as[String]

    val wordsIntoColumn = usefulWords.flatMap(_.split(" ")).map(_.toLowerCase.trim)

    val wordCount = wordsIntoColumn.groupBy("value").count().sort($"count".desc)

    print(wordCount.count)

  }
}
