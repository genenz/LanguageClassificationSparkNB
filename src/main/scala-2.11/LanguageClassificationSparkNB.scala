/**
  * Created by gene on 2/2/16.
  *
  * The code is a Apache Spark implementation using mllib's NaiveBayesClassifier.
  *
  * The code uses NaiveBayes from mllib to classify a language as either English, French or Spanish
  * when you provide it a string.
  *
  * This code inspired by a blog post of Burak Kanber (http://burakkanber.com/blog/machine-learning-naive-bayes-1/).

  *
  * Inputs to this program:
  * 1) A file containing language-specific phrases and associated language as the label.  This is used for training the Naive Bayes algorithm.
  * My training set is known as training.txt.  Because I'm lazy, I've just unashamedly stole Burak's (see below) JS code.  Then programmatically
  * cleaned it up to pull his training set.
  *
  * 2) A string of either English, French or Spanish, which you want the program to classify
  *
  */
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{NaiveBayes,NaiveBayesModel}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import scala.collection.mutable.HashMap
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

object LanguageClassificationSparkNB {

  // String you want to Classify
  /*
   * Example Strings
   * Eng: "The social media backlash was sparked by Australian woman Melliiee Hunter, after she published a post a few days ago including photographs of her 9-year-old son and his friend who were left burned following a day at the beach on Australia Day."
   * Fr:  "J'ai dit qu'il fallait que ce soit sur une période suffisamment significative, pour que ce soit crédible, si c'est sur un mois, ça ne sera pas regardé comme étant l'élément déterminant, surtout quand on connaît la fluctuation des statistiques. Donc ce sera sur une période plus longue."
   * Esp: "La Federación de Asociaciones de Padres de Alumnos (FAPAR) sostiene que la jornada partida es mejor que la continua y es el modelo que mejor garantiza la igualdad de oportunidades. FAPAR ha presentado varias alegaciones al borrador de la orden que regula la organización de tiempos escolares, que está ahora en periodo de exposición pública."
   */
  val stringToClassify = "The social media backlash was sparked by Australian woman Melliiee Hunter, after she published a post a few days ago including photographs of her 9-year-old son and his friend who were left burned following a day at the beach on Australia Day."

  // Program Inputs
  val trainingSetLocation = "target/scala-2.11/resource_managed/main/training.txt"    // Location of the training set


  val labels = new HashMap[String,Int]()
  val hashingTF = new HashingTF()
  var labelWordCount = 0

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("LanguageClassificationSparkNB")
    val sc = new SparkContext(conf)

    // Create RDD with training Dataset
    val cleanedRDD = cleanseDataset(sc.textFile(trainingSetLocation))

    // Train the Model
    val model = trainModel(cleanedRDD)

    // Run the prediction
    println("Original String: " + stringToClassify + "\n" + "Predicted Language: " + invertLabel(model.predict(tokenizeAndVectorizeString(stringToClassify))))
  }

  // This takes a string and vectorizes it
  def tokenizeAndVectorizeString(testString: String): Vector = {
    val tokenizedString = tokenize(testString)
    hashingTF.transform(tokenizedString)
  }

  def trainModel(thisRDD: RDD[(String, String)]): NaiveBayesModel = {
    val labeledPointRDD = thisRDD.map({case (label, features) =>
      val numerisedLabel = registerLabel(label)
      val tfVector = tokenizeAndVectorizeString(features) // This takes a String and vectorizes.  HashingTF is a common method to vectorize text: https://spark.apache.org/docs/1.6.0/mllib-feature-extraction.html
      LabeledPoint(numerisedLabel.toDouble, tfVector)
    })
    NaiveBayes.train(labeledPointRDD, 1.0, "multinomial")
  }

  // Records the number of labels/languages we see.  In theory, it allows more than just the 3 languages (of course they'll have to be an alphabet-based language)
  def registerLabel(label: String): Int = {
    if (labels.contains(label))
      labels(label)
    else {
      labels(label) = labelWordCount
      labelWordCount += 1
      (labelWordCount - 1)
    }
  }

  // Reverses the registerLabel
  def invertLabel(num: Int): String = {
    val inverseHash = labels.map(_.swap)
    inverseHash(num)
  }

  // Reverses the registerLabel
  def invertLabel(num: Double): String = {
    val intNum = num.toInt
    val inverseHash = labels.map(_.swap)
    inverseHash(intNum)
  }

  // Cleaning up the training set.
  // This is a bit of a ugly hack.  I was too lazy to manually format the code with a text editor so I programmatically cleaned up the training set.
  // Take a look at the training document and you'll see what I mean.  The output is an RDD with the training string and the associated label.
  // The output of
  def cleanseDataset(fileRDD: RDD[String]): RDD[(String, String)] = {
    val cleanedRDD = fileRDD.filter(line => line.contains("Bayes.train("))
    cleanedRDD.map(line => {
      val trainingStringAndLabel = line.replaceAll("""Bayes.train\("""", "")
      val words = tokenize(trainingStringAndLabel)
      val lang = words.takeRight(1)(0)
      val trainingString = trainingStringAndLabel.replaceAll(""", '""" + lang + """'\);""","")
      (lang,trainingString.trim)
    })
  }

  // Tokenize the string.  This is where most of the data science will be performed.  Our tokenize is pretty simple
  def tokenize(line: String): Seq[String] = {
    line.toLowerCase.replaceAll("""[\p{Punct}]""", " ").replaceAll(" +"," ").trim.split(" ")
  }
}