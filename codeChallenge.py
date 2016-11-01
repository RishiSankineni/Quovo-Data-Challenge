
# coding: utf-8

# In[15]:

"""
Hi

Thank you for taking the time to apply to Quovo. We like to send potential candidates a SHORT coding test/exercise so 
we could get a sense of how they approach problems. This also gives you the a good opportunity to see if Quovo-style 
challenges are a good fit for you. Don't go crazy on time, we'd just like to see enough progress on it where we can 
all have a conversation looking at your code together and talk about how you attacked the problem.

The concept:

In each row of the included datasets, products X and Y are considered to refer to the same security if 
they have the same ticker, even if the descriptions don't exactly match. 

Your challenge is to use these descriptions to predict whether each pair in the test set also refers to the 
same security. The difficulty of predicting each row will vary significantly, so please do not aim for 100% accuracy. 
There are several good ways to approach this, and we have no preference between them. 
The only requirement is that you do all of the work in this file, and return it to us.

Hint: Don't be afraid if you have no experience with text processing. You are in the majority of applicants. Check out this algorithm, 
and see how far you can go with it:
https://en.wikipedia.org/wiki/Tf–idf
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Good luck!
"""

#import pandas as pd
#from IPython.display import display

#train = pd.DataFrame.from_csv('code_challenge_train.csv')
#test = pd.DataFrame.from_csv('code_challenge_test.csv')

#display(train)

---- MY CODE ------

import pandas as pd
from IPython.display import display
import numpy as np
import numpy.linalg as LA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

-- ## Loading the data ###
Train_data = pd.read_csv("/dbfs/FileStore/tables/nfg5a3np1477092781030/code_challenge_train.csv")
docsY = Train_data['description_y'].tolist()
docsX = Train_data['description_x'].tolist()
docsTX = Train_data['ticker_x'].tolist()
docsTY = Train_data['ticker_y'].tolist()
docsSec = Train_data['same_security'].tolist()
Train_data.shape

#docs = docsX
#docs.append(docsY)

#docsT = docsTX
#docsT.append(docsTY)

--- ## Value Count for each column ###
Train_data['description_x'].value_counts()
Train_data['description_y'].value_counts()
Train_data['ticker_x'].value_counts()
Train_data['ticker_y'].value_counts()


#vectorizerone = CountVectorizer(stop_words='english').fit(['docs','docsT'])

--- ## Assign stop words and fit the data accordingly to the CountVectorizer ##

vectorizerone = CountVectorizer(stop_words='english').fit(['docsX','docsY'])
vectorizertwo = CountVectorizer(stop_words='english').fit(['docsTX','docsTY'])
vectorizerthree = CountVectorizer(stop_words='None').fit('docsSec') ## No stop words assigned for Boolean Values.

ticker_freq_matrix = vectorizertwo.fit_transform(docsTX,docsTY)
print("TF of Tickers:",vectorizertwo.vocabulary_) ## Prints the Vocabulary of Ticker X and Ticker Y from the train set
tf_idf_matrixtwo = tfidftwo.transform(ticker_freq_matrix)
print(tf_idf_matrixtwo.todense()) ## Converting it to a dense matrix

doc_term_matrix = vectorizerone.fit_transform(docsX,docsY)
print("TF of Descriptions:", vectorizerone.vocabulary_) ## Prints the Vocabulary of Description X and Description Y from the train set
tf_idf_matrix = tfidfsix.transform(doc_term_matrix)
print (tf_idf_matrix.todense()) ## Converting it to a dense matrix

sec_term_matrix = vectorizerthree.fit_transform(docsSec) ## Prints the Vocabulary of same_security from the train set
print("TF of Security label:",vectorizerthree.vocabulary_)


## Calculating the TFIDF Score##
##Now that we have the description term matrix (called doc_term_matrix) and Ticker term matrix(called ticker_freq_matrix) we can instantiate the TfidfTransformer, which is going to be responsible to calculate the tf-idf weights for our term frequency matrix ##

from sklearn.feature_extraction.text import TfidfTransformer

tfidfsix = TfidfTransformer(norm="l2") ##Note that I’ve specified the norm as L2, this is optional (actually the default is L2-norm), but I’ve added the parameter to make it explicit to you that it it’s going to use the L2-norm ##
tfidfsix.fit(doc_term_matrix)
print ("IDF score:", tfidfsix.idf_)
## Also note that you can see the calculated idf weight by accessing the internal attribute called idf_. Now that fit() method has calculated the idf for the matrix, let’s transform the doc_term_matrix to the tf-idf weight matrix:
tf_idf_matrix = tfidfsix.transform(doc_term_matrix)
print (tf_idf_matrix.todense())


tfidftwo = TfidfTransformer(norm="l2")  ##Note that I’ve specified the norm as L2, this is optional (actually the default is L2-norm), but I’ve added the parameter to make it explicit to you that it it’s going to use the L2-norm ##
tfidftwo.fit(ticker_freq_matrix)
print ("IDF score:", tfidftwo.idf_)
## Also note that you can see the calculated idf weight by accessing the internal attribute called idf_. Now that fit() method has calculated the idf for the matrix, let’s transform the ticker_freq_matrix to the tf-idf weight matrix:
tf_idf_matrixtwo = tfidftwo.transform(ticker_freq_matrix)
print(tf_idf_matrixtwo.todense())


## Loading the TEST DATA ##

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv("/dbfs/FileStore/tables/1jan81qj1477098178857/code_challenge_test.csv")
docstest = data['description_x'].tolist()
docstestone = data['description_y'].tolist()

--- ## Assign stop words and fit the data accordingly to the CountVectorizer ##

vectorizertest = CountVectorizer(stop_words='english')
document_term_matrixnew = vectorizertest.fit_transform(docstest,docstestone)
print ("TF of Description X and Y:" ,vectorizertest.vocabulary_)
freq_term_matrix = vectorizertest.transform(docstest,docstestone)
print (freq_term_matrix.todense())

#vectorizertestone = CountVectorizer(stop_words='english')
#document_term_matrixtwo = vectorizertestone.fit_transform(docstestone)
#print ("TF of Description Y:" ,vectorizertest.vocabulary_)
#freq_term_matrixnew = vectorizertestone.transform(docstestone)
#print (freq_term_matrixnew.todense())

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
print ("IDF score:", tfidf.idf_)
tf_idf_matrix = tfidf.transform(freq_term_matrix)
print (tf_idf_matrix.todense())
tf_idf_matrix.shape

#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf = TfidfTransformer(norm="l2")
#tfidf.fit(freq_term_matrixnew)
#print ("IDF score:", tfidf.idf_)
#tf_idf_matrixtwo = tfidf.transform(freq_term_matrixnew)
#print (tf_idf_matrixtwo.todense())
#tf_idf_matrixtwo.shape



## Calculating Cosine Similarities ##

trainVectorizerArray = vectorizerone.fit_transform(docsX,docsY).toarray()
testVectorizerArray = vectorizertest.fit_transform(docstest,docstestone).toarray()
print ('Fit Vectorizer to train set', trainVectorizerArray)
print ('Transform Vectorizer to test set', testVectorizerArray)
transformer = TfidfTransformer()
transformer.fit(trainVectorizerArray)
print
print(transformer.transform(trainVectorizerArray).toarray())
transformer.fit(testVectorizerArray)
print
tfidf = transformer.transform(testVectorizerArray)
print tfidf.todense()
tfidf[0:1]
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf[0:1] , tfidf).flatten()
cosine_similarities
related_docs_indices = cosine_similarities.argsort()[:-2000:-1]
related_docs_indices
cosine_similarities[related_docs_indices]

## I tried my best to predict the same_security for the test set using this approach, but, i was not able to do so. Hence, i tried another approach(code below) ##


                          ## %md Another Approach ##

import nltk
import numpy as np
from IPython.display import display
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

## Loading Train Data ##

Train_data = pd.read_csv("/Users/Rishi/downloads/data_sci_code_challenge/code_challenge_train.csv")
docsY = Train_data['description_y'] #.tolist()
docsX = Train_data['description_x'] #.tolist()
docsTX = Train_data['ticker_x'] #.tolist()
docsTY = Train_data['ticker_y'] #.tolist()
docsSec = Train_data['same_security'].tolist()

docs = docsX
docs.append(docsY) ## append Desc X and Desc Y to docs
docsT = docsTX
docsT.append(docsTY) ## append Ticker X and Ticker Y to docsT
Train_data.shape

## Value count for each column ##

Train_data['description_x'].value_counts()
Train_data['description_y'].value_counts()
Train_data['ticker_x'].value_counts()
Train_data['ticker_y'].value_counts()
docsSec = Train_data['same_security'].value_counts()

vectorizerone = CountVectorizer(stop_words='english').fit(['docs','docsT']) ### Fit both docs and docsT
doc_term_matrix = vectorizerone.fit_transform(docs,docsT)## Fit and transform
doc_term_matrix.shape
doc_new_matrix = doc_term_matrix.todense ## Dense matrix
doc_new_matrix

## loading test data ###


Test_data = pd.read_csv("/Users/Rishi/downloads/data_sci_code_challenge/code_challenge_test.csv")
docstest = Test_data['description_x'] #.tolist()
docstestone = Test_data['description_y'] #.tolist()
docstestSec = Test_data['same_security']
docsTN = docstest
docsTN.append(docstestone)


from sklearn.feature_extraction.text import CountVectorizer
vectorizertest = CountVectorizer(stop_words='english').fit(['docsTN'])
new_test_matrix = vectorizertest.transform(docsTN)
new_test = new_test_matrix.todense
print(new_test)


### Random forest classifier for predicting the same_secuirty label of test data ##

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(doc_term_matrix, Train_data["same_security"])
result = forest.predict(docstestSec)
display(result)

## Error: Input and output values don't match ## - Not able to figure it out.


---------------------- ## I have attached a sample lab code of mine. Please go through it ######

---------------------- ## Power Plant Machine Learning Pipeline Application (Linear Regression Model) ## ----------------------------------

#Our goal is to accurately predict power output given a set of environmental readings from various sensors in a natural gas-fired power generation plant.*
#Here is an real-world example of predicted demand (on two time scales), actual demand, and available resources from the California power grid: http://www.caiso.com/Pages/TodaysOutlook.aspx


### This code is written using Apache Spark's Python API on Databricks Notebook. So, you might have to use Databricks community edition account for executing this code.###

display(dbutils.fs.ls("/databricks-datasets/power-plant/data"))
print dbutils.fs.head("/databricks-datasets/power-plant/data/Sheet1.tsv")

## Create and RDD(Resilient Distributed Datasets) ##

rawTextRdd = sc.textFile("dbfs:/databricks-datasets/power-plant/data")
print rawTextRdd.take(5)
rawTextRdd.count()
powerPlantDF = sqlContext.read.format('com.databricks.spark.csv').options(delimiter ='\t',header = 'true', inferschema = 'true').load("/databricks-datasets/power-plant/data")
powerPlantDF.show()
powerPlantDF.count()
print powerPlantDF.dtypes
display(powerPlantDF)

from pyspark.sql.types import *

# Custom Schema for Power Plant
customSchema = StructType ([ StructField ('AT', DoubleType(), True), StructField ('V', DoubleType(), True), StructField ('AP', DoubleType(), True), StructField ('RH', DoubleType(), True),StructField ('PE', DoubleType(), True) ])
altPowerPlantDF = sqlContext.read.format('com.databricks.spark.csv').options(delimiter='\t',header = 'true').load("/databricks-datasets/power-plant/data/",schema = customSchema)
altPowerPlantDF.show()
print altPowerPlantDF.dtypes
display(altPowerPlantDF)

sqlContext.sql("DROP TABLE IF EXISTS power_plant")
dbutils.fs.rm("dbfs:/user/hive/warehouse/power_plant", True)
sqlContext.registerDataFrameAsTable(powerPlantDF,"power_plant")

%sql
-- We can use %sql to query the rows
SELECT * FROM power_plant
%sql
desc power_plant
df = sqlContext.table("power_plant")
display(df.describe())

from pyspark.ml.feature import VectorAssembler

datasetDF = df

vectorizer = VectorAssembler()
vectorizer.setInputCols(["AT","V","AP","RH"])
vectorizer.setOutputCol("features")

# We'll hold out 20% of our data for testing and leave 80% for training
seed = 1800009193L
(split20DF, split80DF) = datasetDF.randomSplit([0.2,0.8], 1800009193L)

# Let's cache these datasets for performance
testSetDF=split20DF.cache()
trainingSetDF=split80DF.cache()

# ***** LINEAR REGRESSION MODEL ****

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml import Pipeline

# Let's initialize our linear regression learner
lr = LinearRegression()

# We use explain params to dump the parameters we can use
print(lr.explainParams())

# Now we set the parameters for the method
lr.setPredictionCol("Predicted_PE")\
    .setLabelCol("PE")\
        .setMaxIter(100)\
            .setRegParam(0.1)


# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()

lrPipeline.setStages([vectorizer, lr])

# Let's first train on the entire dataset to see what we get
lrModel = lrPipeline.fit(trainingSetDF)

# The intercept is as follows:
intercept = lrModel.stages[1].intercept

# The coefficents (i.e., weights) are as follows:
weights = lrModel.stages[1].coefficients

# Create a list of the column names (without PE)
featuresNoLabel = [col for col in datasetDF.columns if col != "PE"]

# Merge the weights and labels
coefficents = zip(weights, featuresNoLabel)

# Now let's sort the coefficients from greatest absolute weight most to the least absolute weight
coefficents.sort(key=lambda tup: abs(tup[0]), reverse=True)

equation = "y = {intercept}".format(intercept=intercept)
variables = []
for x in coefficents:
    weight = abs(x[0])
    name = x[1]
    symbol = "+" if (x[0] > 0) else "-"
    equation += (" {} ({} * {})".format(symbol, weight, name))

# Finally here is our equation
print("Linear Regression Equation: " + equation)

# Apply our LR model to the test data and predict power output
predictionsAndLabelsDF = lrModel.transform(testSetDF).select("AT", "V", "AP", "RH", "PE", "Predicted_PE")

display(predictionsAndLabelsDF)

# Now let's compute an evaluation metric for our test dataset
from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Predicted_PE", labelCol="PE", metricName="rmse")

# Run the evaluator on the DataFrame
rmse = regEval.evaluate(predictionsAndLabelsDF)

print("Root Mean Squared Error: %.2f" % rmse)

# Now let's compute another evaluation metric for our test dataset
r2 = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print("r2: {0:.2f}".format(r2))

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=lrPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our regularization parameter from 0.01 to 0.10
regParam = [x / 100.0 for x in range(1, 11)]

# We'll create a paramter grid using the ParamGridBuilder, and add the grid to the CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, regParam)
             .build())
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
cvModel = crossval.fit(trainingSetDF).bestModel

# TODO: Replace <FILL_IN> with the appropriate code.
# Now let's use cvModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = cvModel.transform(testSetDF)

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseNew = regEval.evaluate(predictionsAndLabelsDF)

# Now let's compute the r2 evaluation metric for our test dataset
r2New = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print("Original Root Mean Squared Error: {0:2.2f}".format(rmse))
print("New Root Mean Squared Error: {0:2.2f}".format(rmseNew))
print("Old r2: {0:2.2f}".format(r2))
print("New r2: {0:2.2f}".format(r2New))

print("Regularization parameter of the best model: {0:.2f}".format(cvModel.stages[-1]._java_obj.parent().getRegParam()))

from pyspark.ml.regression import DecisionTreeRegressor

# Create a DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.setLabelCol("PE")\
    .setPredictionCol("Predicted_PE")\
        .setFeaturesCol("features")\
            .setMaxBins(100)

# Create a Pipeline
dtPipeline = Pipeline()

# Set the stages of the Pipeline
dtPipeline.setStages([vectorizer,dt])

# Let's just reuse our CrossValidator with the new dtPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(dtPipeline)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
#dt = DecisionTreeRegressor(maxDepth = 2)
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth,(2,3))
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
dtModel = crossval.fit(trainingSetDF).bestModel


# Now let's use dtModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = dtModel.transform(testSetDF)
# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseDT = regEval.evaluate(predictionsAndLabelsDF)

# Now let's compute the r2 evaluation metric for our test dataset
r2DT = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})
print("LR Root Mean Squared Error: {0:.2f}".format(rmse))
print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("LR r2: {0:.2f}".format(r2))
print("DT r2: {0:.2f}".format(r2DT))

from pyspark.ml.regression import RandomForestRegressor

# Create a RandomForestRegressor
rf = RandomForestRegressor()

rf.setLabelCol("PE")\
    .setPredictionCol("Predicted_PE")\
        .setFeaturesCol("features")\
            .setSeed(100088121L)\
                .setMaxDepth(8)\
                    .setNumTrees(30)

# Create a Pipeline
rfPipeline = Pipeline()

# Set the stages of the Pipeline
rfPipeline.setStages([vectorizer,rf])

# Let's just reuse our CrossValidator with the new rfPipeline,  RegressionEvaluator regEval, and 3 fold cross validation
crossval.setEstimator(rfPipeline)

# Let's tune over our rf.maxBins parameter on the values 50 and 100, create a parameter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxBins,(50,100))
             .build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
rfModel = crossval.fit(trainingSetDF).bestModel


# Now let's use rfModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = rfModel.transform(testSetDF)

# Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
rmseRF = regEval.evaluate(predictionsAndLabelsDF)
# Now let's compute the r2 evaluation metric for our test dataset
r2RF = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})

print("LR Root Mean Squared Error: {0:.2f}".format(rmseNew))
print("DT Root Mean Squared Error: {0:.2f}".format(rmseDT))
print("RF Root Mean Squared Error: {0:.2f}".format(rmseRF))
print("LR r2: {0:.2f}".format(r2New))
print("DT r2: {0:.2f}".format(r2DT))
print("RF r2: {0:.2f}".format(r2RF))

