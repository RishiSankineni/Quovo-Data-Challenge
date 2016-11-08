
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


