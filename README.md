# Quovo - Data Science startup based in NY.
# Data Challenge - Data Scientist Position.

Quovo - Quovo is a provider of account aggregation and data analytics technology for finance.

## Total Equity Funding

$15.2M in 3 Rounds from 8 Investors

# Challenge details:

Thank you for taking the time to apply to Quovo. We like to send potential candidates a SHORT coding test/exercise so 
we could get a sense of how they approach problems. This also gives you the a good opportunity to see if Quovo-style 
challenges are a good fit for you. Don't go crazy on time, we'd just like to see enough progress on it where we can 
all have a conversation looking at your code together and talk about how you attacked the problem.

--

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

# My Approach

Calculated Unigram, Bigram,and Trigram. Used XGboost and log-loss to predict the probability instead of 0 or 1 prediction. You can take a look at the Quovo-submiss.ipynb for the code and issimilar_predicted for the output values. Also, you can use this link http://nbviewer.jupyter.org/github/RishiSankineni/Quovo-Data-Challenge/blob/master/Quovo-Submiss.ipynb to share the file. Thanks - Rishi

log loss- 0.56 (not bad). Nevertheless, will try to improve it by approaching the problem in a different way.
