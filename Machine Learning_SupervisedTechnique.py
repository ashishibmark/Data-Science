
# coding: utf-8

# In[1]:

import sklearn

import pandas as pd

import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import gensim


# In[2]:

# read file into pandas using a relative path
directory = ('C:/UOC/Labeled data/')
path = directory+'Labeled_data1.csv'
data = pd.read_table(path, skiprows=1, header=None, sep=',', names=['comment', 'sentiment'])
#citation = pd.read_table(path, sep=',', header='infer')


# In[3]:

# examine the shape
data.shape


# In[4]:

data.head(5)


# In[5]:

# examine the class distribution
data.comment.value_counts().head(10)


# In[6]:

# convert label to a binary numerical variable
data['sentiment_flag'] = data.sentiment.map({'p':'1', 'n':'2', 'o':'0'})


# In[7]:

# check that the conversion worked
data.head(10)


# In[8]:

# Remove special characters to avoid problems with analysis
data['comment_clean'] = data['comment'].map(lambda x: re.sub('[^a-zA-Z0-9 @ . , : - _ ! () % * ]', '',str(x)))


# In[9]:

#data.comment.head(5)
data[['comment', 'comment_clean']].head(10)


# In[10]:

# how to define X and y (from the citation data) for use with COUNTVECTORIZER
X = data.comment_clean
y = data.sentiment
print(X.shape)
print(y.shape)


# In[11]:

#tokens=nltk.tokenize.word_tokenize(X)


# In[12]:

#stopwords = stopwords.words('english')
#stopwords = set(nltk.corpus.stopwords.words('english'))
# Remove stopwords
#words = [word for word in X if word not in stopwords]


# In[13]:

# Remove single-character tokens (mostly punctuation)
#words = [word for word in X if len(word) > 1]
#text=words.to_string()


# In[14]:

# Remove numbers
#words = [word for word in words if not word.isnumeric()]


# In[15]:

# Remove punctuation
#X = [word for word in words if word.isalpha()]


# In[16]:

#Lemmatizer
#wnl = nltk.WordNetLemmatizer()
#words= [wnl.lemmatize(t) for t in words]


# In[17]:

# remove English stop words
vect = CountVectorizer(stop_words='english',ngram_range=(1,),max_df=0.5, min_df=2)


# In[18]:

# show default parameters for CountVectorizer
print(vect)


# In[19]:

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:

# instantiate the vectorizer
vect = CountVectorizer()


# In[21]:

# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[22]:

# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)


# In[23]:

# examine the document-term matrix
X_train_dtm


# In[24]:

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# # Naive Bayes

# In[25]:

# instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()


# In[26]:

# train and time the model using X_train_dtm
get_ipython().magic('time nb.fit(X_train_dtm, y_train)')


# In[27]:

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[28]:

# calculate accuracy of class predictions
print(metrics.accuracy_score(y_test, y_pred_class))


# In[29]:

# calculate precision and recall
print(classification_report(y_test, y_pred_class))


# In[30]:

# calculate the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))


# # Logistic Regression Model

# In[31]:

# instantiate a logistic regression model
logreg = LogisticRegression()


# In[32]:

get_ipython().magic('time logreg.fit(X_train_dtm, y_train)')


# In[33]:

# make class predictions for X_test_dtm - Test
y_pred_class = logreg.predict(X_test_dtm)


# In[34]:

# calculate accuracy of class predictions - Test
print(metrics.accuracy_score(y_test, y_pred_class))


# In[35]:

# calculate precision and recall - Test
print(classification_report(y_test, y_pred_class))


# In[36]:

# calculate the confusion matrix - Test
print(metrics.confusion_matrix(y_test, y_pred_class))


# # Support Vector Machine

# In[37]:

# instantiate a SVM model
svm = SGDClassifier()


# In[38]:

# train the model using X_train_dtm
get_ipython().magic('time svm.fit(X_train_dtm, y_train)')


# In[39]:

# make class predictions for X_test_dtm
y_pred_class = svm.predict(X_test_dtm)


# In[90]:

# calculate accuracy of class predictions - Test
print(metrics.accuracy_score(y_test, y_pred_class))


# In[41]:

# calculate precision and recall - Test
print(classification_report(y_test, y_pred_class))


# In[42]:

# calculate the confusion matrix - Test
print(metrics.confusion_matrix(y_test, y_pred_class))


# # Sentiment from the model

# In[159]:

directory = ('C:/UOC/Labeled data/')
path = directory+'Bank_2017.csv'


# In[160]:

data = pd.read_table(path, skiprows=1, header=None, sep=',', names=['comment'])


# In[161]:

data.shape
#data.head(5)


# In[162]:

data.head(5)


# In[163]:

# Remove special characters to avoid problems with analysis
data['comment'] = data['comment'].map(lambda x: re.sub('[^a-zA-Z0-9 @ . , : - _ ! () % * ]', '',str(x)))


# In[164]:

X_Actual=data.comment


# In[165]:

X_Actual.shape


# In[166]:

# transform testing data (using fitted vocabulary) into a document-term matrix
X_Actual_dtm = vect.transform(X_Actual)
X_Actual_dtm


# In[167]:

Y_Actual_class = logreg.predict(X_Actual_dtm)


# In[168]:

#print(metrics.accuracy_score(Y_Actual_class, Y_Actual))


# In[169]:

#print(metrics.confusion_matrix(Y_Actual_class, Y_Actual))


# In[170]:

Y_Actual_class.shape


# In[172]:

### Create a dataframe from the results
###column_names = ["SNo","Comment", "Sentiment"]
###sentiment_results = [tweet_list, sentiment_scores]
results_dict = list(zip(X_Actual, Y_Actual_class))
all_tweets_df = pd.DataFrame.from_dict(results_dict, orient='columns')
###all_tweets_df = all_tweets_df[column_names]   # set specific column order


# In[173]:

writer = pd.ExcelWriter(directory+'Bank_2017_sent.xlsx', engine='xlsxwriter')
all_tweets_df.to_excel(writer, sheet_name='Sentiment')
writer.save()


# In[ ]:



