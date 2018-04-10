
# coding: utf-8

# In[55]:

import time
import math
import re
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import string

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim import corpora, models


# In[2]:

import pickle as pkl
directory = 'C:/UOC/Class/NLPCognitive/Project/'
#file = 'jeep.txt'
file = 'news_chicago_il.pkl'
path = directory + file


# In[3]:

#print(path)
news_df = pd.read_pickle(path)


# In[6]:

news_df.head(5)


# In[4]:

# Filter non-English tweets
news_eng = news_df[news_df['language']=='english'].reset_index(drop=True)


# In[5]:

news_eng.head(5)


# # Clean up the noise - Keep articles related to population

# In[18]:

news_pop=news_eng[news_eng['text'].str.contains("population")]


# In[21]:

news_pop.shape[0]


# In[23]:

news_pop.head(5)


# In[24]:

# Remove special characters to avoid problems with analysis
news_pop['title_clean'] = news_pop['title'].map(lambda x: re.sub('[^a-zA-Z0-9 @ . , : - _]', '', str(x)))


# In[42]:

pd.set_option('display.max_colwidth', 100)
news_pop[['text', 'title_clean']].head(20)


# In[26]:

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)
# tf(word, blob) computes "term frequency" which is the number of times a word appears in a document blob, 
# normalized by dividing by the total number of words in blob. We use TextBlob for breaking up the text into words 
# and getting the word counts.


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)
# n_containing(word, bloblist) returns the number of documents containing word. 
# A generator expression is passed to the sum() function.


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
# idf(word, bloblist) computes "inverse document frequency" which measures how common a word is 
# among all documents in bloblist. The more common a word is, the lower its idf. 
# We take the ratio of the total number of documents to the number of documents containing word, 
# then take the log of that. Add 1 to the divisor to prevent division by zero


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
# tfidf(word, blob, bloblist) computes the TF-IDF score. It is simply the product of tf and idf.


# In[27]:

bloblist = []
del bloblist[:]

for i  in range(0,len(news_pop)):
    bloblist.append(TextBlob(news_pop['title_clean'].iloc[i]))
    
len(bloblist) 


# In[40]:

for i, blob in enumerate(bloblist):
# Print top 5 values
    if i == 20:
        break
    print("Top words in news article {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 10)))


# # Applying LDA 

# In[29]:

news_list = news_pop['title_clean'].tolist()
#news_list[:1]


# In[13]:

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# In[30]:

news_clean = [clean(doc).split() for doc in news_list]


# In[31]:

len(news_clean)


# In[32]:

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 

dictionary = corpora.Dictionary(news_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

get_ipython().magic('time doc_term_matrix = [dictionary.doc2bow(doc) for doc in news_clean]')


# # 20 topic model

# In[72]:

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
numtopics = 20

# Running and Trainign LDA model on the document term matrix.
get_ipython().magic('time ldamodel = Lda(doc_term_matrix, num_topics=numtopics, id2word = dictionary, passes=200)')


# In[75]:

print(*ldamodel.print_topics(num_topics=numtopics, num_words=20), sep='\n\n')


# # Sentiment Analysis

# In[44]:

file_pos = 'positive.txt'
file_neg = 'negative.txt'


# In[45]:

pos_sent = open(directory +file_pos).read()
pos_words = pos_sent.split('\n')
neg_sent = open(directory +file_neg).read()
neg_words = neg_sent.split('\n')


# In[46]:

news_pop['text_clean'] = news_pop['text'].map(lambda x: re.sub('[^a-zA-Z0-9 @ . , : - _]', '', str(x)))


# In[61]:

news_list = news_pop['text_clean']


# In[60]:

news_pop['text_clean'].head(5)


# In[62]:

# customize the dictionaries by adding and removing your own positive and negative words and get some counts

pos_add = ['your_pos_term_1, your_pos_term_2']

for term in pos_add:
    pos_words.append(term)

neg_add = ['your_neg_term_1, your_neg_term_2']

for term in neg_add:
    neg_words.append(term)

import re
from string import punctuation
from __future__ import division  
sentiment_scores=[]
for news in news_list:
    sentiment_score=0
    for p in list(punctuation):
        news=news.replace(p,'')
        words=news.split(' ')
    for word in words:
        if word in pos_words:
            sentiment_score=sentiment_score+1
        if word in neg_words:
            sentiment_score=sentiment_score-1
    sentiment_scores.append(sentiment_score/len(words))

news_sentiment=zip(news_list,sentiment_scores)


# In[63]:

# Create a dataframe from the results
column_names = ["Text", "Sentiment_Score"]
sentiment_results = [news_list, sentiment_scores]
results_dict = dict(zip(column_names,sentiment_results))
all_news_df = pd.DataFrame.from_dict(results_dict, orient='columns')
all_news_df = all_news_df[column_names]   # set specific column order


# In[64]:

# Create a list to store the sentiments
sent_list = []

# For each row in the column,
for row in all_news_df['Sentiment_Score']:
    if row > 0:
        sent_list.append('Positive')
    elif row < 0:
        sent_list.append('Negative')
    else:
        sent_list.append('Neutral')

# Create a column from the list
all_news_df['Sentiment_Label'] = sent_list


# In[65]:

#Make sure I didn't loose any records
len(news_list) - len(all_news_df)


# In[66]:

pd.set_option('display.max_colwidth', 150)


# In[67]:

all_news_df.sample(frac=0.005, replace=True)


# In[68]:

plt.figure().set_size_inches(10, 5)

CountSentiment = pd.value_counts(all_news_df['Sentiment_Label'].values, sort=True)
print (CountSentiment)

#CountStatus.plot.barh()
CountSentiment.plot.bar()
plt.show()


# In[69]:

writer = pd.ExcelWriter(directory+'news_sentiment.xlsx', engine='xlsxwriter')
all_news_df.to_excel(writer, sheet_name='news_Sentiment')
writer.save()


# # POS Tagging - Identify Organization and People

# In[100]:

file='NER.txt'


# In[101]:

text = open(directory +file).read()


# In[109]:

entities = []
labels = []
for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)), binary = False):
    if hasattr(chunk, 'label'):
        entities.append(' '.join(c[0] for c in chunk)) #Add space as between multi-token entities
        labels.append(chunk.label())

#entities_labels = list(zip(entities, labels))
entities_labels = list(set(zip(entities, labels))) #unique entities


# In[115]:

entities_all = list(zip(entities, labels))


# In[110]:

entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities", "Labels"]
entities_df.head(20)


# In[111]:

entities_df.groupby('Labels').count()


# In[118]:

entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities", "Labels"]
persons_df = entities_df.loc[entities_df["Labels"].isin(['ORGANIZATION','PERSON'])]
counts_df = persons_df.groupby('Entities').count()
counts_df.rename(columns={"Labels": "Mentions"}, inplace=True)
counts_df.sort_values(by=['Mentions'], ascending=False).head(20)


# In[ ]:



