
# coding: utf-8

# In[1]:

# LSTM for sequence classification in the IMDB dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.callbacks import TensorBoard


# In[2]:

import pandas as pd
# read file into pandas using a relative path
directory = ('C:/Labeled data/')
path = directory+'Bank_Labeled_data1.csv'
data = pd.read_table(path, skiprows=1, header=None, sep=',', names=['text', 'sentiment'])
#citation = pd.read_table(path, sep=',', header='infer')


# In[3]:

print(data['text'])


# In[4]:

#data = data[data.sentiment != "neutral"]
data.text=data.text.astype(str)
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-Z0-9 @ . , : - _ ! () % *]','',x)))
data['text'] = data['text'].apply(lambda word: word if word not in stop_words else '')
print(data['text'])


# In[5]:

print(data[ data['sentiment'] == 'p'].size)
print(data[ data['sentiment'] == 'n'].size)
print(data[ data['sentiment'] == 'o'].size)


# In[6]:

#for idx,row in data.iterrows():
    #row[0] = row[0].replace('rt',' ')


# In[30]:

max_features = 20000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X)


# In[31]:

embed_dim = 2000
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[32]:

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:

batch_size = 2000
model.fit(X_train, Y_train, epochs = 3, batch_size=batch_size, verbose = 1)


# In[ ]:

from sklearn.metrics import classification_report, confusion_matrix
#validation_size =10

#X_validate = X_test[-validation_size:]
#Y_validate = Y_test[-validation_size:]
#X_test = X_test[:-validation_size]
#Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_train, Y_train, verbose = 2, batch_size = batch_size)
Y_pred=model.predict(X_test)
y_pred=np.argmax(Y_pred, axis=1)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
#print(Y_pred)
#print(y_pred)


# In[ ]:

target_names=['o', 'n', 'p']
print(classification_report(np.argmax(Y_test,axis=1), y_pred))
print(confusion_matrix(np.argmax(Y_test,axis=1),y_pred))


# In[ ]:

target_names=['o', 'n', 'p']
print(classification_report(np.argmax(Y_test,axis=1), y_pred))
print(confusion_matrix(np.argmax(Y_test,axis=1),y_pred))


# In[ ]:



