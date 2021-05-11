#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import Word2Vec


# In[2]:


model = Word2Vec.load('data/imdb-word2vec.w2v')


# In[3]:


from tensorflow.keras.preprocessing import text_dataset_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split


# In[4]:


# Importing & preprocessing the dataset

train_ds = text_dataset_from_directory('../neuralnets/aclImdb/train')
test_ds = text_dataset_from_directory('../neuralnets/aclImdb/test')

dfTrain = pd.DataFrame(train_ds.unbatch().as_numpy_iterator(), columns=['text', 'label'])
dfTest = pd.DataFrame(test_ds.unbatch().as_numpy_iterator(), columns=['text', 'label'])
_, xts = train_test_split(dfTest, stratify=dfTest['label'], test_size=0.25)


# In[5]:


len(model.wv)


# In[6]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[7]:


dfTrain['text'] = dfTrain['text'].map(lambda x: x.decode())
xts['text'] = xts['text'].map(lambda x: x.decode())


# In[8]:


tokenizer = Tokenizer(num_words=len(model.wv))
tokenizer.fit_on_texts(dfTrain['text'].tolist())


# In[9]:


train_sequences = tokenizer.texts_to_sequences(dfTrain['text'])


# In[10]:


test_sequences = tokenizer.texts_to_sequences(xts['text'])


# In[11]:


word_index = tokenizer.word_index


# In[12]:


word_index


# In[13]:


len(word_index)


# In[14]:


MAX_SEQUENCE_LENGTH = max(map(len, train_sequences))


# In[15]:


MAX_SEQUENCE_LENGTH


# In[16]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[17]:


train_labels = dfTrain['label'].values
test_labels = xts['label'].values


# In[40]:


# Prepare the embedding layer


# In[18]:


import numpy as np


# In[19]:


embedding_matrix = np.zeros((len(word_index) + 1, model.wv.vector_size))


# In[20]:


for word, i in word_index.items():
    try:
        vector = model.wv.get_vector(word, False)
        embedding_matrix[i] = vector
    except KeyError:
        continue


# In[21]:


(embedding_matrix.sum(axis=1) == 0).sum()


# In[22]:


from tensorflow.keras.layers import Embedding


# In[23]:


el = Embedding(len(word_index) + 1, model.wv.vector_size, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)


# In[24]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


# In[25]:


clf = Sequential([
    Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
    el,
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[31]:


from keras.optimizers import SGD, Adam


# In[32]:


clf.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[33]:


clf.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=10, batch_size=128)


# In[78]:


# With Glove


# In[34]:


from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = el(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# In[36]:


clf.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=100, batch_size=128)


# In[ ]:




