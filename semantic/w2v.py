#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

import json
from collections import Counter
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from utils import MetricCallback, plot_vectors

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load and display data

with open('data/countries.json', 'r') as fout:
    countries = json.load(fout)


# In[3]:


countries['India'][:20]


# In[4]:


print(' '.join(countries['India'])[:512] + ' ...')


# ## Basic Word2Vec Usage

# In[5]:


# Create and train a simple model

model = Word2Vec(sentences=countries.values())


# In[6]:


# Check word similarities learnt by the model

model.wv.most_similar('India', topn=5)


# In[7]:


# Enable computation of loss

model = Word2Vec(
    sentences=countries.values(),
    compute_loss=True
)
model.get_latest_training_loss()


# ## Heuristics for Word2vec algorithms

# ### Determining size of the vocabulary

# In[8]:


# How many unique words in the vocabulary?

counter = Counter()
for words in countries.values():
    for word in words:
        counter.update([word])

print(len(counter))


# In[9]:


# Default vocabulary size of the original model

len(model.wv)


# In[10]:


# Retrain - increased vocabulary size, more epochs, larger word vectors

metric = MetricCallback(every=1)
model = Word2Vec(
    sentences=countries.values(),
    vector_size=128,
    epochs=10,
    max_vocab_size=65536,
    compute_loss=True,
    callbacks=[metric]
)
plt.plot(metric.myloss)


# In[11]:


# Check similarities again

model.wv.most_similar('India')


# In[12]:


# Retrain - more epochs

metric = MetricCallback(every=10)
model = Word2Vec(
    sentences=countries.values(),
    vector_size=128,
    epochs=100,
    max_vocab_size=65536,
    compute_loss=True,
    callbacks=[metric],
    min_alpha=0.001,
    workers=9
)
plt.plot(metric.myloss)


# In[13]:


model.wv.most_similar('India')


# In[14]:


# Examine the vector space

X = ['India', 'Pakistan', 'Bangladesh', 'France', 'England', 'Spain']
Y = ['Delhi', 'Islamabad', 'Dhaka', 'Paris', 'London', 'Madrid']
plot_vectors(X, Y, model.wv)


# In[15]:


# Visualize vectors for all countries

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


names = []
vectors = []
for country in countries:
    if country in model.wv:
        names.append(country)
        vectors.append(model.wv[country])

X = np.r_[vectors]
x_red = TSNE(n_components=2).fit_transform(X)
fig, ax = plt.subplots(figsize=(16, 16))
ax.scatter(*x_red.T)

for i, word in enumerate(names):
        plt.annotate(word, x_red[i])


# ## Word Analogies

# In[16]:


# India: Ganges -> Brazil: __ ?

model.wv.most_similar(positive=['Ganges', 'Brazil'], negative=['India'])


# In[17]:


# America: Washington -> France: __ ?

model.wv.most_similar(positive=['Washington', 'France'], negative=['America'])


# In[18]:


# India: Hindi -> Germany: __ ?

model.wv.most_similar(positive=['Hindi', 'Germany'], negative=['India'])


# In[19]:


# Save the model

model.save('wiki-countries.w2v')

