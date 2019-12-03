#!/usr/bin/env python
# coding: utf-8

# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import re

#load in data from twitter and instagram
data = pd.read_csv('~/Downloads/shooting_insta.csv',nrows=2000)
data2 = pd.read_csv('~/Downloads/shooting_twitter 2.csv', nrows=1000)
data_list = data.values.tolist()
data2_list = data2.values.tolist()
documents = []

#exclude words from data that are irrelevant and add data to documents
exclusion = ["www","school","https","1195008411504537600", "372ez17", "twitter", "50422956", "html", "tt", "santa", "clarita", "shooting"]
for row in data_list:
    str = row[3].lower()
    for n in exclusion:
        str = str.replace(n, " ")
    documents.append(str)
for row in data2_list:
    str = row[3].lower()
    for n in exclusion:
        str = str.replace(n, " ")
    documents.append(str)

#show the total amount of captions/tweets being processed
print("Counted %d documents" % len(documents))

#TFIDF to analyze words (and their frequency) in the data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

##have to choose number of pca and tsne
pca_num_components = 2

#K-Means clustering
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

#print cluster terms
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# In[68]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


#visualization 
num_clusters = 10
num_seeds = 10
max_iterations = 300
labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}
colors = ['#20b2aa', '#ff7373', '#ffe4e1', '#005073', '#4d0404', '#ccc0ba', '#4700f9', '#f6f900', '#00f91d', '#da8c49']
labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

# PCA
Y = X.todense()
reduced_data = PCA(n_components=pca_num_components).fit_transform(Y)
centers = np.array(order_centroids)
reduced_centers = PCA(n_components=pca_num_components).fit_transform(centers)

labels = model.fit_predict(reduced_data)

#plot the found clusters
fig, ax = plt.subplots()
legend_check = []
for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    if labels_map[labels[index]] in legend_check:
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    else:
        ax.scatter(pca_comp_1, pca_comp_2, c=color, label = labels_map[labels[index]])
        legend_check.append(labels_map[labels[index]])

leg = ax.legend(ncol=2)


# In[ ]:





# In[ ]:





# In[ ]:




