from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
data = pd.read_csv('~/Downloads/shooting_insta.csv',nrows=2000)
data2 = pd.read_csv('~/Downloads/shooting_twitter 2.csv', nrows=1000)
data_list = data.values.tolist()
data2_list = data2.values.tolist()
documents = []
exclusion = ["www","school","https", "327ez17","1195008411504537600", "twitter", "50422956"]
for row in data_list:
    str = row[3]
    for n in exclusion:
        str = str.replace(n, " ")
    documents.append(str)
for row in data2_list:
    str = row[3]
    for n in exclusion:
        str = str.replace(n, " ")
    documents.append(str)

print("Counted %d documents" % len(documents))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
print(order_centroids)
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
    
##visualization

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

num_clusters = 10
num_seeds = 10
max_iterations = 300
labels_color_map = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}

##have to choose number of pca and tsne
pca_num_components = 2
tsne_num_components = 2

centers = np.array(model.cluster_centers_)
print(centers)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

labels = model.fit_predict(X)
Y = X.todense()
reduced_data = PCA(n_components=pca_num_components).fit_transform(Y)
fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
plt.show()


embeddings = TSNE(n_components=tsne_num_components)
Y = embeddings.fit_transform(Y)
plt.scatter(Z[:, 0], Z[:, 1], cmap=plt.cm.Spectral)
plt.show()
