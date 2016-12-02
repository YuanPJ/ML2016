from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

import sys
from time import time

import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plot_tsne = False
plot_mds = False
kmeans = False
doc = True
verbose = False

plot_data = 5000
cos_min = 0.9
n_components = 20
max_df = 0.4
min_df = 2
n_clusters = 100
max_features = None
n_init = 40
a = str(n_components)
b = str(max_df)
c = str(n_clusters)

mystopwords = ['use', 'file', 'problem', 'line', 'way'
               'function', 'error', 'list', 'type', 'best']
stopwordslist = stopwords.words('english')
stopwordslist.extend(mystopwords)

t0 = time()
data = []
with open(sys.argv[1]+'/title_StackOverflow.txt', 'r') as txtfile :
    stemmer = PorterStemmer()
    for line in txtfile :
        lowers = line.lower()
        no_punctuation = lowers.translate(str.maketrans({key: ' ' for key in string.punctuation}))
        tokens = nltk.word_tokenize(no_punctuation)
        stemmed = []
        for w in tokens :
            stemmed.append(stemmer.stem(w))
        filtered = [w for w in stemmed if not w in stopwordslist]
        tmp = ''
        for w in filtered :
            tmp = tmp + ' ' + w
        data.append(tmp)
    txtfile.close()

if doc :
    corpus = []
    # Read Word Vector File 
    with open (sys.argv[1] + '/docs.txt', 'r') as txtfile :
        stemmer = PorterStemmer()
        for line in txtfile :
            lowers = line.lower()
            no_punctuation = lowers.translate(str.maketrans({key: ' ' for key in string.punctuation}))
            tokens = nltk.word_tokenize(no_punctuation)
            stemmed = []
            for w in tokens :
                stemmed.append(stemmer.stem(w))
            filtered = [w for w in stemmed if not w in stopwordslist]
            tmp = ''
            for w in filtered :
                tmp = tmp + ' ' + w
            if tmp != '' :
                corpus.append(tmp)
        txtfile.close()

with open(sys.argv[1]+'/check_index.csv', 'r') as csvfile :
    reader = csv.reader(csvfile)
    for line in reader :
        testdata = list(reader)
    csvfile.close()
testdata = np.array(testdata, dtype=int)


if doc :
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                 max_features=max_features, stop_words='english')
    train = corpus + data
    vectorizer = vectorizer.fit(train)
else :
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                 max_features=max_features, stop_words='english')
    vectorizer = vectorizer.fir(data)
X = vectorizer.transform(data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if True:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    print(svd.explained_variance_ratio_)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

if plot_tsne :
    tsne = TSNE()
    pltX = tsne.fit_transform(X[:plot_data])
    plt.figure(figsize = (10,10))
    fig = plt.scatter(pltX[:,0], pltX[:,1]).get_figure()
    #fig.savefig('1130/' + sys.argv[3] + '_tsne.png')
    print("Use TSNE to plot.")
    print()

if plot_mds :
    mds = MDS()
    pltX = mds.fit_transform(X[:plot_data])
    plt.figure(figsize = (10,10))
    fig = plt.scatter(pltX[:,0], pltX[:,1]).get_figure()
    #fig.savefig('1130/' + sys.argv[3] + '_mds.png')
    print("Use MDS to plot.")
    print()

###############################################################################
# Do the actual clustering

if kmeans:
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=n_init,
                verbose=verbose)
    km.fit(X)

    print("Top terms per cluster:")
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    tag = []
    for i in range(n_clusters) :
        tag.append(terms[order_centroids[i, 0]])
    tags = np.sort(np.array(tag))
    print(tags)

    labels = np.array(km.labels_)
    ans = np.equal(labels[testdata[:, 1]], labels[testdata[:, 2]])
    zero = np.zeros(np.size(ans), dtype=int)
    myans = ans + zero

    color = cm.rainbow(np.linspace(0, 1, n_clusters))
    colors = []
    for i in range(len(labels)) :
        colors.append(color[labels[i]])

    #plt.figure(figsize = (10,10))
    #fig = plt.scatter(X[:,1], X[:,2], c=colors).get_figure()
    #fig.savefig('1130/' + sys.argv[3] + '.png')
else :
    cos = cosine_similarity(X)
    ans = np.greater(cos[testdata[:, 1], testdata[:, 2]], cos_min)
    zero = np.zeros(np.size(ans), dtype=int)
    myans = ans +zero

with open(sys.argv[2], 'w') as csvfile :
    fieldnames = ['ID', 'Ans']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for id in range(np.size(myans)) :
        writer.writerow({'ID' : id, 'Ans' : myans[id]})

