import numpy as np
import pandas as pd


#+++++++++++++++++++++++++++ Echo State NetWork (Reservoir Computing) ESN ++++++++++++++++++++++++++++++++++++++++++++++++
#https://www.quantamagazine.org/machine-learnings-amazing-ability-to-predict-chaos-20180418/

#http://strategy.doubledoji.com/how-to-use-echo-state-network-in-forex-trading/


'''
#import ESN module
from pyESN import ESN
#import matplotlib
import matplotlib.pyplot as plt

#read the open, high, low and closing price from the csv files
o, h, l, c=np.loadtxt("E:/MarketData/GBPUSD30.csv", delimiter=',', 
                      usecols=(2,3,4,5), unpack=True)

##build an Echo State Network
esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 500,
          spectral_radius = 1.5,
          random_state=42)
#choose the training set

trainlen = 500
future = 20
#start training the model
pred_training = esn.fit(np.ones(trainlen),c[len(c)-trainlen: len(c)])
#make the predictions
prediction = esn.predict(np.ones(future))
print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() \
- c[trainlen:trainlen+future])**2))))

#print the predicted values of the closing price
prediction
#plot the predictions
plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen),c[len(c)-trainlen:len(c)],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()


plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')


'''






#++++++++++++++++++++++++++++++++++++++ Self Organizing Maps (SOM) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://github.com/JRC1995/Self-Organizing-Map/blob/master/SOM.ipynb





#+++++++++++++++++++++++++++++++++++++++++ Boltzman Machines (BM) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++ Restricted Boltzman Machines (RBM) +++++++++++++++++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++++++++++++++++ Deep Belief Networks (DBN) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#+++++++++++++++++++++++++++++++++++++++++++++++++++++ K-MEANS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


print("++++++++++++++++++++++++++ Hierarchical Clustering +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# some setting for this notebook to actually show the graphs inline
# you probably won't need this
#%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
c = np.random.multivariate_normal([-2,1],[[3,1],[1,4]],size=[70,])
d = np.random.multivariate_normal([10,20],[[3,1],[1,4]],size=[50,])
#X = np.concatenate((a, b),)
#X = np.concatenate((a,b,c),)
X = np.concatenate((a,b,c,d),)

print(X.shape)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()

# generate the linkage matrix
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
print(c)

print(Z[0])
print(Z[1])
print(Z[:20])

print(X[[33, 68, 62]])

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1])  # plot all points
plt.scatter(X[idxs,0], X[idxs,1], c='r')  # plot interesting points in red again
plt.show()

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1])
plt.scatter(X[idxs,0], X[idxs,1], c='r')
idxs = [15, 69, 41]
plt.scatter(X[idxs,0], X[idxs,1], c='y')
plt.show()


# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

print(Z[-8:,2])


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()


# set cut-off to 50
max_d = 50  # max_d as in max_distance

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()


fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=16,
)
plt.show()


from scipy.cluster.hierarchy import inconsistent

depth = 5
incons = inconsistent(Z, depth)
incons[-10:]

depth = 3
incons = inconsistent(Z, depth)
incons[-10:]

last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)



c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[200,])
d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[200,])
e = np.random.multivariate_normal([0, 100], [[100, 1], [1, 100]], size=[200,])
X2 = np.concatenate((X, c, d, e),)
plt.scatter(X2[:,0], X2[:,1])
plt.show()


Z2 = linkage(X2, 'ward')
plt.figure(figsize=(10,10))
fancy_dendrogram(
    Z2,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=40,
    max_d=170,
)
plt.show()


last = Z2[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)

print(inconsistent(Z2, 5)[-10:])


from scipy.cluster.hierarchy import fcluster
max_d = 50
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

k=2
fcluster(Z, k, criterion='maxclust')

from scipy.cluster.hierarchy import fcluster
fcluster(Z, 8, depth=10)


plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()







































#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Mixture Models ++++++++++++++++++++++++++++++++++++++++++++++++++

