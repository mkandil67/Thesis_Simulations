from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from mlxtend import evaluate
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
import csv, time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/Spam Dataset/spambase.csv")

# Separating out the features
x = data.drop('target', axis = 1).values
# Separating out the target
y = data.loc[:,['target']].values

data = np.array(data)

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
data = pd.DataFrame(data)
print(data)
finalDf = pd.concat([principalDf, data[57]], axis = 1)
print(finalDf)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[57] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# pca = PCA(n_components=58)
# pca.fit(data)
# print(pca.explained_variance_ratio_)
# plt.bar(list(range(58)), pca.explained_variance_ratio_)
# plt.show()