from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from mlxtend import evaluate
import numpy as np
import pandas as pd 
import time
from joblib import Parallel, delayed

data = pd.read_csv("data/Spam Dataset/spambase.csv")

# print(data.isnull().sum()) NO MISSING vALUES

y = data['target']
y = np.array(y)
x = data.drop('target', axis = 1)
x = np.array(x)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.15, random_state = 54)

SVM_Model = svm.SVC(kernel = 'linear', C=1, random_state=128)
# SVM_Model.fit(x_train, y_train)
# y_pred = SVM_Model.predict(x_test)
# print(y_pred, "\n", y_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
tik = time.perf_counter()
def train(x, y):
    scores = model_selection.cross_val_score(SVM_Model, x, y, cv=10)
    print(scores.mean())
# scores2 = evaluate.bootstrap_point632_score(SVM_Model, x, y)
# print(scores2.mean())
tok = time.perf_counter()
print(tok-tik)