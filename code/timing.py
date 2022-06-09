from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from metric_learn import LMNN
from mlxtend import evaluate
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
import csv, time
from joblib import Parallel, delayed
from sklvq import GMLVQ
from sklearn.naive_bayes import GaussianNB

tic = time.perf_counter()

# SEEDING _______________________________________-

data = pd.read_csv("data/Spam Dataset/spambase.csv")
balance = [40,60]
samples = 4599
times_cv = []
times_bs = []

for k in range(10):
    data = shuffle(data)
    firstClass = 0
    secondClass = 0
    new_data = []
    firstClassFlag = 1
    secondClassFlag = 1
    firstClassCount = 0
    secondClassCount = 0
    i = 0
    while len(new_data) != samples:
        if secondClassFlag == 1 and data['target'][i] == 1:
            new_data.append(data.iloc[i])
            secondClassCount += 1
            if secondClassCount == (samples/100)*balance[0]: secondClassFlag = 0
        if firstClassFlag == 1 and data['target'][i] == 0:
            new_data.append(data.iloc[i])
            firstClassCount += 1
            if firstClassCount == (samples/100)*balance[1]: firstClassFlag = 0
        i += 1

    new_df = pd.DataFrame(new_data)

    y = new_df['target']
    y = np.array(y)
    x = new_df.drop('target', axis = 1)
    x = np.array(x)

    # print(x)

    # SVM_Model = svm.SVC(kernel = 'linear', C=1, random_state=128)
    # LogReg_Model = LogisticRegression(random_state=0, max_iter = 5000)
    # lmnn = LMNN(k=5, learn_rate=1e-6) # 4600: 170 secs per sim
    # GMLVQ_Model = GMLVQ(random_state=128)
    NaiveBayes_Model = GaussianNB()

    print("Model Created")

    # model_selection.cross_val_score(SVM_Model, x, y, cv=10)

    # print("Cross Validation Done")

    cv_start = time.perf_counter()
    
    for i in range(0,3):
        model_selection.cross_val_score(NaiveBayes_Model, x, y, cv=10)

    cv_end = time.perf_counter()

    print(f"CV took {cv_end - cv_start:0.4f} seconds")
    times_cv.append(cv_end - cv_start)

    # print("Repeated Cross Validation Done")

    # evaluate.bootstrap_point632_score(SVM_Model, x, y, n_splits = 50)

    # print("632 Bootstrap Done")

    bs_start = time.perf_counter()

    evaluate.bootstrap_point632_score(NaiveBayes_Model, x, y, n_splits = 30, method='.632+')

    bs_end = time.perf_counter()

    print(f"BS took {bs_end - bs_start:0.4f} seconds")
    times_bs.append(bs_end - bs_start)

    # print("632+ Bootstrap Done")

toc = time.perf_counter()

print(f"Simulation took {toc - tic:0.4f} seconds")

print(f"CV on average took {sum(times_cv)/10} seconds")
print(f"BS on average took {sum(times_bs)/10} seconds")