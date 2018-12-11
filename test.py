import pickle
from sklearn
import pandas as pd
import numpy as np


#finding all the classes
df_train = pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
print(df_train.head())
train_feature = np.array(df_train["features"].tolist())
train_label = np.array(df_train["labels"].tolist())
p = train_label
p = list(set(train_label))
p.sort()

#loading the models
models = []
for i in range(0,40):
    filename = "./mfcc_delta_delta/without_energy_coeff/model_of_"+p[i]+".pkl"
    f = open(filename, 'rb')
    m = pickle.load(f)         
    models.append(m)
    f.close() 
    print("Generated model for "+p[i])


df_test = pd.read_hdf("./features/mfcc_delta_delta/Test.hdf")
test_feature = np.array(df_test["features"].tolist())
test_label = np.array(df_test["labels"].tolist())

test_feature=np.delete(test_feature, [0,13,26], axis=1) 


#finding scores for every class
scores = np.empty([test_labels.shape[0],0])
for i in range(0,40):
    tmp = models[i].score_samples(test_features)
    scores = np.concatenate((scores,tmp.reshape(test_labels.shape[0],1)),axis=1)


#finding maximum score class
pred_labels = np.argmax(scores,axis=1)
pred_labels2 = []
for i in range(0,pred_labels.shape[0]):
    pred_labels2.append(p[pred_labels[i]])


#calculating accuracy score
sklearn.metrics.accuracy_score(test_labels,pred_labels2)
    



