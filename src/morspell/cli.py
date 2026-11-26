
import dataloader as dl

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

thmat, morpho_cols = dl.squirrel_mat()
print(thmat.head())
thmat.loc[thmat["lineage"] != "north", "lineage"] = "no-north"

y = thmat.loc[(thmat["sex"] == "male"), "lineage"].values
X = thmat.loc[(thmat["sex"] == "male"), morpho_cols].values

print(X.shape, y.shape)

scaler = StandardScaler()
labenc = LabelEncoder()
X_sca = scaler.fit_transform(X)


clust = SpectralClustering(
	n_clusters=4, 
	assign_labels="discretize",
	random_state=12345
	).fit(X_sca)

print(clust.labels_)

y_enc = labenc.fit_transform(y)
print([i for i in range(len(labenc.classes_))])


X_train, X_test, y_train, y_test = train_test_split(X_sca, y_enc, test_size=0.2, random_state=12345, stratify=y_enc)
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
y_pred_prob = lda.predict_proba(X_test)
ohenc = OneHotEncoder(
	sparse_output=False, 
	handle_unknown="ignore",
	categories=[[i for i in range(len(labenc.classes_))]]
)
y_test_oh = ohenc.fit_transform(y_test.reshape(-1, 1))
print(y_pred_prob.shape, y_test_oh.shape)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test_oh, y_pred_prob, multi_class="ovr")
print(acc, roc)
print(lda.coef_)
print(lda.get_params())
exit()