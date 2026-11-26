
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

clustme = True
score_clust = "ch" # 'ch' | 'db' | 's'

thmat, morpho_cols = dl.squirrel_mat()
print(thmat.head())
thmat.loc[thmat["lineage"] != "north", "lineage"] = "no-north"

y = thmat.loc[(thmat["sex"] == "male"), "lineage"].values
X = thmat.loc[(thmat["sex"] == "male"), morpho_cols].values

print(X.shape, y.shape)

scaler = StandardScaler()
labenc = LabelEncoder()
X_sca = scaler.fit_transform(X)

if clustme:

	chscore = []
	dbscore = []
	sscore = []
	lab_hist = []

	for i in range(2, int(X.shape[0])):

		print(f"Assuming {i} clusters")
		clust = SpectralClustering(
			n_clusters=i, 
			assign_labels="discretize",
			random_state=12345
			).fit(X_sca)

		lab_hist.append(clust.labels_)
		chscore.append(calinski_harabasz_score(X_sca, clust.labels_))
		dbscore.append(davies_bouldin_score(X_sca, clust.labels_))
		sscore.append(silhouette_score(X_sca, clust.labels_))
		#print(f"calinski_harabasz_score = {chs}")
		#print(f"davies_bouldin_score = {dbs}")
		#print(f"silhouette_score = {ss}")
		#print("#" * 80)

	print(f"Best Calinski Harabasz score: {max(chscore)}")
	print(f"Best Calinski Harabasz clustering:\n{lab_hist[chscore.index(max(chscore))]}")

	print(f"Best Davies Bouldin score:{min(dbscore)}")
	print(f"Best Davies Bouldin clustering:\n{lab_hist[dbscore.index(min(dbscore))]}")

	print(f"Best Silhouette score: {max(sscore)}")
	print(f"Best Silhouette clustering:\n{lab_hist[sscore.index(max(sscore))]}")

exit()

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