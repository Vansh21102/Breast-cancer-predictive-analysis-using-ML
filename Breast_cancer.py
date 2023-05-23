import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

df = pd.read_csv("breast-cancer.csv")
df.drop(["id"], axis = 1, inplace = True)

#sns.heatmap(df.corr())
#plt.show()

df["diagnosis"] = np.where(df["diagnosis"].str.contains("M"), 1, 0)

#print(np.unique(df["diagnosis"]))

dflabel = df.copy()
df.drop('diagnosis', axis = 1, inplace = True)
dflabel.drop(df[df.columns[0:31]], axis = 1, inplace = True)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df, dflabel, test_size = 0.3)

regressor.fit(X_train, Y_train)

score = regressor.score(X_test,Y_test)

print("score = ", score)

#print(df.head())
#print(dflabel.head())