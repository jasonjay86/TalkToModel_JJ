import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Loading + splitting the data
data = pd.read_csv("./austrailian.csv", index_col=None)
y = data.pop('A15')

# I noticed that having class labels on a range besides 0 ==> N introduces some bugs because of the way certain explanation packages we use handle these labels...
# I suggest adjusting them to start with 0. In general, we developed this project using binary classification tasks, so it's a bit better tested for this setting.
print(y.min(), y.max())
if y.min() != 0:
  y += -1
print(y.min(), y.max())

X_train, X_test, y_train, y_test = train_test_split(data, y)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# fitting the model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
# Random Forest
rf_pipeline = Pipeline([('scaler', StandardScaler()),
                        ('rf', RandomForestClassifier())])
rf_pipeline.fit(X_train.values, y_train.values)
print(f"RF Score: {rf_pipeline.score(X_test.values, y_test.values)}")

#XGBoost
xg = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)
xg.fit(X_train, y_train)
print(f"XGBoost Score: {xg.score(X_test, y_test)}")

#Logistic Regression
lr = LogisticRegression(random_state=0,max_iter=10000)
lr.fit(X_train, y_train)
print(f"Logistic Regression Score: {lr.score(X_test, y_test)}")

#MLP Classifier (Neural Net)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,max_iter=10000,
                    hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train)
print(f"MLP Classifier Score: {mlp.score(X_test, y_test)}")

import pickle as pkl
X_train['y'] = y_train
X_train.to_csv("./background_austrailian.csv")
X_test['y'] = y_test
X_test.to_csv("./dataset_austrailian.csv")

with open("./austrailian_modelRF.pkl", "wb") as f:
    pkl.dump(rf_pipeline, f)
with open("./austrailian_modelXG.pkl", "wb") as f:
    pkl.dump(xg, f)
with open("./austrailian_modelLR.pkl", "wb") as f:
    pkl.dump(lr, f)
with open("./austrailian_modelMLP.pkl", "wb") as f:
    pkl.dump(mlp, f)