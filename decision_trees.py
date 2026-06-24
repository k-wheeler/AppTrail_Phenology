from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

def split_data(feature_df):
    response_name = "label"
    response = feature_df[response_name]
    predictor_cols = [c for c in feature_df.columns if c != response_name]
    predictors = feature_df[predictor_cols]

    x_train, x_test, y_train, y_test = train_test_split(predictors,response,random_state = 1234)
    
    return x_train, x_test, y_train, y_test

def fit_tree(x_train, y_train, prune=False):
    clf = DecisionTreeClassifier(random_state=1234)

    #Without pruning
    if not prune:
        mdl = clf.fit(x_train, y_train)

    #With pruning
    else:
        n = len(x_train)
        path = clf.cost_complexity_pruning_path(x_train, y_train)
        ccp_alphas = path.ccp_alphas

        grid = {
            'max_depth':        list(range(1, int(np.log2(n)) + 1)), #1 up to log2 of sample size
            'min_samples_split': [max(2, int(n * p)) for p in [0.005, 0.01, 0.02, 0.05]], #2-5% of sample size
            'min_samples_leaf':  [max(1, int(n * p)) for p in [0.005, 0.01, 0.02]], #1-2% of sample size
            'ccp_alpha':         list(np.linspace(ccp_alphas[0], ccp_alphas[-1], 10)), #Only select 10 to test
        }

        gcv = GridSearchCV(estimator=clf, param_grid=grid)
        gcv.fit(x_train, y_train)
        mdl = gcv.best_estimator_

    return mdl