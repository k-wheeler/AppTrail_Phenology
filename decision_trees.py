from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(feature_df):
    response_name = "label"
    response = feature_df[response_name]
    predictor_cols = [c for c in feature_df.columns if c != response_name]
    predictors = feature_df[predictor_cols]

    x_train, x_test, y_train, y_test = train_test_split(predictors,response,random_state = 1234)
    
    return x_train, x_test, y_train, y_test

def fit_tree(x_train,y_train):
    clf = DecisionTreeClassifier()
    mdl = clf.fit(x_train, y_train)
    return mdl