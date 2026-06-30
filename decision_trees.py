from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn import tree
import os
import time

def split_data(feature_df):
    """Split the feature table into 75/25 train/test sets.

    Args:
        feature_df: DataFrame with a 'label' column and numeric predictors.

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    response_name = "label"
    response = feature_df[response_name]
    predictor_cols = [c for c in feature_df.columns if c != response_name]
    predictors = feature_df[predictor_cols]

    x_train, x_test, y_train, y_test = train_test_split(predictors,response,random_state = 1234)
    
    return x_train, x_test, y_train, y_test

def plot_decision_tree(mdl, x_train, MODEL_DIR):
    fig, ax = plt.subplots(figsize=(40, 20))
    feature_names = list(x_train.columns)
    tree.plot_tree(mdl, feature_names=list(feature_names), max_depth = 3, filled=True, ax=ax)
    fig.savefig(os.path.join(MODEL_DIR, 'decision_tree.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot feature importances
    importances = mdl.feature_importances_
    sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    ax.barh([feature_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title('Decision Tree Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, 'feature_importances.png'), dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_decision_tree(mdl, x_test, y_test, x_train, y_train):
    # ----------------------------
    # Test on test data
    # ----------------------------
    y_pred = mdl.predict(x_test)

    # ----------------------------
    # Evaluate model
    # ----------------------------
    print('\nTest Accuracy Score:')
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #Check for overfitting by comparing accuracy of predicting training data to predicting test data
    y_pred_train = mdl.predict(x_train)
    print('\nTrain Accuracy Score:')
    print(accuracy_score(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))


def fit_tree(x_train, y_train, prune=False):
    """Fit a DecisionTreeClassifier, optionally with hyperparameter tuning.

    When prune=True, tunes max_depth, min_samples_split, min_samples_leaf, and
    ccp_alpha via GridSearchCV using data-driven grids scaled to training set size.

    Args:
        x_train: Feature matrix for training.
        y_train: Label vector for training.
        prune: If True, run GridSearchCV to select hyperparameters.

    Returns:
        Tuple of (fitted DecisionTreeClassifier, training_time_sec).
    """
    clf = DecisionTreeClassifier(random_state=1234)
    t_start = time.time()

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

    training_time_sec = time.time() - t_start
    print(f'  Training time: {training_time_sec:.1f} sec')
    return mdl, training_time_sec