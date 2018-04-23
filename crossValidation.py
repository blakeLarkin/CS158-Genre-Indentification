
# Gabriel Womark, Blake Larkin,  & Flora Gallina Jones

"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Feb 14
Description : Twitter
"""
from pprint import pprint

from string import punctuation

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

import pandas as pd
import utils
import getFeatures as gf

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from scipy.stats import norm, ttest_ind_from_stats

def performance(y_true, y_pred, metric="accuracy") :
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (discrete-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp

    accuracy = metrics.accuracy_score(y_true, y_pred)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)
    auroc = metrics.roc_auc_score(y_true, y_pred)
    f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

    if metric == 'accuracy':
        return accuracy
    elif metric == 'auroc':
        return auroc
    elif metric ==  'f1_score':
        return f1_score
    elif metric == 'sensitivity':
        return sensitivity
    elif metric == 'specificity':
        return specificity
    elif metric == 'precision':
        return precision

    ### ========== TODO : END ========== ###


def test_performance() :
    # np.random.seed(1234)
    # y_true = 2 * np.random.randint(0,2,10) - 1
    # np.random.seed(2345)
    # y_pred = (10 + 10) * np.random.random(10) - 10

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    #y_pred = [ 1, -1,  1, -1,  1,  1, -1, -1,  1, -1]
    # confusion matrix
    #          pred pos     neg
    # true pos      tp (2)  fn (4)
    #      neg      fp (3)  tn (1)
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics) :
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metric="accuracy") :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier 
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()


def performance_CI(clf, X, y, metric="accuracy") :
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC or DummyClassifier)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower        -- float, lower limit of confidence interval
        upper        -- float, upper limit of confidence interval
    """
    try :
        y_pred = clf.decision_function(X)
    except :
        y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)

    ### ========== TODO : START ========== ###
    # part 4b: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...)
    n, d = X.shape

    scores = np.zeros(1000)

    for t in range(1000):
        sample_X = np.zeros((n, d))
        sample_y = np.zeros(n)

        # Sample with replacement
        for i in range(n):
            index = np.random.randint(n)
            sample_X[i] = X[index]
            sample_y[i] = y[index]

        try:
            sample_y_pred = clf.decision_function(sample_X)
        except:
            sample_y_pred = clf.predict(sample_X)

        scores[t] = performance(sample_y, sample_y_pred, metric)

    lower, upper = (np.percentile(scores, 2.5), np.percentile(scores, 97.5))

    return score, lower, upper
    ### ========== TODO : END ========== ###


def bootstrap_mean_std(clf, X, y, metric='accuracy'):
    """
    Clf already trained
    """
    n, d = X.shape

    scores = np.zeros(1000)

    for t in range(1000):
        sample_X = np.zeros((n, d))
        sample_y = np.zeros(n)

        # Sample with replacement
        for i in range(n):
            index = np.random.randint(n)
            sample_X[i] = X[index]
            sample_y[i] = y[index]

        try:
            sample_y_pred = clf.decision_function(sample_X)
        except:
            sample_y_pred = clf.predict(sample_X)

        scores[t] = performance(sample_y, sample_y_pred, metric)

    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std 

def ttest(clf1, X_1, y_1, clf2, X_2, y_2, alpha=0.5):
    mean1, std1 = bootstrap_mean_std(clf1, X_1, y_1)
    mean2, std2 = bootstrap_mean_std(clf2, X_2, y_2)

    _, pvalue = ttest_ind_from_stats(mean1, std1, 1000, mean2, std2, 100)
    reject = pvalue <= alpha 

    return {'mean1': mean1, 'std1': std1, 'mean2': mean2, 'std2': std2, 'pvalue':pvalue, 'reject':reject}
    



def random_forest_hyperparameter_selection(data, iterations):
    X_train, y_train, X_test, y_test = data
    n,d = X_train.shape

    # Set up hyperparameter grid for randomized search
    n_estimators = np.arange(2, 101, 10)
    max_features = np.arange(1, d+1, (5 if d > 3 else max(1, (d+1)/float(6))))
    print(max_features)
    max_depth = np.arange(2, 101, 10)

    random_param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth
    }

    ##scoring = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
    print("Starting random grid search...")
    # peform randomized cv hyperparameter search
    # n_iterations is the number of random samples tested from the grid
    rs = RandomizedSearchCV(RandomForestClassifier(criterion='entropy'), 
                        random_param_grid, n_iter=iterations, scoring='accuracy', cv=9, random_state=10, n_jobs=-1)

    print("Fitting data ...")
    rs.fit(X_train, y_train)
    print("Done fitting...\n")
    # print("Best params:")
    # pprint(rs.best_params_)
    print("Starting full grid search...")

    # Construct a new parameter grid centered on the best value from the random search
    param_grid = {}

    for key in rs.best_params_:
        val = rs.best_params_[key]
        low_bound = max(val-30, 1)
        up_bound = val+30

        if key == 'max_features':
            up_bound = min(up_bound, d)

        param_grid[key] = np.arange(low_bound, up_bound+1, max(1, int((up_bound - low_bound)/6) ) )
    

    # perform cv for every combination of hyperparameters
    gs = GridSearchCV(RandomForestClassifier(criterion='entropy'), 
                    param_grid, scoring='accuracy', cv=9, n_jobs=-1)

    print("Fitting data ...")
    gs.fit(X_train,y_train)
    print("Done fitting...\n")
    print("Best params:")

    # these should give us the best params
    pprint(gs.best_params_)
    print("Best accuracy score:")
    pprint(gs.best_score_)


    return gs.best_params_, gs.best_score_








