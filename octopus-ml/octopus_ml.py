#                                                                         
#        _/_/      _/_/_/  _/_/_/_/_/    _/_/    _/_/_/    _/    _/    _/_/_/
#      _/    _/  _/            _/      _/    _/  _/    _/  _/    _/  _/       
#      _/    _/  _/            _/      _/    _/  _/_/_/    _/    _/    _/_/    
#      _/    _/  _/            _/      _/    _/  _/        _/    _/        _/   
#        _/_/      _/_/_/      _/        _/_/    _/          _/_/    _/_/_/   
#
#
#    __          _____            __               _____    __     _ __          
#   / /  __ __  / ___/__ _______ / /  ___  ___    / ___/__ / /__  (_) /_____ ____
#  / _ \/ // / / (_ / -_) __(_-</ _ \/ _ \/ _ \  / /__/ -_) / _ \/ /  '_/ -_) __/
# /_.__/\_, /  \___/\__/_/ /___/_//_/\___/_//_/  \___/\__/_/_//_/_/_/\_\\__/_/   
#      /___/                                                                     

from tqdm import tqdm
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from .misc import mem_measure, timer

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tracemalloc

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.simplefilter(action='ignore', category=FutureWarning)


# ML visualizations

def plot_imp(clf, X, title, model='lgbm', num=30):
    # Feature importance plot supporting LGBM, RN and Catboost, return the list of features importance sorted by their contribution
    sns.set_style("whitegrid")

    if model == 'catboost':
        feature_imp = pd.DataFrame(
            {'Value': clf.get_feature_importance(), 'Feature': X.columns})
    elif model == 'lgbm':
        feature_imp = pd.DataFrame(
            {'Value': clf.feature_importance(importance_type='gain'), 'Feature': X.columns})
    else:
        feature_imp = pd.DataFrame(
            {'Value': clf.feature_importance(), 'Feature': X.columns})

    plt.figure(figsize=(16, 20))
    sns.set(font_scale=1.1)
    sns.barplot(color="#3498db", x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                                      ascending=False)[0:num])
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return feature_imp


def confusion_matrix_plot(y_test, y_predict):
    # Confusion Matrix plot, binary classification including both normalized and absolute values plots

    plt.figure()
    cm = confusion_matrix(y_test, y_predict)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
    sns.set(font_scale=1.1)
    labels = ['0', '1']
    plt.figure(figsize=(7, 5.5))
    #sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
    sns.heatmap(cmn, xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

    plt.figure()
    cm = confusion_matrix(y_test, y_predict)
    cmn = cm.astype('int')
    sns.set(font_scale=1.1)
    labels = ['0', '1']
    plt.figure(figsize=(7, 5.5))
    #sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
    sns.heatmap(cmn, xticklabels=labels, yticklabels=labels,
                annot=True, fmt='d', cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()


def roc_curve_plot(y_test, predictions):
    # Roc curve visualization, binary classification including AUC calculation

    sns.set_style("whitegrid")
    rf_roc_auc = roc_auc_score(y_test, predictions)
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(rf_fpr, rf_tpr, label='LGBM (area = %0.3f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver operating characteristic', fontsize=17)
    plt.legend(loc="lower right")
    plt.show()


def cv_plot(arr_f1_weighted, arr_f1_macro, arr_f1_positive, AxisName):
    # Visualization of the CV folds, F1 macro and F1 positive class

    sns.set_style("whitegrid")
    plt.figure(figsize=(13, 7))
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_palette(gcbest)

    index = np.arange(len(arr_f1_weighted))
    bar_width = 0.30
    opacity = 0.8

    plt.bar(index - bar_width/2, arr_f1_macro, bar_width,
            alpha=opacity, label='F1 Macro')

    plt.bar(index + bar_width/2, arr_f1_positive, bar_width,
            alpha=opacity, label='F1 Positive')

    plt.xticks(np.arange(len(arr_f1_weighted)), fontsize=14)
    plt.ylabel('F1', fontsize=14)
    plt.xlabel('Folds', fontsize=14)
    plt.title('%s, 5-Folds Cross Validation' %
              AxisName[0:len(AxisName)], fontsize=17)
    plt.legend(['F1 macro', 'F1 positive'], loc='upper right', fontsize=14)
    plt.grid(True)


def lgbm(X_train, y_train, X_test, y_test, num, params=None):
    # Training function for LGBM with basic categorical features treatment and close to default params

    categorical_features = []
    for c in X_train.columns:
        col_type = X_train[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            # an option in case the data(pandas dataframe) isn't passed with the categorical column type
            #X[c] = X[c].astype('category')
            categorical_features.append(c)

    lgb_train = lgb.Dataset(
        X_train, y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(
        X_test, y_test, categorical_feature=categorical_features)

    if params == None:
        params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'scale_pos_weight': 0.02,
            'learning_rate': 0.005,
            'seed': 100
            # 'categorical_feature': 'auto',
            # 'metric': 'auc',
            # 'scale_pos_weight':0.1,
            # 'learning_rate': 0.02,
            # 'num_boost_round':2000,
            # "min_sum_hessian_in_leaf":1,
            # 'max_depth' : 100,
            # "bagging_freq": 2,
            # "num_leaves":31,
            # "bagging_fraction" : 0.4,
            # "feature_fraction" : 0.05,
        }

    clf = lgb.train(params, lgb_train, num)

    return clf


def adjusted_classes(y_scores, t):
    # transformation from prediction probabolity to class given the threshold
    return [1 if y >= t else 0 for y in y_scores]


@timer
@mem_measure
def cv(X, y, threshold, iterations, shuffle=True, params=None):
    # Cross Validation - stratified with and without shuffeling
    arr_f1_weighted = np.array([])
    arr_f1_macro = np.array([])
    arr_f1_positive = np.array([])
    arr_recall = np.array([])
    arr_precision = np.array([])
    prediction_folds = []
    preds_folds = []
    y_folds = []

    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=shuffle)

    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = lgbm(X_train, y_train, X_test, y_test, iterations, params)
        preds = clf.predict(X_test)

        predictions = []
        predictions = adjusted_classes(preds, threshold)

        """ Multiclass 
        predictions = clf.predict(X_test)
        predictions_classes = []
        for i in predictions:   
            print (np.argmax(i))
            predictions_classes.append(np.argmax(i))  
        """

        prediction_folds.extend(predictions)
        preds_folds.extend(preds)
        y_folds.extend(y_test)
        arr_f1_weighted = np.append(arr_f1_weighted, f1_score(
            y_test, predictions, average='weighted'))
        arr_f1_macro = np.append(arr_f1_macro, f1_score(
            y_test, predictions, average='macro'))
        arr_f1_positive = np.append(arr_f1_positive, f1_score(
            y_test, predictions, average='binary'))

    return clf, arr_f1_weighted, arr_f1_macro, arr_f1_positive, prediction_folds, preds_folds, y_folds




def cv_grouped(X,y,threshold,iterations,group_name,shuffle=True):
    arr_f1_weighted=np.array([])
    arr_f1_macro=np.array([])    
    arr_f1_positive=np.array([])    
    arr_recall=np.array([])    
    arr_precision=np.array([])
    prediction_folds=[]
    preds_folds=[]
    y_folds=[]
    
    skf=StratifiedKFold(n_splits=5, random_state=2, shuffle=shuffle)

    for train_index, test_index in tqdm(skf.split(skf.split(X, y))):
        
        X_train, X_test =  X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf=lgbm(X_train,y_train,X_test,y_test,iterations)
        preds = clf.predict(X_test)
        
        predictions = []
        predictions=adjusted_classes(preds, threshold)
          
        """ Multiclass 
        predictions = clf.predict(X_test)
        predictions_classes = []
        for i in predictions:   
            print (np.argmax(i))
            predictions_classes.append(np.argmax(i))  
        """
        prediction_folds.extend(predictions)
        preds_folds.extend(preds)
        y_folds.extend(y_test)
        arr_f1_weighted=np.append(arr_f1_weighted, f1_score(y_test, predictions, average='weighted'))
        arr_f1_macro=np.append(arr_f1_macro, f1_score(y_test, predictions, average='macro'))
        arr_f1_positive=np.append(arr_f1_positive, f1_score(y_test, predictions, average='binary'))
   
    return clf,arr_f1_weighted,arr_f1_macro,arr_f1_positive,prediction_folds,preds_folds,y_folds

