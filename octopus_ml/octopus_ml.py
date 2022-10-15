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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from .misc import mem_measure, timer
from sklearn.linear_model import LassoCV
from IPython.display import set_matplotlib_formats

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tracemalloc

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
warnings.simplefilter(action="ignore", category=FutureWarning)


# ML visualizations

def plot_imp(
    clf, X, title, model="lgbm", num=30, importaince_type="gain", save_path=None
):
    # Feature importance plot supporting LGBM, RN and Catboost, return the list of features importance sorted by their contribution
    #sns.set_style("whitegrid")
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_context("paper", font_scale=1)
    sns.set_palette(gcbest)

 

    if model == "catboost":
        feature_imp = pd.DataFrame(
            {"Value": clf.get_feature_importance(), "Feature": X.columns}
        )
    elif model == "lgbm":
        feature_imp = pd.DataFrame(
            {
                "Value": clf.feature_importance(importance_type=importaince_type),
                "Feature": X.columns,
            }
        )
    else:
        feature_imp = pd.DataFrame(
            {"Value": clf.feature_importance(), "Feature": X.columns}
        )

    plt.figure(figsize=(11, num / 2.2))
    sns.set(font_scale=0.85)
    sns.barplot(
        color="#3498db",
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()

    return feature_imp


def confusion_matrix_plot(y_test, y_predict, save_path=None):
    # Confusion Matrix plot, binary classification including both normalized and absolute values plots
    set_matplotlib_formats('svg')

    plt.figure()
    cm = confusion_matrix(y_test, y_predict)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.set(font_scale=0.85)
    labels = ["0", "1"]
    plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
    sns.heatmap(
        cmn, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()

    plt.figure()
    cm = confusion_matrix(y_test, y_predict)
    cmn = cm.astype("int")
    sns.set(font_scale=0.85)
    labels = ["0", "1"]
    plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
    sns.heatmap(
        cmn, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()



def target_corr(X, y, df_cols, save_path=None):
    # Feature correlations to the target

    # catCols = X.select_dtypes(object").columns
    catCols = X[df_cols].select_dtypes(include=["category", object]).columns

    # print (catCols)
    X = X[df_cols].drop(columns=catCols)

    # reg = LassoCV(n_alphas=30, eps=1e-3, max_iter=20, precompute=False)
    reg = LassoCV()
    reg.fit(X.fillna(-1), y)

    sns.set_style("whitegrid")
    # print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X.fillna(-1), y))
    coef = pd.Series(reg.coef_, index=X.columns)
    imp_coef = coef.sort_values()
    size = len(X.columns) / 1.6
    plt.rcParams["figure.figsize"] = (10, size)
    imp_coef.plot(kind="barh", color="#3498db")
    plt.title("Features correlations to target")

    if save_path:
        plt.savefig(save_path)


def label_dist(df, label, y=None):
    # Target distribution analysis
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_context("paper", font_scale=1)
    sns.set_palette(gcbest)

    fig, ax = plt.subplots(1, 2)
    plt.figure(figsize=(2, 3))
    #sns.set_style("whitegrid")
    set_matplotlib_formats('svg')
    #plt.style.use("seaborn-notebook")
    #sns.set_context("paper", font_scale=1.3)
    sns.set_context(font_scale=1)
    
    if y is not None:
        splot = sns.countplot("label", data=y.to_frame("label").reset_index(), ax=ax[0])

        for p in splot.patches:
            splot.annotate(
                format(p.get_height(), ".0f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
        y.value_counts().plot.pie(explode=[0, 0.2], autopct="%1.2f%%", ax=ax[1])
    else:
        splot = sns.countplot(label, data=df, ax=ax[0])
        for p in splot.patches:
            splot.annotate(
                format(p.get_height(), ".0f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        df[label].value_counts().plot.pie(explode=[0, 0.2], autopct="%1.2f%%", ax=ax[1])
    fig.show()


def roc_curve_plot(y_test, predictions, save_path=None):
    # Roc curve visualization, binary classification including AUC calculation
    set_matplotlib_formats('svg')
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_palette(gcbest)
    sns.set_context("paper", font_scale=1)

    #plt.style.use('classic')
    sns.set_style("whitegrid")
    rf_roc_auc = roc_auc_score(y_test, predictions)
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, predictions)

    plt.figure(figsize=(6, 5.5))
    plt.plot(rf_fpr, rf_tpr, label="AUC = %0.3f" % rf_roc_auc, color="#3498db")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Receiver operating characteristic", fontsize=12)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()



def hist_target(df, feature, target, fill=False):
    # histogram with an hue of the target class
    set_matplotlib_formats('svg')
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_context("paper", font_scale=1)
    sns.set_palette(gcbest)

    if fill==True:
        sns.histplot(
            data=df, 
            x=feature, 
            hue=target, 
            stat="probability", 
            multiple="fill", 
            shrink=.7
            )
    else:
        sns.displot(
            data=df,
            bins=25,
            kind="hist",
            x=feature,
            hue=target,
            multiple="stack",
            height=3.2,
            aspect=1.6,
        )


def target_pie(df, target):
    # pie chart of the target class distribution
    sns.set_style("whitegrid")
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_palette(gcbest)

    #plt.style.use("fivethirtyeight")
    plt.figure(figsize=(4, 3))
    #sns.set_context("paper", font_scale=1)
    df[target].value_counts().plot.pie(explode=[0, 0.2], autopct="%1.2f%%")


def cv_plot(
    arr_f1_weighted,
    arr_f1_macro,
    arr_f1_positive,
    AxisName,
    mode="full",
    save_path=None,
):
    # Visualization of the CV folds, F1 macro and F1 positive class
    set_matplotlib_formats('svg')
    sns.set_context(font_scale=1)
    gcbest = ["#3498db", "#2ecc71"]
    sns.set_palette(gcbest)

    #plt.style.use("fivethirtyeight")
    #sns.set_context("paper", font_scale=1)
    sns.set_style("whitegrid")

    if mode == "fast":
        plt.figure(figsize=(5, 6))
    else:
        plt.figure(figsize=(10,5.58))
           
    index = np.arange(len(arr_f1_weighted))
    bar_width = 0.30
    opacity = 0.8

    plt.bar(
        index - bar_width / 2, arr_f1_macro, bar_width, alpha=opacity, label="F1 Macro"
    )

    plt.bar(
        index + bar_width / 2,
        arr_f1_positive,
        bar_width,
        alpha=opacity,
        label="F1 Positive",
    )

    plt.xticks(np.arange(len(arr_f1_weighted)), fontsize=10)
    #plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=13)
    plt.yticks(fontsize=10)
    plt.ylabel("F1", fontsize=10)
    plt.xlabel("Folds", fontsize=10)
    plt.title("%s, 5-Folds Cross Validation" % AxisName[0 : len(AxisName)], fontsize=13)
    plt.legend(["F1 macro", "F1 positive"], loc="upper right", fontsize=10)
    plt.grid(True)
    

    if save_path:
        plt.savefig(save_path)


def preds_distribution(
    y_true,
    y_pred,
    bins=100,
    title="Predictions Distribution",
    normalize=False,
    ax=None,
    title_fontsize="medium",
    max_y=None,
    save_path=None,
):  
    set_matplotlib_formats('svg')
    sns.set_style("whitegrid")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))

    predictions_proba = np.array(y_pred)
    y_bool = np.array(y_true) > 0
    y_pred_true = predictions_proba[y_bool]
    y_pred_false = predictions_proba[~y_bool]

    # print (y_pred_true)
    # matplotlib normalize is using the bin width, just calculate it by our own...
    weights_false = (
        np.ones(len(y_pred_false)) / len(y_pred_false) if normalize else None
    )
    weights_true = np.ones(len(y_pred_true)) / len(y_pred_true) if normalize else None

    ax.hist(
        y_pred_false,
        bins=bins,
        color="r",
        alpha=0.5,
        label="negative",
        weights=weights_false,
    )
    ax.hist(
        y_pred_true,
        bins=bins,
        color="g",
        alpha=0.5,
        label="positive",
        weights=weights_true,
    )
    ax.set_title(title, fontsize=title_fontsize)
    # _set_lim(max_y, ax.set_ylim)
    ax.set_ylim(0, max_y)
    ax.legend(loc="best")

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()

    return ax


def target_shape(df, target):
    fig, ax = plt.subplots(1, 2)
    set_matplotlib_formats('svg')

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(2.5, 3.5))
    sns.set_context("paper", font_scale=1.2)
    splot = sns.countplot(target, data=df, ax=ax[0])

    for p in splot.patches:
        splot.annotate(
            format(p.get_height(), ".0f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    df[target].value_counts().plot.pie(explode=[0, 0.2], autopct="%1.2f%%", ax=ax[1])
    fig.show()


def lgbm(X_train, y_train, X_test, y_test, num, params=None):
    # Training function for LGBM with basic categorical features treatment and close to default params

    categorical_features = []
    for c in X_train.columns:
        col_type = X_train[c].dtype
        if col_type == "object" or col_type.name == "category":
            # an option in case the data(pandas dataframe) isn't passed with the categorical column type
            # X[c] = X[c].astype('category')
            categorical_features.append(c)

    for c in X_test.columns:
        col_type = X_test[c].dtype
        if col_type == "object" or col_type.name == "category":
            # an option in case the data(pandas dataframe) isn't passed with the categorical column type
            # X[c] = X[c].astype('category')
            categorical_features.append(c)

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(X_test, y_test, categorical_feature=categorical_features)

    if params == None:
        params = {
            "objective": "binary",
            "boosting": "gbdt",
            "scale_pos_weight": 0.02,
            "learning_rate": 0.005,
            "seed": 100
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

    clf = lgb.train(
        params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=num
    )

    return clf


def adjusted_classes(y_scores, t):
    # transformation from prediction probabolity to class given the threshold
    return [1 if y >= t else 0 for y in y_scores]


@timer
@mem_measure
def cv_adv(
    X,
    y,
    threshold,
    iterations,
    shuffle=True,
    params=None,
    mode="classification",
    method="full",
):
    # Cross Validation - stratified with and without shuffeling

    print(
        "--------------------------- Running Cross-Validation - "
        + mode
        + ", mode: "
        + method
        + " ---------------------------"
    )
    print("-> Starting 5-folds CV - Shuffle: " + str(shuffle))
    arr_f1_weighted = np.array([])
    arr_f1_macro = np.array([])
    arr_f1_positive = np.array([])
    arr_recall = np.array([])
    arr_precision = np.array([])
    prediction_folds = []
    preds_folds = []
    y_folds = []
    stacked_models = []
    index_column = []

    if mode == "regression":
        skf = KFold(n_splits=5)
    else:
        if shuffle == False:
            skf = StratifiedKFold(n_splits=5,  shuffle=shuffle)
        else:
            skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=shuffle)
            
    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = lgbm(X_train, y_train, X_test, y_test, iterations, params)
        preds = clf.predict(X_test)
        predictions = []
        predictions = adjusted_classes(preds, threshold)
        stacked_models.append(clf)
        index_column.extend(X_test.index.values.tolist())

        """ Multiclass 
        predictions = clf.predict(X_test)
        predictions_classes = []
        for i in predictions:   
            print (np.argmax(i))
            predictions_classes.append(np.argmax(i))  
        """

        if mode == "regression":
            prediction_folds.extend(predictions)
            preds_folds.extend(preds)
            y_folds.extend(y_test)
            final_clf = lgbm(X, y, X_test, y_test, iterations, params)

        else:
            prediction_folds.extend(predictions)
            preds_folds.extend(preds)
            y_folds.extend(y_test)
            arr_f1_weighted = np.append(
                arr_f1_weighted, f1_score(y_test, predictions, average="weighted")
            )
            arr_f1_macro = np.append(
                arr_f1_macro, f1_score(y_test, predictions, average="macro")
            )
            arr_f1_positive = np.append(
                arr_f1_positive, f1_score(y_test, predictions, average="binary")
            )

        if method == "fast":
            break

    print("-> Training final model on the full-set\n")
    final_clf = lgbm(X, y, X_test, y_test, iterations, params)

    return {
        "final_clf": final_clf,
        "f1_weighted": arr_f1_weighted,
        "f1_macro": arr_f1_macro,
        "f1_positive": arr_f1_positive,
        "predictions_folds": prediction_folds,
        "predictions_proba": preds_folds,
        "y": y_folds,
        "index": index_column,
        "stacked_models": stacked_models,
    }


def cv_grouped(X, y, threshold, iterations, group_name, shuffle=True):
    arr_f1_weighted = np.array([])
    arr_f1_macro = np.array([])
    arr_f1_positive = np.array([])
    arr_recall = np.array([])
    arr_precision = np.array([])
    prediction_folds = []
    preds_folds = []
    y_folds = []

    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=shuffle)

    for train_index, test_index in tqdm(skf.split(skf.split(X, y))):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = lgbm(X_train, y_train, X_test, y_test, iterations)
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
        arr_f1_weighted = np.append(
            arr_f1_weighted, f1_score(y_test, predictions, average="weighted")
        )
        arr_f1_macro = np.append(
            arr_f1_macro, f1_score(y_test, predictions, average="macro")
        )
        arr_f1_positive = np.append(
            arr_f1_positive, f1_score(y_test, predictions, average="binary")
        )

    return (
        clf,
        arr_f1_weighted,
        arr_f1_macro,
        arr_f1_positive,
        prediction_folds,
        preds_folds,
        y_folds,
    )


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
        arr_f1_weighted = np.append(
            arr_f1_weighted, f1_score(y_test, predictions, average="weighted")
        )
        arr_f1_macro = np.append(
            arr_f1_macro, f1_score(y_test, predictions, average="macro")
        )
        arr_f1_positive = np.append(
            arr_f1_positive, f1_score(y_test, predictions, average="binary")
        )

        final_clf = lgbm(X, y, X_test, y_test, iterations, params, no_val=True)

    return (
        final_clf,
        arr_f1_weighted,
        arr_f1_macro,
        arr_f1_positive,
        prediction_folds,
        preds_folds,
        y_folds,
    )
