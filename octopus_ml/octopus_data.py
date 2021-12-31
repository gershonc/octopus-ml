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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import tracemalloc


def correlations(df, cols):
    sns.set(style="white")
    corr = df[cols].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(8, 7.5))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )


def diff_list(li1, li2):
    return list(list(set(li1) - set(li2)) + list(set(li2) - set(li1)))


def recieve_fps(df, index, y_folds, preds_folds, threshold=0.5, top=10):
    preds_labels = pd.DataFrame(index, columns=["index"])
    preds_labels["label"] = y_folds
    preds_labels["preds_proba"] = preds_folds
    preds_labels["preds_class"] = preds_labels["preds_proba"].apply(
        lambda x: 1 if x > threshold else 0
    )
    if threshold != 0.5:
        print("recieved threshold of: ", threshold, "\n")
    else:
        print("calculating using threshold of: ", threshold)
    preds_labels_sorted = preds_labels[preds_labels["label"] == 0].sort_values(
        by="preds_proba", ascending=False
    )
    return preds_labels_sorted.head(top).reset_index(drop=True)


def recieve_fns(df, index, y_folds, preds_folds, threshold=0.5, top=10):
    preds_labels = pd.DataFrame(index, columns=["index"])
    preds_labels["label"] = y_folds
    preds_labels["preds_proba"] = preds_folds
    preds_labels["preds_class"] = preds_labels["preds_proba"].apply(
        lambda x: 1 if x > threshold else 0
    )
    if threshold != 0.5:
        print("recieved threshold of: ", threshold, "\n")
    else:
        print("calculating using threshold of: ", threshold)

    preds_labels_sorted = preds_labels[preds_labels["label"] == 1].sort_values(
        by="preds_proba", ascending=True
    )
    return preds_labels_sorted.head(top).reset_index(drop=True)


@timer
@mem_measure
def sampling_by_group(df, target, num_negative_instances, group_by_name):
    # Sampling method by
    grouped_df = df.groupby(group_by_name)
    print("Number of groups: ", len(grouped_df))

    grouped_df.target.max()
    g_df = grouped_df.target.max().reset_index()
    positive = len(g_df[g_df[target] == 1])
    negative = len(g_df[g_df[target] == 0])

    print(
        "number of positive groups: ",
        positive,
        "\nnumber of negative groups: ",
        negative,
    )

    negative_indices = g_df[g_df[target] == 0].index
    positive_indices = g_df[g_df[target] == 1].index
    random_indices = np.random.choice(
        negative_indices, num_negative_instances, replace=False
    )
    sample_indices = np.concatenate([positive_indices, random_indices])

    g_df_sampled = g_df.loc[sample_indices]
    print(g_df_sampled[group_by_name])

    sampled_df = df[df[group_by_name].isin(g_df_sampled[group_by_name])]
    # print ("new dataset shape: ", df_sampled.shape)
    return sampled_df


@timer
@mem_measure
def sampling(df, target, num_instances):
    positive = len(df[df[target] == 1])
    negative = len(df[df[target] == 0])
    print(
        "number of positive instances:",
        positive,
        "\nnumber of negative instance : ",
        negative,
    )
    negative_indices = df[df[target] == 0].index
    positive_indices = df[df[target] == 1].index
    random_indices = np.random.choice(negative_indices, num_instances, replace=False)
    sample_indices = np.concatenate([positive_indices, random_indices])
    sampled_df = df.loc[sample_indices]
    print("new dataset shape: ", sampled_df.shape)
    return sampled_df


@timer
@mem_measure
def data_leakage(df, cols):
    CRED = "\033[91m"
    CEND = "\033[0m"
    CGREEN = "\33[32m"

    duplicateRowsDF = df[df.duplicated(cols, keep="first")]

    if round(len(duplicateRowsDF) / len(df) * 100, 2) > 0:
        print(
            "\n-> Total number of duplicate instances:",
            len(duplicateRowsDF),
            "out of",
            len(df),
            ":",
            CRED,
            round(len(duplicateRowsDF) / len(df) * 100, 2),
            "%",
            CEND,
            "\n",
        )
        print("Top duplicate instances:", "\n")
        df["All_features"] = df[cols].astype(str).apply(" | ".join, axis=1)
        print(df["All_features"].value_counts().head(12))

    else:
        print(
            "->",
            CGREEN,
            "Passed the data leakage test - no duplicate intstances detected",
            CEND,
        )


def sampling_within_group(df, target, num_negative_instances, group_by_name, frac):
    # Sampling method by
    grouped_df = df.groupby(group_by_name)
    print("Original shape: ", df.shape)

    print("Number of groups: ", len(grouped_df))

    g_df = grouped_df[target].max().reset_index()
    positive = len(g_df[g_df[target] == 1])
    negative = len(g_df[g_df[target] == 0])

    print(
        "number of positive groups: ",
        positive,
        "\nnumber of negative groups: ",
        negative,
    )

    negative_indices = g_df[g_df[target] == 0].index
    positive_indices = g_df[g_df[target] == 1].index
    random_indices = np.random.choice(
        negative_indices, num_negative_instances, replace=False
    )
    # sample_indices = np.concat([positive_indices, random_indices])

    # g_df_sampled = g_df.loc[sample_indices]
    # print (g_df_sampled[group_by_name])

    # sampled_df = df[df[group_by_name].isin(g_df_sampled[group_by_name])]
    positive_instances = df[df[target] == 1]
    negative_instances = df[df[target] == 0]
    # selected_negative_instances = df.sample(frac=fracture, replace=False, random_state=1)
    # selected_negative_instances = df.sample(frac=fracture, replace=False, random_state=1)
    selected_negative_instances = (
        df[df[target] == 0].groupby(group_by_name).sample(frac=frac, random_state=2)
    )

    sampled_df = pd.concat([positive_instances, selected_negative_instances])

    # sampled_df = df[df[group_by_name].isin(g_df_sampled[group_by_name])]
    # print ("new dataset shape: ", df_sampled.shape)
    # return sampled_df

    print("new dataset shape: ", sampled_df.shape)
    positive = len(sampled_df[sampled_df[target] == 1])
    negative = len(sampled_df[sampled_df[target] == 0])

    print(
        "number of positive instances: ",
        positive,
        "\nnumber of negative instances: ",
        negative,
    )
    return sampled_df


def cat_features_proccessing(df):
    # detect and change tyoe of categorical features, retun a list of categorical features and modified dataframe
    categorical_features = []
    for c in df.columns:
        col_type = df[c].dtype
        if col_type == "object" or col_type.name == "category":
            df[c] = df[c].astype("category")
            categorical_features.append(c)
    return categorical_features, df


def detect_categorical(df, num_category_max, threshold=1):
    category_features = []

    for each in df.columns:
        if (df[each].nunique() < num_category_max) & (df[each].nunique() > 2):
            # print (df[each].nunique())
            if df[each].nunique() / df[each].count() < threshold:
                category_features.append(each)

    return category_features


def convert_to_categorical(df, categorical_features):
    for i, each in enumerate(categorical_features):
        if i % 10 == 0:
            print(i)
        df[each] = df[each].astype("category")


def anomalies(df, df_cols):
    clf = IsolationForest(max_samples=100)
    clf.fit(df[df_cols].fillna(0))
    y_pred_train = clf.predict(df[df_cols].fillna(0))
    # y_pred_dev = clf.predict(X_dev)
    y_pred_test = clf.predict(df[df_cols].fillna(0))
    return df[y_pred_test == 1][df_cols].head(10)
