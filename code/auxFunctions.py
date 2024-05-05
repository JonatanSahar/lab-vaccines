#!/usr/bin/env python3

# Plotting related
import os
import sys

sys.tracebacklimit = 0

import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio


# Scikit-learn related imports
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_colwidth", None)
pd.options.mode.copy_on_write = True

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.cluster import KMeans
from math import log

import importlib

# Constants for this project
import constants
from constants import *

import papermill as pm
import shutil

from IPython.display import Markdown as md

# Change the current working directory
os.chdir("/home/yonatan/Documents/projects/vaccines/code")


# Define auxilary functions
def get_dir_by_name(dir_name):
    # Define the starting directory
    current_dir = os.getcwd()

    # Traverse up the directory tree until we find a directory named "data"
    while current_dir != "/":
        if dir_name in os.listdir(current_dir):
            data_dir = os.path.join(current_dir, dir_name)
            return data_dir
        current_dir = os.path.dirname(current_dir)
    else:
        print(f"Directory {dir_name} not found in the parent directories.")
        raise (Exception())

def remove_duplicate_accessions(dataset, immage_col, uid_col):
    """Sometimes there are multiple geo_accession numbers, like in GSE48018.SDY1276.
    Average the IMMAGE, since all else is the same"""
    first_uid = dataset.iloc[0][uid_col]
    accessions = dataset[dataset[uid_col] == first_uid]["geo_accession"].unique()
    if len(accessions) > 1:
        # print(f"Multiple accession detected, Collapsing by averaging on IMMAGE value")
        dataset = dataset.groupby(uid_col, as_index=False).agg(
            {
                immage_col: "mean",
                **{col: "first" for col in dataset.columns if col not in [uid_col, immage_col]},
            }
        )

    accessions = dataset[dataset[uid_col] == first_uid]["geo_accession"].unique()
    assert len(accessions) == 1

    return dataset

def get_threshold_from_probability(prob, intercept, slope):
    return -1 * (log(1 / prob - 1) + intercept) / slope

def plot_response(data, dataset_name, strain, features=""):
    """
    Plots the response distribution based on specified features for a given dataset and strain.

    Parameters:
    - data: DataFrame containing the dataset to be plotted
    - dataset_name: Name of the dataset
    - strain: Strain information for the dataset
    - features: List of features to be plotted (optional)

    Returns:
    - Displays a visual representation of the response distribution based on the specified features.

    Example Usage:
    plot_response(my_data, "Example Dataset", "Strain A", ["Feature1", "Feature2"])
    """
    custom_palette = {"Non-Responders": "orange", "Responders": "#3498db"}
    if len(features) == 1:
        col_name = features[0]
        sorted_data = data.sort_values(col_name, ignore_index=True).reset_index()
        sns.scatterplot(
            data=sorted_data, x="index", y=col_name, hue="Label text", palette=custom_palette
        )

        plt.title(f"Sorted {col_name} vs Index")
        plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
        plt.show()

    if len(features) == 2:
        fig, axs = plt.subplots(
            1, 2, figsize=(8, 3)
        )  # Creates a figure with two side-by-side subplots
        col_name_1 = features[0]
        col_name_2 = features[1]

        sorted_data = data.sort_values(col_name_1, ignore_index=True).reset_index()
        scatter_1 = sns.scatterplot(
            ax=axs[0],
            data=sorted_data,
            x="index",
            y=col_name_1,
            hue="Label text",
            palette=custom_palette,
        )
        axs[0].set_title(f"Sorted {col_name_1} vs Index")
        axs[0].get_legend().remove()  # Removes the legend

        sorted_data = data.sort_values(col_name_2, ignore_index=True).reset_index()
        scatter_2 = sns.scatterplot(
            ax=axs[1],
            data=sorted_data,
            x="index",
            y=col_name_2,
            hue="Label text",
            palette=custom_palette,
            legend=False,  # turns off individual legend
        )
        axs[1].set_title(f"Sorted {col_name_2} vs Index")

        handles, labels = scatter_1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.2, 0.5))
        fig.suptitle(f"Response distribution: {dataset_name} {strain}")
        plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
        plt.show()

def plot_desicion_threshold_ROC(
    data,
    fpr,
    tpr,
    prob_column,
    optimal_idx,
    feature_threshold,
    prob_threshold,
    AUC,
    dataset_name,
    strain,
    features="",
):
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 4)
    )  # Creates a figure with two side-by-side subplots

    naive_classification_precision = data["y"].mean()

    # Plot ROC on the first subplot
    axs[0].plot(
        fpr, tpr, label=f"ROC curve (area = {AUC : 0.2f})", color="#9b59b6"
    )
    axs[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axs[0].plot(fpr[optimal_idx], tpr[optimal_idx], marker="o", markersize=5, color="red")
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel("fpr")
    axs[0].set_ylabel("tpr")
    axs[0].set_title("ROC curve")
    axs[0].legend(loc="lower right")

    custom_palette = {"Non-Responders": "orange", "Responders": "#3498db"}

    if len(features) == 1:
        col_name = features[0]
        # Plot sorted feature values vs Index on the second subplot
        sorted_data = data.sort_values(col_name, ignore_index=True).reset_index()
        sns.scatterplot(
            ax=axs[1],
            data=sorted_data,
            x="index",
            y=col_name,
            hue="Label text",
            palette=custom_palette,
        )
        axs[1].axhline(y=feature_threshold, color="black", linestyle="--")
        axs[1].set_title(f"Sorted {col_name} vs Index")
    else:  # len(features) > 1
        predicted_true = data.loc[data[prob_column] >= prob_threshold]

        sns.scatterplot(
            ax=axs[1],
            data=predicted_true,
            x=features[0],
            y=features[1],
            marker="x",
            s=100,
            color="red",
        )

        sns.scatterplot(
            ax=axs[1],
            data=data,
            x=features[0],
            y=features[1],
            hue="Label text",
            palette=custom_palette,
        )
        axs[1].set_title(f"IMMAGE and Age, X=predicted non-responder")

    fig.suptitle(f"Probability-based threshold with ROC\n{dataset_name} {strain}")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
    plt.show()

def plot_desicion_threshold_PRC(
    data,
    precision,
    recall,
    prob_column,
    optimal_idx,
    feature_threshold,
    prob_threshold,
    AUC,
    dataset_name,
    strain,
    features="",
):
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 4)
    )  # Creates a figure with two side-by-side subplots

    naive_classification_precision = data["y"].mean()

    # Plot PRC on the first subplot
    axs[0].plot(
        recall, precision, label=f"Precision-Recall curve (area = {AUC : 0.2f})", color="#9b59b6"
    )
    axs[0].axhline(y=naive_classification_precision, color="black", linestyle="--")
    axs[0].plot(recall[optimal_idx], precision[optimal_idx], marker="o", markersize=5, color="red")
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].set_title("Precision-Recall curve")
    axs[0].legend(loc="lower right")

    custom_palette = {"Non-Responders": "orange", "Responders": "#3498db"}

    if len(features) == 1:
        col_name = features[0]
        # Plot sorted feature values vs Index on the second subplot
        sorted_data = data.sort_values(col_name, ignore_index=True).reset_index()
        sns.scatterplot(
            ax=axs[1],
            data=sorted_data,
            x="index",
            y=col_name,
            hue="Label text",
            palette=custom_palette,
        )
        axs[1].axhline(y=feature_threshold, color="black", linestyle="--")
        axs[1].set_title(f"Sorted {col_name} vs Index")
    else:  # len(features) > 1
        predicted_true = data.loc[data[prob_column] >= prob_threshold]

        sns.scatterplot(
            ax=axs[1],
            data=predicted_true,
            x=features[0],
            y=features[1],
            marker="x",
            s=100,
            color="red",
        )

        sns.scatterplot(
            ax=axs[1],
            data=data,
            x=features[0],
            y=features[1],
            hue="Label text",
            palette=custom_palette,
        )
        axs[1].set_title(f"IMMAGE and Age, X=predicted non-responder")

    fig.suptitle(f"Probability-based threshold with PRC\n{dataset_name} {strain}")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area.
    plt.show()

def calc_and_plot_threshold_ROC(
    data,
    classifier,
    prob_column,
    dataset_name,
    strain,
    bPlotThreshold,
    features=[],
):


    fpr, tpr, thresholds = roc_curve(data["y"], data[prob_column])
    AUC = auc(fpr, tpr)
    intercept = classifier.intercept_[0]
    slope = classifier.coef_[0][0]

    # Identifying the optimal threshold (using Youden’s Index)
    optimal_idx = np.argmax(tpr - fpr)
    prob_threshold = thresholds[optimal_idx]

    # Calculate the cutoff value
    feature_threshold = get_threshold_from_probability(
        prob_threshold, intercept=intercept, slope=slope
    )

    score = np.max(tpr-fpr)

    # Calculate the cutoff value
    feature_threshold = get_threshold_from_probability(
        prob_threshold, intercept=intercept, slope=slope
    )

    if bPlotThreshold:
        plot_desicion_threshold_ROC(
            data,
            fpr,
            tpr,
            prob_column,
            optimal_idx,
            feature_threshold,
            prob_threshold,
            AUC,
            dataset_name,
            strain,
            features=features,
        )

    return (score, prob_threshold, feature_threshold, AUC)

def calc_and_plot_threshold_PRC(
    data,
    classifier,
    precision,
    recall,
    thresholds,
    prob_column,
    dataset_name,
    strain,
    bPlotThreshold,
    features=[],
):
    precision, recall, thresholds = precision_recall_curve(data["y"], data[prob_column])
    AUC = auc(recall, precision)
    intercept = classifier.intercept_[0]
    slope = classifier.coef_[0][0]

    naive_classification_precision = data["y"].mean()

    # Identifying the optimal threshold (maximal F1 score)
    beta = 0.7
    a = (1 + pow(beta, 2)) * (precision * recall)
    b = pow(beta, 2) * precision + recall
    b = np.where((b == 0) | np.isnan(b), np.nan, b)
    F_scores = np.divide(a, b)
    optimal_idx = np.nanargmax(F_scores)
    prob_threshold = thresholds[optimal_idx]
    score = F_scores[optimal_idx]

    # Calculate the cutoff value
    feature_threshold = get_threshold_from_probability(
        prob_threshold, intercept=intercept, slope=slope
    )

    if bPlotThreshold:
        plot_desicion_threshold_PRC(
            data,
            precision,
            recall,
            prob_column,
            optimal_idx,
            feature_threshold,
            prob_threshold,
            AUC,
            dataset_name,
            strain,
            features=features,
        )

    return (score, prob_threshold, feature_threshold, AUC)

def get_classifier_stats(data, column, threshold):
    # Global measures (entire dataset)
    optimal_prediction = data[column].apply(lambda x: 1 if x >= threshold else 0)
    data["optimal_pred"] = optimal_prediction
    test_accuracy = accuracy_score(data["y"], optimal_prediction)
    # Performance above the threshold
    y_over_thr = data.loc[data[column] >= threshold, ["y"]]
    non_response_rate_over_thr = y_over_thr.mean().y
    y_under_thr = data.loc[data[column] < threshold, ["y"]]
    non_response_rate_under_thr = y_under_thr.mean().y
    return non_response_rate_over_thr, non_response_rate_under_thr

def preprocess_dataset(dataset, P):
    dataset_name = P["dataset_name"]
    bAdjustMFC = P["bAdjustMFC"]
    strain_index = P["strain_index"]
    day = P["day"]
    day0 = P["day0"]
    strains = P["strains"]
    strain = P["strain"]

    if len(strains) > 1:
        dataset = dataset.loc[dataset[strain_col] == strain].reset_index(drop=True)

    # Discard seroprotected subjects based on HAI > 40 threshold)
    if bDiscardSeroprotected:
        day0_mask = dataset[day_col] == day0
        threshold_mask = dataset[response_col] > HAI_threshold

        # Get a list of all protected patients
        serprotected_subjects = dataset.loc[(day0_mask) & (threshold_mask)][uid_col].unique()
        # keep only patients not in the serprotected_subjects list
        dataset = dataset.loc[~dataset[uid_col].isin(serprotected_subjects)]
        subjects_left = dataset[uid_col].unique()
        # print(f"Discarding {len(serprotected_subjects)} seroprotected subjects")
        # print(f"Subjects left: N={len(subjects_left)}")

    # Pivot the dataset such that different days' samples appear in their own columns, witn NaN where there are missing samples
    t = dataset[[dataset_col, uid_col, age_col, immage_col, accesion_col, day_col, response_col]]
    pivot_t = t.pivot_table(index=uid_col, columns=day_col, values=response_col, aggfunc="first")
    age_t = dataset[["uid", "Age"]].drop_duplicates()

    # Average IMMAGE values across geo_accessions (if they exist) and merge
    immage_t = t.groupby("uid")[immage_col].mean()
    tmp_t = age_t.merge(immage_t, on="uid", how="left").drop_duplicates()
    pivot_t = tmp_t.merge(pivot_t, on="uid", how="left")

    # Reset index to make uid a column again
    pivot_t.reset_index(inplace=True, drop=True)

    # Remove the name of the columns and index name
    pivot_t.columns.name = None
    pivot_t.index.name = None

    # jonathan Currently only used by AdjustMFC branch. TODO: convert the "regular" branch to use it too
    pivot_dataset = pivot_t

    # Use adjusted MFC (HAI) as per John Tsang
    cluster_col = day0
    data = pd.DataFrame()
    if bAdjustMFC:
        # print("Preprocessing dataset, computing adjusted FC")
        metadata = pd.DataFrame(dataset_day_dicts_for_adjFC)
        days = metadata[metadata[dataset_col] == dataset_name]["Days"].iloc[0]
        sampleDay = [x for x in days if "D0" not in x][0]
        day0 = [x for x in days if "D0" in x][0]
        cluster_col = day0

        # Pivot the table to have a column per day
        dataset = pivot_dataset[[uid_col, immage_col, age_col, day0, sampleDay]]
        dataset = dataset.loc[(~pivot_t[day0].isna()) & (~pivot_t[sampleDay].isna())]
        dataset["FC"] = dataset[sampleDay] / dataset[day0]

        # Remove outliers
        mean = dataset[day0].mean()
        std = dataset[day0].std()
        threshold = 3 * std
        dataset = dataset[(dataset[day0] >= mean - threshold) & (dataset[day0] <= mean + threshold)]

        # Bin subjects into 2-3 bins using k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        dataset["Cluster"] = kmeans.fit_predict(dataset[[cluster_col]])

        def normalize(x):
            return (x - x.median()) / x.std()

        # Normalize the FC within each bin to obtain the adjFC
        dataset["adjFC"] = dataset.groupby("Cluster")["FC"].transform(normalize)

        # Take relevant columns only
        data = (
            dataset[[immage_col, "adjFC", age_col, cluster_col, "Cluster"]]
            .rename(columns={"adjFC": response_col})
            .dropna()
        )
        if len(data) == 0:
            print(dataset)
            raise (Exception("Unable to claculate adjFC - day0 sample is zero"))

        # data.groupby("Cluster").count()

    else:  # bAdjustMFC == False
        # If not computing adjMFC, take a specific strain from the given post-vaccine day & assay
        day_mask = dataset[day_col] == day
        dataset = dataset.loc[(day_mask)].reset_index(drop=True)
        dataset = remove_duplicate_accessions(dataset, immage_col, uid_col)

        # Take relevant columns only
        data = dataset[[immage_col, response_col, age_col]]
    # Keep older subjects only, since that's what's actually interesting, and may show IMMAGE's advantage
    if bOlderOnly == True:
        young_subjects = data.loc[data[age_col] < age_threshold]
        data = data.loc[data[age_col] >= age_threshold]
        if len(data) == 0:
            raise (Exception("No subjects over the age of {age_threshold}. Exiting."))
        # print(f"Discarding {len(young_subjects)} seroprotected subjects")
        # print(f"Subjects left: N={len(data)}")

    return data

def analyze_dataset(dataset, P):
    """
    Perform analysis on the dataset based on specified parameters.

    Parameters:
    - dataset (DataFrame): The dataset to be analyzed.
    - P (dict): Dictionary containing analysis parameters.

    Returns:
    - DataFrame summarizing the analysis results.
    """
    dataset_name = P["dataset_name"]
    bAdjustMFC = P["bAdjustMFC"]
    strain_index = P["strain_index"]
    day = P["day"]
    day0 = P["day0"]
    strains = P["strains"]
    strain = P["strain"]
    bPlotOnly = P["bPlotOnly"]
    bPlotThreshold = P["bPlotThreshold"]

    try:
        data = preprocess_dataset(dataset, P)
    except Exception as e:
        summary = pd.DataFrame()
        return summary

    #### Dataset & Strain info
    age_restrict_str = f", Subjects over the age of {age_threshold}" if bOlderOnly else ""
    adjFC_str = ", using adjusted FC" if bAdjustMFC else ""

    md(
        f"### Analysis for dataset: {dataset_name}, strain: {strain}, day: {day}{age_restrict_str}{adjFC_str}"
    )

    data.reset_index(inplace=True, drop=True)

    # Get a boolean map of sub and super threshold values
    low_response_thr = data[[response_col]].quantile(q=0.3).item()

    # Generate labels
    # Note that we define y=1 for all responses <= 30th percentile (and not <)
    # Also note that we defined y=1 as *non* responders, since later on that's what we'll care about detecting
    data["y"] = data[response_col].apply(lambda x: 1 if x <= low_response_thr else 0)

    # Add a text label for plot legends
    data["Label text"] = data["y"].apply(lambda x: "Responders" if x == 0 else "Non-Responders")

    # For an overview of all datasets, we may want to plot here and skip the rest of the analysis
    if bPlotOnly:
        plot_response(data, dataset_name, strain, features=[immage_col, age_col])
        return

    # Classifying with logistic regression - fit on the entire dataset
    log_regress_immage = LogisticRegression()
    log_regress_age = LogisticRegression()
    log_regress_combined = LogisticRegression()

    # Train a classifier based on immage and on age for comparison
    log_regress_immage.fit(data[[immage_col]], data["y"])
    log_regress_age.fit(data[[age_col]], data["y"])
    log_regress_combined.fit(data[[immage_col, age_col]], data["y"])

    non_responder_col = "p_non_responder"
    non_responder_col_age = "p_non_responder_age"
    non_responder_col_combined = "p_non_responder_combined"

    proba = pd.DataFrame(log_regress_immage.predict_proba(data[[immage_col]]))
    data[non_responder_col] = proba[1]
    proba = pd.DataFrame(log_regress_age.predict_proba(data[[age_col]]))
    data[non_responder_col_age] = proba[1]
    proba = pd.DataFrame(log_regress_combined.predict_proba(data[[immage_col, age_col]]))
    data[non_responder_col_combined] = proba[1]

    # #### Thresholding based on logistic regression probabilties
    # #### IMMAGE-based classification
    # Run for immage and age to compare
    # IMMAGE
    immage_score, threshold, immage_threshold, immage_auc = calc_and_plot_threshold_ROC(
        data,
        log_regress_immage,
        non_responder_col,
        dataset_name,
        strain,
        bPlotThreshold,
        features=[immage_col],
    )
    non_response_rate_over_thr, non_response_rate_under_thr = get_classifier_stats(
        # data, non_responder_col, threshold
        data,
        immage_col,
        immage_threshold,
    )

    # #### Age-based classification
    # Age
    age_score, prob_threshold_age, age_threshold, age_auc = calc_and_plot_threshold_ROC(
        data,
        log_regress_age,
        non_responder_col_age,
        dataset_name,
        strain,
        bPlotThreshold,
        features=[age_col],
    )
    age_non_response_rate_over_thr, age_non_response_rate_under_thr = get_classifier_stats(
        # data, non_responder_col_age, prob_threshold_age
        data,
        age_col,
        age_threshold,
    )

    # #### Age & IMMAGE combined
    # Combined
    combined_score, prob_threshold_combined, _, combined_auc = calc_and_plot_threshold_ROC(
        data,
        log_regress_combined,
        non_responder_col_combined,
        dataset_name,
        strain,
        bPlotThreshold,
        features=[immage_col, age_col],
    )
    combined_non_response_rate_over_thr, combined_non_response_rate_under_thr = (
        get_classifier_stats(data, non_responder_col_combined, prob_threshold_combined)
    )

    # #### Comparison of using the different features
    summary_dict = {
        ("F score", "IMMAGE"): [immage_score],
        ("F score", "Age"): [age_score],
        ("F score", "Multivariate"): [combined_score],
        ("NR rate over threshold", "IMMAGE"): [non_response_rate_over_thr],
        ("NR rate over threshold", "Age"): [age_non_response_rate_over_thr],
        ("NR rate over threshold", "Multivariate"): [combined_non_response_rate_over_thr],
        ("NR rate under threshold", "IMMAGE"): [non_response_rate_under_thr],
        ("NR rate under threshold", "Age"): [age_non_response_rate_under_thr],
        ("NR rate under threshold", "Multivariate"): [combined_non_response_rate_under_thr],
    }

    # Create a MultiIndex
    multi_index = pd.MultiIndex.from_product(
        [
            ["F score", "NR rate over threshold", "NR rate under threshold"],
            ["IMMAGE", "Age", "Multivariate"],
        ]
    )

    # Create the DataFrame
    summary = pd.DataFrame(summary_dict, columns=multi_index)
    summary["Composite", "IMMAGE"] = summary[
        [("F score", "IMMAGE"), ("NR rate over threshold", "IMMAGE")]
    ].mean(axis=1)
    summary["Composite", "Age"] = summary[
        [("F score", "Age"), ("NR rate over threshold", "Age")]
    ].mean(axis=1)
    summary["Composite", "Multivariate"] = summary[
        [("F score", "Multivariate"), ("NR rate over threshold", "Multivariate")]
    ].mean(axis=1)
    # print(summary.to_string(index=False))

    summary["Max difference"] = summary.apply(
        lambda row: max(
            row["Composite", "IMMAGE"] - row["Composite", "Age"],
            row["Composite", "Multivariate"] - row["Composite", "Age"],
        ),
        axis=1,
    )

    return summary


def get_strains(dataset, day):
    strains = dataset.loc[dataset[day_col] == day][strain_col].unique()
    if len(strains) > 1:
        # "Influenza" denotes an MFC calculation in the original dataset
        strains = list(set(strains) - set(["Influenza"]))
        # Sort to maintain consistency (sorts in-place)
        strains.sort()
    elif len(strains) < 1:
        strains = ["single strain - missing data"]

    return strains


# %%
def analyze_all_datasets(datasets, metadata, bPlotOnly=False, bPlotThreshold=False):
    accumulated_results = pd.DataFrame()
    if bPlotOnly:
        bAdjustMFC_list = [False]
    else:
        bAdjustMFC_list = [True, False]
    for dataset_name in metadata[dataset_col].unique():
        curr_metadata = metadata.loc[metadata[dataset_col] == dataset_name]
        dataset = datasets.loc[datasets[dataset_col] == dataset_name]
        day = [x for x in curr_metadata["Days"].iloc[0] if "D0" not in x][0]
        strains = get_strains(dataset, day)
        for strain_index in range(len(strains)):
            strain_name = strains[strain_index].replace("/", "_").replace(" ", "_")
            # print(f'exporting {dataset_name}, strain no. {strain_index}: {strain_name}, day: {day}')
            # Define parameters for curr_metadata and strain
            day0_list = [x for x in curr_metadata["Days"].iloc[0] if "D0" in x]
            if len(day0_list) < 1:
                break
            day0 = day0_list[0]
            for bAdjustMFC in bAdjustMFC_list:
                P = {
                    "bAdjustMFC": bAdjustMFC,
                    "dataset_name": dataset_name,
                    "strain_index": strain_index,
                    "strain": strains[strain_index],
                    "strains": strains,
                    "day": day,
                    "day0": day0,
                    "bPlotOnly": bPlotOnly,
                    "bPlotThreshold": bPlotThreshold,
                }
                try:
                    tmp_dict = {
                        dataset_col: dataset_name,
                        strain_col: strain_name,
                        strain_index_col: strain_index,
                        day_col: day,
                        "bAdjustMFC": bAdjustMFC,
                    }
                    if bPlotOnly:
                        analyze_dataset(dataset, P)
                    else:
                        # initialize the row for this database with some metadata
                        res = analyze_dataset(dataset, P)
                        if len(res) > 0:
                            row = pd.DataFrame([tmp_dict])
                            # Concat along lines, adding new columns
                            row = pd.concat([row, res], axis=1)
                            # Concat along columns, adding new lines
                            accumulated_results = pd.concat(
                                [accumulated_results, row], ignore_index=True
                            )
                except Exception as e:
                    print(f"Caught exception when runnnig {dataset_name}")
                    raise (e)

    return accumulated_results


# %%
def load_data():
    # Read in Data and drop missing values
    data_dir = get_dir_by_name("data")
    df = pd.read_csv(os.path.join(data_dir, "../data/all_vaccines.csv"))
    datasets = df.dropna(subset=[immage_col, age_col, dataset_col, uid_col, day_col, response_col])
    dataset_names = datasets[dataset_col].unique()
    # Get the info for all influenza datasets, excluding some.
    influenza_df = pd.DataFrame(influenza_dicts)
    all_sets_df = pd.DataFrame(dataset_day_dicts_for_adjFC)

    # We can choose to analyze all datasets, only influenza ones and only non-influenza ones
    if bInfluenza:
        # Get the info for all influenza datasets, excluding some.
        metadata = influenza_df
        dataset_names = metadata[dataset_col].unique().astype(str)
        dataset_names = list(set(dataset_names) - set(exclude_datasets))
    elif bNonInfluenza:
        metadata = all_sets_df
        dataset_names = all_sets_df[dataset_col].unique().astype(str)
        dataset_names = list(
            set(dataset_names) - set(influenza_df[dataset_col]) - set(exclude_datasets)
        )
    else:
        metadata = all_sets_df
        dataset_names = all_sets_df[dataset_col].unique().astype(str)
        dataset_names = list(set(dataset_names) - set(exclude_datasets))

    # dataset_names = ["GSE48023.SDY1276"]
    # Filter datasets and metadata according to the list of datasets we want to look at.
    datasets = datasets.loc[datasets["Dataset"].isin(dataset_names)]
    metadata = metadata.loc[metadata["Dataset"].isin(dataset_names)]
    return datasets, metadata


def debug_single_dataset(datasets, metadata, dataset_name="GSE48023.SDY1276", strain_index=0):
    # Narrow to a specific datset
    # Filter data
    name_mask = datasets[dataset_col] == dataset_name
    dataset = datasets.loc[name_mask].reset_index(drop=True)

    # Filter metadata
    name_mask = metadata[dataset_col] == dataset_name
    metadata = metadata.loc[name_mask].reset_index(drop=True)

    day = [x for x in metadata["Days"].iloc[0] if "D0" not in x][0]
    day0 = [x for x in metadata["Days"].iloc[0] if "D0" in x][0]

    strains = get_strains(dataset, day)

    P = {
        "bAdjustMFC": bAdjustMFC,
        "dataset_name": dataset_name,
        "strain_index": strain_index,
        "day": day,
        "day0": day0,
        "strain": strains[strain_index],
        "strains": strains,
        "bPlotOnly": False,
        "bPlotThreshold": False,
    }
    return analyze_dataset(dataset, P)


def run_single_dataset(datasets, metadata, params):
    # Narrow to a specific datset
    # Filter data
    name_mask = datasets[dataset_col] == params["dataset_name"]
    dataset = datasets.loc[name_mask].reset_index(drop=True)

    # Filter metadata
    name_mask = metadata[dataset_col] == params["dataset_name"]
    metadata = metadata.loc[name_mask].reset_index(drop=True)

    day0 = [x for x in metadata["Days"].iloc[0] if "D0" in x][0]

    strains = get_strains(dataset, params["day"])

    P = {
        "bAdjustMFC": params["bAdjustMFC"],
        "dataset_name": params["dataset_name"],
        "strain_index": params["strain_index"],
        "day": params["day"],
        "day0": day0,
        "strain": strains[params["strain_index"]],
        "strains": strains,
        "bPlotOnly": False,
        "bPlotThreshold": True,
    }
    return analyze_dataset(dataset, P)


def save_and_show_plot(filename, dpi=300):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.show()
