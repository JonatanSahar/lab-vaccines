
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
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from math import log

# Constants for this project
from constants import *

# Change the current working directory
os.chdir("/home/yonatan/Documents/projects/vaccines/code")

# Define auxilary functions

def get_data_dir():
    # Define the starting directory
    current_dir = os.getcwd()

    # Traverse up the directory tree until we find a directory named "data"
    while current_dir != "/":
        if "data" in os.listdir(current_dir):
            data_dir = os.path.join(current_dir, "data")
            return data_dir
        current_dir = os.path.dirname(current_dir)
    else:
        print("Directory 'data' not found in the parent directories.")
        raise (Exception())


def remove_duplicate_accessions(dataset, immage_col, uid_col):
    '''Sometimes there are multiple geo_accession numbers, like in GSE48018.SDY1276.
    Average the IMMAGE, since all else is the same'''
    first_uid = dataset.iloc[0][uid_col]
    accessions = dataset[dataset[uid_col] == first_uid]["geo_accession"].unique()
    if len(accessions) > 1:
        # print(f"Multiple accession detected, Collapsing by averaging on IMMAGE value")
        dataset = dataset.groupby(uid_col, as_index=False).agg({immage_col: "mean", **{col: "first" for col in dataset.columns if col not in [uid_col, immage_col]},})

    accessions = dataset[dataset[uid_col] == first_uid]["geo_accession"].unique()
    assert len(accessions) == 1

    return dataset

def get_threshold_from_probability(prob, intercept, slope):
    return -1 * (log(1 / prob - 1) + intercept) / slope

def calc_and_plot_prob_threshold(data, classifier, precision, recall, thresholds, prob_column, features=""):
    AUC = auc(recall, precision)
    intercept = classifier.intercept_[0]
    slope = classifier.coef_[0][0]

    naive_classification_precision = data["y"].mean()

    # Identifying the optimal threshold (maximal F1 score)
    beta = 0.7
    F_scores = (1+pow(beta, 2))*(precision * recall)/(pow(beta, 2)*precision + recall)
    optimal_idx = np.nanargmax(F_scores)
    prob_threshold = thresholds[optimal_idx]
    score = F_scores[optimal_idx]

    # Calculate the cutoff value
    feature_threshold = get_threshold_from_probability(
        prob_threshold, intercept=intercept, slope=slope
    )

    return (score, prob_threshold, feature_threshold, AUC)


def get_classifier_stats_prob(data, prob_column, prob_threshold):
    # Global measures (entire dataset)
    optimal_pred = data[prob_column].apply(lambda x: 1 if x >= prob_threshold else 0)
    test_accuracy = accuracy_score(data["y"], optimal_pred)
    # Performance above the prob_threshold
    y_over_thr = data.loc[data[prob_column] >= prob_threshold, ["y"]]
    non_response_rate_over_thr = y_over_thr.mean().y
    y_under_thr = data.loc[data[prob_column] < prob_threshold, ["y"]]
    non_response_rate_under_thr = y_under_thr.mean().y
    return non_response_rate_over_thr, non_response_rate_under_thr

def analyze_dataset(dataset, P):
    # These parameters are overridden by papermill
    strain_index = ["strain_index"]
    day = ["day"]
    day0  = ["day0"]
    dayMFC = ["dayMFC"]

    bAdjustMFC = False
    bDiscardSeroprotected = True
    bOlderOnly = False

    # Pivot the dataset such that different days' samples appear in their own columns, witn NaN where there are missing samples
    t = dataset[[dataset_col, uid_col, age_col, immage_col, "geo_accession", day_col, response_col]]
    pivot_t = t.pivot_table(index=uid_col, columns=day_col, values=response_col, aggfunc='first')
    age_t = dataset[['uid', 'Age']].drop_duplicates()

    # Average IMMAGE values across geo_accessions (if they exist) and merge
    immage_t = t.groupby('uid')[immage_col].mean()
    tmp_t = age_t.merge(immage_t, on='uid', how='left').drop_duplicates()
    pivot_t = tmp_t.merge(pivot_t, on='uid', how='left')

    # Reset index to make uid a column again
    pivot_t.reset_index(inplace=True, drop=True)

    # Remove the name of the columns and index name
    pivot_t.columns.name = None
    pivot_t.index.name = None

    # TODO complete generating MFC for this dataset
    # if dataset_name == 'SDY296' or dataset_name == 'GSE48023.SDY1276':
    #     dataset = dataset.loc[(~pivot_t[day0].isna()) & (~pivot_t["FC.HAI"].isna())]
    #     pivot_t[dayMFC] = pivot_t[day] / pivot_t[day0]

    # Currently only used by AdjustMFC branch. TODO: convert the "regular" to use it too
    pivot_dataset = pivot_t

    from sklearn.cluster import KMeans
    # Use adjusted MFC (HAI) as per John Tsang
    cluster_col = day0
    if bAdjustMFC:
        exit(1)
        dataset = pivot_dataset[[uid_col, immage_col, age_col, day0, dayMFC]]
        print("Preprocessing dataset, computing adjusted MFC (HAI)")
        dataset = dataset.loc[(~pivot_t[day0].isna()) & (~pivot_t[dayMFC].isna())]

        mean = dataset[day0].mean()
        std = dataset[day0].std()
        threshold = 3 * std
        dataset = dataset[(dataset[day0] >= mean - threshold) & (dataset[day0] <= mean + threshold)]

        # Bin subjects into 2-3 bins using k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        dataset['Cluster'] = kmeans.fit_predict(dataset[[cluster_col]])

        def normalize(x):
            return (x - x.median()) / x.std()

        # Normalize the MFC within each bin to obtain the adjMFC
        dataset['adjMFC'] = dataset.groupby('Cluster')[dayMFC].transform(normalize)

        # Take relevant columns only
        data = dataset[[immage_col, 'adjMFC', age_col, cluster_col, "Cluster"]].rename(columns={'adjMFC': response_col}).dropna()
        # data.groupby("Cluster").count()
        strain = "Influenza"
        strains = "Influenza"


    # Visualize clusters
    if bAdjustMFC:
        custom_palette = {0: "red", 1: "blue", 2: "green"}
        sorted_data = data.sort_values(cluster_col, ignore_index=True).reset_index()
        sns.scatterplot(data=sorted_data, x="index", y=cluster_col, hue="Cluster", palette=custom_palette)
        # plt.axhline(y=threshold, color="black", linestyle="--")
        plt.title(f"{cluster_col} clustered")


    # Discard seroprotected subjects based on HAI > 40 threshold)
    if bDiscardSeroprotected:
        HAI_threshold = 40
        day0_mask = dataset[day_col] == day0
        threshold_mask = dataset[response_col ]> HAI_threshold

        # Get a list of all protected patients
        serprotected_subjects = dataset.loc[(day0_mask) & (threshold_mask)][uid_col].unique()
        # keep only patients not in the serprotected_subjects list
        dataset = dataset.loc[~dataset[uid_col].isin(serprotected_subjects)]
        subjects_left = dataset[uid_col].unique()
        print(f"Discarding {len(serprotected_subjects)} seroprotected subjects")
        print(f"Subjects left: N={len(subjects_left)}")


    if bAdjustMFC == False:
        # If not computing adjMFC, take a specific strain from the given post-vaccine day & assay
        dayMFC_mask = dataset[day_col] == day
        dataset = dataset.loc[(dayMFC_mask)].reset_index(drop=True)

        # Somtimes there are multiple strains - so multiple rows per day
        strains = dataset[strain_col].unique()
        if len(strains) > 1:
            dataset = dataset.loc[dataset[strain_col] == strains[strain_index]].reset_index(drop=True)

        strains_t = dataset[strain_col].unique()
        assert len(strains_t) == 1
        strain = strains_t[0]

        dataset = remove_duplicate_accessions(dataset, immage_col, uid_col)

        # Take relevant columns only
        data = dataset[[immage_col, response_col, age_col]]


    # Keep older subjects only, since that's what's actually more interesting, and may show IMMAGE's advantage
    age_threshlod = 60
    if bOlderOnly == True:
        young_subjects = data.loc[data[age_col] < age_threshlod]
        data = data.loc[data[age_col] >= age_threshlod]
        if len(data) == 0:
            raise(Exception("No subjects over the age of {age_threshlod}. Exiting."))
        print(f"Discarding {len(young_subjects)} seroprotected subjects")
        print(f"Subjects left: N={len(data)}")


    #### Dataset & Strain info
    age_restrict_str = f", Subjects over the age of {age_threshlod}" if bOlderOnly else ""
    day_str = "Adjusted MFC" if bAdjustMFC else f"day: {day}"

    print(f"""### Analysis for dataset: {dataset_name}, strain: {strain}, {day_str}{age_restrict_str}""")

    data.reset_index(inplace=True, drop=True)

    # Get a boolean map of sub and super threshold values
    low_response_thr = data[[response_col]].quantile(q=0.3).item()

    # Generate labels
    # Note that we define y=1 for all responses <= 30th percentile (and not <)
    # Also note that we defined y=1 as *non* responders, since later on that's what we'll care about detecting
    data["y"] = data[response_col].apply(lambda x: 1 if x <= low_response_thr else 0)

    # Add a text label for plot legends
    data["Label text"] = data["y"].apply(lambda x: "Responders" if x == 0 else "Non-Responders")


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

    # data.reset_index(in_place=True, drop=True)

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
    precision, recall, thresholds = precision_recall_curve(data["y"], data[non_responder_col])
    immage_score, prob_threshold, immage_threshold, immage_auc = calc_and_plot_prob_threshold(
        data, log_regress_immage, precision, recall, thresholds, non_responder_col, features=[immage_col]
    )
    non_response_rate_over_thr, non_response_rate_under_thr = get_classifier_stats_prob(
        data, non_responder_col, prob_threshold
    )


    # #### Age-based classification
    # Age
    precision, recall, thresholds = precision_recall_curve(data["y"], data[non_responder_col_age])
    age_score, prob_threshold_age, age_threshold, age_auc = calc_and_plot_prob_threshold(
        data, log_regress_age, precision, recall, thresholds, non_responder_col_age, features=[age_col]
    )
    age_non_response_rate_over_thr, age_non_response_rate_under_thr = get_classifier_stats_prob(
        data, non_responder_col_age, prob_threshold_age
    )


    # #### Age & IMMAGE combined
    # Combined
    precision, recall, thresholds = precision_recall_curve(data["y"], data[non_responder_col_combined])
    combined_score, prob_threshold_combined, _, combined_auc = calc_and_plot_prob_threshold(
        data, log_regress_combined, precision, recall, thresholds, non_responder_col_combined, features=[immage_col, age_col]
    )
    combined_non_response_rate_over_thr, combined_non_response_rate_under_thr = (
        get_classifier_stats_prob(data, non_responder_col_combined, prob_threshold_combined)
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
    multi_index = pd.MultiIndex.from_product([["F score", "NR rate over threshold", "NR rate under threshold"], ["IMMAGE", "Age", "Multivariate"]])

    # Create the DataFrame
    summary = pd.DataFrame(summary_dict, columns=multi_index)
    summary["Composite", "IMMAGE"] = summary[[("F score", "IMMAGE"), ("NR rate over threshold", "IMMAGE")]].mean(axis=1)
    summary["Composite", "Age"] = summary[[("F score", "Age"), ("NR rate over threshold", "Age")]].mean(axis=1)
    summary["Composite", "Multivariate"] = summary[[("F score", "Multivariate"), ("NR rate over threshold", "Multivariate")]].mean(axis=1)
    print(summary.to_string(index=False))

    return summary

def analyze_all_datasets(datasets, metadata, dataset_names):
    for dataset_name in dataset_names:
            dataset_info = metadata.loc[metadata[dataset_col] == dataset_name]
            dataset = datasets.loc[datasets[dataset_col] == dataset_name]
            print(dataset_name)
            days = dataset_info["Days"].iloc[0]
            for day in days:
                    print(day)
                    day_mask = dataset[day_col] == day
                    name_mask = dataset[dataset_col] == dataset_name
                    data = dataset.loc[(name_mask) & (day_mask)].reset_index()
                    strains = data[strain_col].unique()
                    print(strains)
                    for strain_index in range(len(strains)):
                            strain_name = strains[strain_index].replace("/", "_").replace(" ", "_")
                            print(f'exporting {dataset_name}, strain no. {strain_index}: {strain_name}, day: {day}')
                            # Define parameters for dataset_info and strain
                            P = {
                                "dataset_name": dataset_name,
                                "strain_index": strain_index,
                                "day": day
                            }
                            try:
                                analyze_dataset(dataset, P)
                            except:
                                    print (f"******\nCaught exception when runnnig {dataset_name}\n******\n")


def main():
    # Read in Data and drop missing values
    data_dir = get_data_dir()
    datasets = pd.read_csv(os.path.join(data_dir, "../data/all_vaccines.csv"))
    datasets.dropna(inplace=True, subset=[immage_col, age_col, dataset_col, uid_col, day_col, response_col, accesion_col])
    dataset_names = datasets[dataset_col].unique()

    bInfluenza = True
    if bInfluenza:
        # Get the info for all influenza datasets, excluding some.
        metadata = pd.DataFrame(influenza_dicts)
        dataset_names = metadata[dataset_col].unique().astype(str)
        dataset_names = list(set(dataset_names) - set(exclude_datasets))
        datasets = datasets.loc[datasets["Dataset"].isin(dataset_names)]
        print("Working with Influenza datasets only")


    # Narrow to a specific datset
    dataset_name = "GSE41080.SDY212"
    name_mask = datasets[dataset_col] == dataset_name
    dataset = datasets.loc[name_mask].reset_index(drop=True)
    name_mask = metadata[dataset_col] == dataset_name
    metadata = metadata.loc[name_mask].reset_index(drop=True)

    P = {
        "dataset_name": dataset_name,
        "strain_index": 0,
        "day":  metadata["Days"].iloc[0]
    }
    s = analyze_dataset(dataset, P)
    print(s)

if __name__ == "__main__":
    main()
