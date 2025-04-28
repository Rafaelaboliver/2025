#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:18:32 2025

@author: rafaela
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def optimize_dbscan_parameters(scaled_data, eps_range, min_samples_values, metrics):
    """
    Tests multiple combinations of DBSCAN parameters and returns the best configuration based on silhouette score.

    Parameters:
    - scaled_data (np.array): Normalized features for clustering.
    - eps_range (np.array): Array of epsilon values to test.
    - min_samples_values (list): List of min_samples values to test.
    - metrics (list): List of distance metrics to test.

    Returns:
    - dict: Best configuration {'eps', 'min_samples', 'metric', 'silhouette', 'davies_bouldin', 'calinski_harabasz'}
    - pd.DataFrame: Full testing results
    """
    best_config = None
    best_silhouette = -1
    test_results = []

    for eps in eps_range:
        for min_samples in min_samples_values:
            for metric in metrics:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = clusterer.fit_predict(scaled_data)

                num_outliers = np.sum(labels == -1)
                n_clusters = len(np.unique(labels[labels != -1]))

                if n_clusters > 1:
                    valid_clusters = labels[labels != -1]
                    valid_features = scaled_data[labels != -1]

                    silhouette_avg = silhouette_score(valid_features, valid_clusters)
                    db_score = davies_bouldin_score(valid_features, valid_clusters)
                    calinski_score = calinski_harabasz_score(valid_features, valid_clusters)

                    test_results.append((
                        eps, min_samples, metric, num_outliers, n_clusters,
                        silhouette_avg, db_score, calinski_score
                    ))

                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_config = (eps, min_samples, metric, silhouette_avg, db_score, calinski_score)

    df_results = pd.DataFrame(
        test_results,
        columns=[
            'eps', 'min_samples', 'metric', 'outliers', 'clusters',
            'silhouette', 'davies_bouldin', 'calinski_harabasz'
        ]
    )

    if best_config:
        best_result = {
            'eps': best_config[0],
            'min_samples': best_config[1],
            'metric': best_config[2],
            'silhouette': best_config[3],
            'davies_bouldin': best_config[4],
            'calinski_harabasz': best_config[5]
        }
    else:
        best_result = None

    return best_result, df_results


def run_dbscan_clustering(df, scaled_data, eps=0.08987975951903807, min_samples=7, metric='manhattan'):
    """
    Applies DBSCAN to scaled data and adds cluster labels to the original DataFrame.
    """
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    df['cluster'] = clusterer.fit_predict(scaled_data)
    
    n_outliers = (df['cluster'] == -1).sum()
    n_clusters = len(np.unique(df['cluster'][df['cluster'] != -1]))

    print(f"Outliers detected: {n_outliers}")
    print(f"Valid clusters (excluding outliers): {n_clusters}")
    
    return df
