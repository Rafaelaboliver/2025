#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:18:32 2025

@author: rafaela
"""

import numpy as np
import pandas as pd
import hdbscan
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

def run_dbscan_clustering(df, scaled_data, eps=0.06993987975951904, min_samples=10, metric='manhattan'):
    """
    Applies DBSCAN to scaled data and adds cluster labels to the original DataFrame.
    """
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    df['cluster'] = clusterer.fit_predict(scaled_data)
    
    n_outliers = (df['cluster'] == -1).sum()
    n_clusters = len(np.unique(df['cluster'][df['cluster'] != -1]))

    print(f"DBSCAN Outliers detected: {n_outliers}")
    print(f"DBSCAN Valid clusters (excluding outliers): {n_clusters}")
    
    return df

def optimize_hdbscan_parameters(scaled_data, min_cluster_sizes, min_samples_values=None, metrics=['euclidean', 'manhattan']):
    best_score = -1
    best_config = None
    test_results = []

    for metric in metrics:  # Agora testando todas as métricas que você passar
        for min_cluster_size in min_cluster_sizes:
            for min_samples in (min_samples_values if min_samples_values is not None else [None]):
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=metric
                    )
                    labels = clusterer.fit_predict(scaled_data)

                    # Ignorar se formou 1 cluster só ou tudo outlier
                    if len(np.unique(labels[labels != -1])) <= 1:
                        continue

                    valid_mask = labels != -1
                    silhouette_avg = silhouette_score(scaled_data[valid_mask], labels[valid_mask])
                    calinski_score = calinski_harabasz_score(scaled_data[valid_mask], labels[valid_mask])

                    test_results.append((
                        metric, min_cluster_size, min_samples,
                        np.sum(labels == -1), len(np.unique(labels[labels != -1])),
                        silhouette_avg, calinski_score
                    ))

                    # Atualiza melhor configuração
                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_config = {
                            'metric': metric,
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'silhouette': silhouette_avg,
                            'calinski_harabasz': calinski_score
                            }

                except Exception as e:
                    # Se uma combinação der erro, continua
                    print(f"Skipping configuration: metric={metric}, min_cluster_size={min_cluster_size}, min_samples={min_samples} due to error: {e}")
                    continue

    df_results = pd.DataFrame(
        test_results,
        columns=[
            'metric', 'min_cluster_size', 'min_samples',
            'outliers', 'clusters', 'silhouette', 'calinski_harabasz'
            ]
        )
    return best_config, df_results

def run_hdbscan_clustering(df, scaled_data, min_cluster_size=16, min_samples=19, metric='manhattan', cluster_selection_method='eom'):
    """
    Perform HDBSCAN clustering on the scaled dataset.

    Parameters:
    - clustering_df (pd.DataFrame): Original clustering dataset (metadata).
    - scaled_data (np.ndarray): Normalized feature array.
    - min_cluster_size (int): Minimum size of clusters.
    - min_samples (int or None): Minimum samples per cluster; if None, defaults automatically.

    Returns:
    - pd.DataFrame: clustering_df with a new 'cluster_hdbscan' column.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method
    )
    labels = clusterer.fit_predict(scaled_data)
    
    df = df.copy()
    df['hdbscan_cluster'] = labels
    
    n_outliers = (labels == -1).sum()
    n_clusters = len(np.unique(labels[labels != -1]))

    print(f"HDBSCAN Outliers detected: {n_outliers}")
    print(f"HDBSCAN Valid clusters (excluding outliers): {n_clusters}")
    
    return df