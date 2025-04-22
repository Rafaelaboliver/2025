#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:18:32 2025

@author: rafaela
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def run_dbscan_clustering(df, scaled_data, eps=0.09, min_samples=12, metric='euclidean'):
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
