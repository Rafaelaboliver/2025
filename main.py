#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:23:00 2025

@author: rafaela
"""

# ===========================================
# IMPORTS
# ===========================================
import os
import pyproj
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('/home/rafaela/internship/2025/')  # Add the correct directory
from functions import (
    load_data,
    convert_coordinates_pyproj,
    filter_by_polygon,
    transform_time_series,
    linear_detrend,
    process_rolling_mean
)
from clustering import ( 
    run_dbscan_clustering,
    optimize_dbscan_parameters,
    optimize_hdbscan_parameters,
    run_hdbscan_clustering
    )
from visualization import (
    plot_clusters_2d,
    plot_clusters_3d,
    generate_cluster_contour_map,
    generate_cluster_interactive_map,
    plot_top10_comparison,
    generate_combined_map,
    plot_model_comparison_v2
)

# ===========================================
# PARAMETERS AND PATHS
# ===========================================
DATA_PATH = "/home/rafaela/internship/2025/raw_data/vertical/"
FILE_NAME = "EGMS_L3_E37N23_100km_U_2019_2023_1.csv"
FILE_PATH = os.path.join(DATA_PATH, FILE_NAME)
RESULTS_PATH = "/home/rafaela/internship/2025/results/plots/vertical/"


SOURCE_EPSG = pyproj.CRS("EPSG:3035")  # Spatial reference system (example)
TARGET_EPSG = pyproj.CRS("EPSG:4326")  # Target CRS (WGS 84 for lat/lon) - use 23032 to UTM32

# Study area bounds (manual filtering)
LAT_MIN = 43.670
LAT_MAX = 43.700
LON_MIN = 3.410
LON_MAX = 3.480



# ===========================================
# 1. LOAD AND PREPARE DATA
# ===========================================
raw_df = load_data(FILE_PATH)
filtered_df = raw_df.iloc[:, [1,2,3,5] + list(range(11, raw_df.shape[1]))]
converted_df = convert_coordinates_pyproj(filtered_df, source_epsg=SOURCE_EPSG, target_epsg=TARGET_EPSG)
filtered_area_df = filter_by_polygon(converted_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# ===========================================
# 2. TIME SERIES PROCESSING
# ===========================================
pixels_dict, metadata_dict = transform_time_series(filtered_area_df)

detrended_pixels_dict = {}
for pixel_id, pixel_data in pixels_dict.items():
    if pixel_data.empty:
        continue
    col = pixel_data.columns[0]
    detrended_pixels_dict[pixel_id] = linear_detrend(col, pixel_data[col])

rolling_mean_reduced, overall_mean, rolling_means_dict = process_rolling_mean(
    pixels_dict=detrended_pixels_dict,
    window_size=2
)

# ===========================================
# 3. VELOCITY CORRECTION
# ===========================================
velocities_df = filtered_area_df[['mean_velocity']].copy()
velocities_avg = velocities_df['mean_velocity'].mean()
velocities_detrended = velocities_df['mean_velocity'] - velocities_avg
velocities_detrended_df = pd.DataFrame({'detrended_velocity': velocities_detrended})
velocities_negative = velocities_detrended_df[velocities_detrended_df['detrended_velocity'] < 0]

# ===========================================
# 4. BUILD THE CLUSTERING DATASET
# ===========================================
filtered_metadata = {k: metadata_dict[k] for k in velocities_negative.index}

clustering_df = pd.DataFrame({
    'pixel': list(filtered_metadata.keys()),
    'latitude': [filtered_metadata[k]['latitude'] for k in filtered_metadata],
    'longitude': [filtered_metadata[k]['longitude'] for k in filtered_metadata],
    'velocity': [velocities_negative.loc[k, 'detrended_velocity'] for k in filtered_metadata]
})
clustering_df.set_index('pixel', inplace=True)

# ===========================================
# 5. NORMALIZE THE DATA
# ===========================================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(clustering_df[['latitude', 'longitude', 'velocity']])

# ===========================================
# 6. OPTMIZE PARAMETERS 
# ===========================================

#DBSCAN
# Define parameter ranges
eps_values = np.linspace(0.05, 10.0, num=500) # Generate epsilon values automatically between 0.05 and 10
min_samples_values = np.arange(5, 21) # Generate min_samples automatically between 5 and 20
metrics = ['euclidean', 'manhattan']

# Optimize
best_dbscan_config, dbscan_df_results = optimize_dbscan_parameters(
    scaled_data=scaled_data,
    eps_range=eps_values,
    min_samples_values=min_samples_values,
    metrics=metrics
)

print("Best configuration found:", best_dbscan_config)

#HBDBSCAN
# Define parameter ranges
min_cluster_sizes = np.arange(5, 30)
min_samples_values_hdb = np.arange(5, 20)

best_hdbscan_config, hdbscan_df_results = optimize_hdbscan_parameters(
    scaled_data=scaled_data,
    min_cluster_sizes=min_cluster_sizes,
    min_samples_values=min_samples_values_hdb,
    metrics=metrics
)

print("Best configuration found:", best_hdbscan_config)

# ===========================================
# 7. CLUSTERING PIPELINE
# ===========================================

clustering_dbscan_df = run_dbscan_clustering(clustering_df, scaled_data)
clustering_hdbscan_df = run_hdbscan_clustering(clustering_df, scaled_data)

# 7.1 Extract metrics
metrics_dbscan = {
    'silhouette': best_dbscan_config['silhouette'],
    'calinski_harabasz': best_dbscan_config['calinski_harabasz'],
    'outliers': (clustering_dbscan_df['cluster'] == -1).sum(),
    'clusters': len(clustering_dbscan_df['cluster'].unique()) - (1 if -1 in clustering_dbscan_df['cluster'].unique() else 0)
}

metrics_hdbscan = {
    'silhouette': best_hdbscan_config['silhouette'],
    'calinski_harabasz': best_hdbscan_config['calinski_harabasz'],
    'outliers': (clustering_hdbscan_df['hdbscan_cluster'] == -1).sum(),
    'clusters': len(clustering_hdbscan_df['hdbscan_cluster'].unique()) - (1 if -1 in clustering_hdbscan_df['hdbscan_cluster'].unique() else 0)
}

# ===========================================
# 8. VISUALIZATION
# ===========================================

#DBSCAN
plot_clusters_2d(clustering_dbscan_df, cluster_column='cluster', prefix='dbscan_', save_path=RESULTS_PATH)
plot_clusters_3d(clustering_dbscan_df, cluster_column='cluster', prefix='dbscan_', save_path=RESULTS_PATH)
generate_cluster_contour_map(clustering_dbscan_df, save_path=RESULTS_PATH, prefix="dbscan_")
generate_cluster_interactive_map(clustering_dbscan_df, cluster_column='cluster', prefix='dbscan_', save_path=RESULTS_PATH)
plot_top10_comparison(dbscan_df_results, save_path=RESULTS_PATH, prefix="dbscan_")
generate_combined_map(clustering_dbscan_df, cluster_column='cluster', prefix='dbscan_', save_path=RESULTS_PATH)

#HDBSCAN
plot_clusters_2d(clustering_hdbscan_df, cluster_column='hdbscan_cluster', prefix='hdbscan_', save_path=RESULTS_PATH)
plot_clusters_3d(clustering_hdbscan_df, cluster_column='hdbscan_cluster', prefix='hdbscan_', save_path=RESULTS_PATH)
generate_cluster_contour_map(clustering_hdbscan_df, save_path=RESULTS_PATH, prefix="hdbscan_")
generate_cluster_interactive_map(clustering_hdbscan_df, cluster_column='hdbscan_cluster', prefix='hdbscan_', save_path=RESULTS_PATH)
plot_top10_comparison(hdbscan_df_results, save_path=RESULTS_PATH, prefix="hdbscan_")
generate_combined_map(clustering_hdbscan_df, cluster_column='hdbscan_cluster', prefix='hdbscan_', save_path=RESULTS_PATH)

#COMPARISON
plot_model_comparison_v2(metrics_dbscan, metrics_hdbscan, save_path='model_comparison.png')


print("Process completed.")