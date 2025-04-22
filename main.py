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
from clustering import run_dbscan_clustering
from visualization import (
    plot_clusters_2d,
    plot_clusters_3d,
    generate_cluster_map
)

# ===========================================
# PARAMETERS AND PATHS
# ===========================================
DATA_PATH = "/home/rafaela/internship/2025/raw_data/vertical/"
FILE_NAME = "EGMS_L3_E37N23_100km_U_2019_2023_1.csv"
FILE_PATH = os.path.join(DATA_PATH, FILE_NAME)

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
# 4. CLUSTERING PIPELINE
# ===========================================
filtered_metadata = {k: metadata_dict[k] for k in velocities_negative.index}

clustering_df = pd.DataFrame({
    'pixel': list(filtered_metadata.keys()),
    'latitude': [filtered_metadata[k]['latitude'] for k in filtered_metadata],
    'longitude': [filtered_metadata[k]['longitude'] for k in filtered_metadata],
    'velocity': [velocities_negative.loc[k, 'detrended_velocity'] for k in filtered_metadata]
})
clustering_df.set_index('pixel', inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(clustering_df[['latitude', 'longitude', 'velocity']])

clustering_df = run_dbscan_clustering(clustering_df, scaled_data)

# ===========================================
# 5. VISUALIZATION
# ===========================================
plot_clusters_2d(clustering_df)
plot_clusters_3d(clustering_df)
generate_cluster_map(clustering_df)

print("Process completed.")