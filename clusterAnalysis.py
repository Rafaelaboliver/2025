#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:26:17 2025

@author: rafaela
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics.pairwise import haversine_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def analyze_clusters(df_clustered, ts_dict, cluster_col='cluster', metadata_dict=None):
    clusters = df_clustered[cluster_col].unique()
    cluster_stats = []
    cluster_series = {}

    # Convert pixel-id-based ts_dict into lat-lon based, if needed
    sample_key = next(iter(ts_dict))
    if isinstance(sample_key, int) and metadata_dict:
        ts_dict = {
            (round(metadata_dict[pid]['latitude'], 6), round(metadata_dict[pid]['longitude'], 6)): ts
            for pid, ts in ts_dict.items()
            if pid in metadata_dict
        }

    for cl in clusters:
        if cl == -1:
            continue  # ignore noise

        subset = df_clustered[df_clustered[cluster_col] == cl]
        coords = list(zip(subset['latitude'], subset['longitude']))

        # Spatial statistics
        center_lat = subset['latitude'].mean()
        center_lon = subset['longitude'].mean()
        vel_mean = subset['velocity_detrended'].mean()
        vel_std = subset['velocity_detrended'].std()
        n_points = len(subset)

        # Time series
        series = [ts_dict.get((lat, lon), []) for lat, lon in coords]
        series = [s for s in series if len(s) > 0]
        if not series:
            continue
        series = np.array(series)
        mean_series = np.mean(series, axis=0)

        # Linear trend
        x = np.arange(len(mean_series))
        slope, intercept, r_value, p_value, std_err = linregress(x, mean_series)

        cluster_series[cl] = {
            'mean_series': mean_series,
            'slope': slope,
            'std_err': std_err
        }

        cluster_stats.append({
            'cluster': cl,
            'n_points': n_points,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'vel_mean': vel_mean,
            'vel_std': vel_std,
            'trend_slope': slope,
            'trend_r2': r_value ** 2
        })

    stats_df = pd.DataFrame(cluster_stats)
    return stats_df, cluster_series

def evaluate_local_cluster_mix(reference_point, df_clustered, ts_dict, radius=100, metadata_dict=None):
    ref_lat, ref_lon = reference_point
    earth_radius = 6371000  # in meters

    sample_key = next(iter(ts_dict))
    if isinstance(sample_key, int) and metadata_dict:
        ts_dict = {
            (round(metadata_dict[pid]['latitude'], 6), round(metadata_dict[pid]['longitude'], 6)): ts
            for pid, ts in ts_dict.items()
            if pid in metadata_dict
        }

    # Convert to radians for haversine
    ref_point_rad = np.radians([[ref_lat, ref_lon]])
    coords_rad = np.radians(df_clustered[['latitude', 'longitude']].values)

    distances = haversine_distances(ref_point_rad, coords_rad)[0] * earth_radius
    df_clustered['distance'] = distances
    local_points = df_clustered[df_clustered['distance'] <= radius].copy()

    cluster_ids = local_points['cluster'].unique()
    if len(cluster_ids) <= 1:
        return {'status': 'homogeneous', 'details': {}}

    details = {}
    for cl in cluster_ids:
        sub = local_points[local_points['cluster'] == cl]
        coords = list(zip(sub['latitude'], sub['longitude']))
        series = [ts_dict.get((latitude, longitude), []) for latitude, longitude in coords]
        series = [s for s in series if len(s) > 0]
        if not series:
            continue
        series = np.array(series)
        mean_series = np.mean(series, axis=0)
        slope, *_ = linregress(np.arange(len(mean_series)), mean_series)

        details[cl] = {
            'n': len(sub),
            'vel_mean': sub['velocity_detrended'].mean(),
            'trend_slope': slope
        }

    return {
        'status': 'mixed',
        'details': details
    }

def get_nearby_cluster_pixels(df_clustered, reference_point, radius=100):
    """
    Returns a DataFrame with points within a given radius (in meters) of a reference point (lat, lon).
    """
    ref_lat, ref_lon = reference_point
    earth_radius = 6371000  # in meters

    ref_rad = np.radians([[ref_lat, ref_lon]])
    coords_rad = np.radians(df_clustered[['latitude', 'longitude']].values)

    distances = haversine_distances(ref_rad, coords_rad)[0] * earth_radius
    df_clustered = df_clustered.copy()
    df_clustered['distance'] = distances

    return df_clustered[df_clustered['distance'] <= radius].copy()

def plot_cluster_series(cluster_series):
    for cl, data in cluster_series.items():
        mean_series = data['mean_series']
        slope = data['slope']

        plt.figure()
        plt.plot(mean_series, label=f'Cluster {cl}')
        plt.title(f'Mean Time Series - Cluster {cl} (Slope={slope:.3f})')
        plt.xlabel('Time (index)')
        plt.ylabel('Cumulative Displacement')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_cluster_map(df_clustered, cluster_col='cluster'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='longitude', y='latitude', hue=cluster_col,
        data=df_clustered[df_clustered[cluster_col] != -1],
        palette='tab20', s=20, linewidth=0, alpha=0.8
    )
    plt.title('Cluster Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()