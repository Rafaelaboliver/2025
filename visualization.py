#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:19:24 2025

@author: rafaela
"""

import os
import math
import folium
import tempfile
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
import matplotlib.patches as patches
from scipy.interpolate import griddata
from folium.raster_layers import ImageOverlay
from sklearn.linear_model import LinearRegression


def plot_clusters_2d(df, cluster_column='cluster', save_path=None, prefix="", 
                     reference_point=(43.7004, 3.4537), reference_label="Approx. Location"):
    """
    Plot a 2D scatter plot of clustering results, excluding outliers, and marking a reference location.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'longitude', 'latitude', and cluster column.
    - cluster_column (str): Name of the column containing cluster labels.
    - save_path (str): Directory to save the image (if provided).
    - prefix (str): Prefix for the output filename.
    - reference_point (tuple): (latitude, longitude) of the reference location to highlight.
    - reference_label (str): Label for the reference marker in the legend.
    """
    # Exclude outliers
    filtered_df = df[df[cluster_column] != -1]

    plt.figure(figsize=(10, 6))
    
    # Plot clusters
    scatter = plt.scatter(filtered_df['longitude'], filtered_df['latitude'],
                          c=filtered_df[cluster_column], cmap='plasma', s=25, label="Clustered Points")
    circle = patches.Circle((reference_point[1], reference_point[0]), 0.015,  # lon, lat, raio em graus
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label=reference_label)
    # Add reference point
    plt.scatter(reference_point[1], reference_point[0],  # lon, lat
                color='red', marker='X', s=100, label=reference_label, zorder=5)
    
    plt.gca().add_patch(circle)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{prefix.upper()} Clusters (2D) — Without Outliers")
    plt.grid()
    plt.legend(loc='upper right')

    if save_path:
        filename = f"{prefix}clusters_2d.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')    
    plt.show()

def plot_clusters_3d(df, cluster_column='cluster', save_path=None, prefix=""):
    # Remove outliers
    filtered_df = df[df[cluster_column] != -1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(filtered_df['longitude'], filtered_df['latitude'], filtered_df['velocity'],
               c=filtered_df[cluster_column], cmap='plasma')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Velocity')
    plt.title(f"{prefix.strip('_').upper()} Clusters - 3D View")

    if save_path:
        filename = f"{prefix}clusters_3d.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()

def plot_top10_comparison(df_results, model_name="DBSCAN", save_path=None, prefix=""):
    """
    Plot a multi-panel comparison of the top 10 configurations for clustering models.
    For DBSCAN (4 metrics): silhouette, davies_bouldin, calinski_harabasz, outliers
    For HDBSCAN (3 metrics): silhouette, calinski_harabasz, outliers

    Parameters:
    - df_results: DataFrame containing results with metrics
    - model_name: "DBSCAN" or "HDBSCAN"
    - save_path: optional path to save figure
    - prefix: optional string to include in title or filename
    """
    top10 = df_results.sort_values(by='silhouette', ascending=False).head(10).reset_index()
    index = top10.index

    metrics_to_plot = ['silhouette', 'calinski_harabasz', 'outliers']
    if 'davies_bouldin' in top10.columns and model_name.upper() == "DBSCAN":
        metrics_to_plot.insert(1, 'davies_bouldin')

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6), sharex=True)

    if num_metrics == 1:
        axes = [axes]  # ensure axes is iterable

    for ax, metric in zip(axes, metrics_to_plot):
        ax.bar(index, top10[metric], color='skyblue')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel("Config Index")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"Top 10 Configurations - {model_name.upper()}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        filename = f"{prefix}top10_comparison_panels_{model_name.lower()}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"✔️ Saved plot to {full_path}")

    plt.show()

def generate_combined_map(df, cluster_column='cluster', save_path=None, prefix="", filename="clusters_combined_map.html"):
    valid_clusters_df = df[df[cluster_column] != -1].copy()

    map_combined = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")

    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 600),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 600)
    )

    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity_detrended'],
        (grid_x, grid_y),
        method='cubic'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity_detrended'].min(), df['velocity_detrended'].max(), 20)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="plasma", alpha=0.7)
    plt.colorbar(contour, label="Vertical Displacement (mm/year)")
    ax.axis('off')

    temp_path = os.path.join(tempfile.gettempdir(), f"{prefix}combined_contour_temp.png")
    plt.savefig(temp_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    overlay = ImageOverlay(
        image=temp_path,
        bounds=[[df['latitude'].min(), df['longitude'].min()],
                [df['latitude'].max(), df['longitude'].max()]],
        opacity=0.65
    )
    overlay.add_to(map_combined)

    unique_clusters = valid_clusters_df[cluster_column].unique()
    cmap = cm.get_cmap('tab10', len(unique_clusters))
    colors_dict = {cluster: mcolors.rgb2hex(cmap(i)[:3]) for i, cluster in enumerate(unique_clusters)}

    for _, row in valid_clusters_df.iterrows():
        color = colors_dict.get(row[cluster_column], 'gray')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(map_combined)

    final_filename = f"{prefix}{filename}"
    output_file = os.path.join(save_path, final_filename)
    map_combined.save(output_file)
    print(f"✔️ Combined map (without outliers) saved: {output_file}")

def plot_model_comparison_v2(dbscan_metrics, hdbscan_metrics, save_path=None):
    """
    Compare DBSCAN and HDBSCAN using silhouette, calinski-harabasz, and outlier count.
    
    Parameters:
    - dbscan_metrics (dict): Dictionary with metrics from DBSCAN.
    - hdbscan_metrics (dict): Dictionary with metrics from HDBSCAN.
    - save_path (str, optional): Path to save the generated image.
    """

    models = ['DBSCAN', 'HDBSCAN']
    silhouette = [dbscan_metrics['silhouette'], hdbscan_metrics['silhouette']]
    calinski = [dbscan_metrics['calinski_harabasz'], hdbscan_metrics['calinski_harabasz']]
    outliers = [dbscan_metrics['outliers'], hdbscan_metrics['outliers']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plots for silhouette and calinski-harabasz
    bars1 = ax1.bar(x - width/2, silhouette, width=width, label='Silhouette', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, calinski, width=width, label='Calinski-Harabasz', color='#ff7f0e')

    ax1.set_ylabel("Score")
    ax1.set_xlabel("Clustering Model")
    ax1.set_title("Model Comparison: DBSCAN vs HDBSCAN")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc="upper left")

    # Secondary axis for outliers
    ax2 = ax1.twinx()
    ax2.plot(x, outliers, 'ro--', label='Outliers', linewidth=2, markersize=7)
    ax2.set_ylabel("Outlier Count")
    ax2.legend(loc="upper right")

    plt.grid(True, linestyle='--', alpha=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_velocity_contour(df, park_lat=43.6990, park_lon=3.4474, save_path=None, prefix=""):
    """
    Generate a contour map based on interpolated vertical velocity and mark the park location.

    Parameters:
    - df (pd.DataFrame): Must contain 'latitude', 'longitude', and 'velocity' columns.
    - park_lat, park_lon (float): Coordinates of the location to highlight.
    - save_path (str or None): Save path for image.
    - prefix (str): Prefix for filename.
    """
    # Interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 600),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 600)
    )
    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity_detrended'],
        (grid_x, grid_y),
        method='cubic'
    )

    # Plotting
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="RdBu_r")
    plt.colorbar(contour, label="Vertical Velocity (mm/year)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{prefix.upper()} Interpolated Velocity Contour Map")

    # Highlight the park
    plt.plot(park_lon, park_lat, marker='x', color='red', markersize=10, label="Approx. Park Location")
    plt.gca().add_patch(plt.Circle((park_lon, park_lat), 0.005, color='red', fill=False, linestyle='--', linewidth=2))
    plt.legend()
    plt.grid(True)

    if save_path:
        filename = f"{prefix}velocity_contour_map.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series_with_amplified_mean_trend(
    data, pixel_ids, metadata_dict, prefix="",
    amplification_factor=15, show_individual=True,
    plot_individual_subplots=False, subplot_grid=(2, 2),
    save_path=None
):

    pixel_ids = [str(pid) for pid in pixel_ids if str(pid) in data.columns]
    if not pixel_ids:
        print("⚠️ No valid pixel IDs found in data columns.")
        return

    # ---------- MAIN PLOT: Mean + Trend ----------
    plt.figure(figsize=(14, 7))
    if show_individual:
        for pid in pixel_ids:
            plt.plot(data.index, data[pid], label=f"Pixel {pid}", linewidth=0.8, alpha=0.6)

    mean_series = data[pixel_ids].mean(axis=1)
    plt.plot(mean_series.index, mean_series.values, label="Mean", color='black', linewidth=2)

    X = np.arange(len(mean_series)).reshape(-1, 1)
    model = LinearRegression().fit(X, mean_series.values)
    trend_amplified = model.intercept_ + (model.coef_[0] * amplification_factor) * X.flatten()
    slope_display = f"{model.coef_[0]:+0.02f}"

    plt.plot(mean_series.index, trend_amplified, 'r--', linewidth=2,
             label=f"Trend (slope = {slope_display} mm/year) ×{amplification_factor}")
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"{prefix.upper()} Vertical Velocity Time Series with Mean and Trend")
    plt.xlabel("Date")
    plt.ylabel("Vertical Velocity (mm/year)")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- SUBPLOTS PER PIXEL ----------
    if plot_individual_subplots:
        rows, cols = subplot_grid
        per_figure = rows * cols
        num_figures = math.ceil(len(pixel_ids) / per_figure)

        for fig_num in range(num_figures):
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
            axes = axes.flatten()

            for i in range(per_figure):
                idx = fig_num * per_figure + i
                if idx >= len(pixel_ids):
                    axes[i].axis('off')
                    continue

                pid = pixel_ids[idx]
                ts = data[pid]
                X = np.arange(len(ts)).reshape(-1, 1)
                y = ts.values

                model = LinearRegression().fit(X, y)
                trend = model.intercept_ + (model.coef_[0] * amplification_factor) * X.flatten()
                slope = f"{model.coef_[0]:+.2f}"

                axes[i].plot(ts.index, y, label=f"Pixel {pid}", alpha=0.8)
                axes[i].plot(ts.index, trend, 'r--', label=f"Trend ×{amplification_factor}\n({slope} mm/yr)")
                axes[i].axhline(0, color='gray', linestyle='--', linewidth=1)
                axes[i].legend()
                axes[i].set_title(f"Pixel {pid}")
                axes[i].set_xlabel("Date")
                axes[i].set_ylabel("Velocity")

            fig.suptitle(f"{prefix.upper()} Individual Time Series with Trends", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

def plot_selected_pixels_with_local_clusters(
    df_clustered, pixel_ids, metadata_dict, park_coords, cluster_column='cluster',
    prefix="", radius_meters=5000, save_path=None
):
    """
    Plots the selected pixels near a park and only shows other pixels in the same clusters.
    
    Parameters:
    - df_clustered: DataFrame with clustered results including latitude, longitude, and cluster labels.
    - pixel_ids: List of selected pixel IDs.
    - metadata_dict: Dictionary containing metadata for each pixel (latitude, longitude).
    - park_coords: Tuple with the approximate (latitude, longitude) of the park.
    - cluster_column: Column name in df_clustered containing cluster labels.
    - prefix: Optional string prefix for the plot title.
    - radius_meters: Radius to draw around the park location.
    - save_path: Path to save the plot (optional).
    """

    pixel_ids = [int(pid) for pid in pixel_ids]  # Ensure correct type
    cluster_ids = df_clustered.loc[pixel_ids, cluster_column].unique()

    # Only show background points from the same clusters
    filtered_background = df_clustered[df_clustered[cluster_column].isin(cluster_ids)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot filtered background points in light gray
    ax.scatter(
        filtered_background['longitude'], filtered_background['latitude'],
        color='lightgray', s=20, label='Clustered Points', alpha=0.5
    )

    # Plot selected pixels
    for pid in pixel_ids:
        coords = metadata_dict[pid]
        ax.scatter(coords['longitude'], coords['latitude'], s=50, label=f"Pixel {pid}")

    # Mark the park location
    ax.scatter(park_coords[1], park_coords[0], color='red', s=100, marker='x', label='Approx. Location')
    #circle = plt.Circle((park_coords[1], park_coords[0]), radius=radius_meters / 111320, color='red', fill=False, linestyle='--')
    #ax.add_patch(circle)

    ax.set_title(f"{prefix.strip('_').upper()}_Selected Pixels")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend(loc='center left', fontsize='small')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_cluster_velocity_distribution(df_region, cluster_column='cluster'):
    """
    Plot histogram of velocities by cluster in the selected region.
    """
    plt.figure(figsize=(10, 6))
    for cluster_id, group in df_region.groupby(cluster_column):
        plt.hist(group['velocity'], bins=30, alpha=0.6, label=f'Cluster {cluster_id}')
    
    plt.title("Velocity Distribution per Cluster (Selected Region)")
    plt.xlabel("Detrended Velocity (mm/year)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
