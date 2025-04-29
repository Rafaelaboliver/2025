#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:19:24 2025

@author: rafaela
"""

import os
import folium
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from folium.raster_layers import ImageOverlay


def plot_clusters_2d(df, cluster_column='cluster', save_path=None, prefix=""):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df[cluster_column], cmap='viridis', s=25)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{prefix.upper()} Clusters (2D)")
    plt.grid()
    
    if save_path:
        filename = f"{prefix}clusters_2d.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')    
    plt.show()

def plot_clusters_3d(df, cluster_column='cluster', save_path=None, prefix=""):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['longitude'], df['latitude'], df['velocity'], c=df[cluster_column], cmap='viridis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Velocity')
    plt.title(f"{prefix.strip('_').upper()} Clusters - 3D View")

    if save_path:
        filename = f"{prefix}clusters_3d.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()

def generate_cluster_contour_map(df, save_path=None, prefix=""):
    map_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")

    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 600),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 600)
    )

    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity'],
        (grid_x, grid_y),
        method='cubic'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity'].min(), df['velocity'].max(), 20)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="Vertical Displacement (mm/year)")
    ax.axis('off')

    temp_path = os.path.join(tempfile.gettempdir(), f"{prefix}contour_map.png")
    plt.savefig(temp_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    overlay = ImageOverlay(
        image=temp_path,
        bounds=[[df['latitude'].min(), df['longitude'].min()],
                [df['latitude'].max(), df['longitude'].max()]],
        opacity=0.65
    )
    overlay.add_to(map_clusters)

    if save_path:
        filename = f"{prefix}clusters_contour_map.html"
        map_clusters.save(os.path.join(save_path, filename))

    print(f"✔️ Contour map saved: {filename}")
    
def generate_cluster_interactive_map(df, cluster_column='cluster', save_path=None, prefix=""):
    map_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")

    unique_clusters = df[cluster_column].unique()
    cmap = cm.get_cmap('tab10', len(unique_clusters))
    colors_dict = {cluster: mcolors.rgb2hex(cmap(i % 10)[:3]) for i, cluster in enumerate(unique_clusters)}

    for _, row in df.iterrows():
        cluster_val = row[cluster_column]
        color = 'grey' if cluster_val == -1 else colors_dict[cluster_val]
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(map_clusters)

    if save_path:
        filename = f"{prefix}clusters_interactive_map.html"
        map_clusters.save(os.path.join(save_path, filename))
    print(f"✔️ Interactive map saved: {filename}")

def plot_top10_comparison(df_results, save_path=None, prefix=""):
    """
    Plot a grouped bar chart comparing Silhouette, Davies-Bouldin, and Calinski-Harabasz for top 10 configurations,
    using two y-axes and saving the figure if a path is provided.
    """

    top10 = df_results.sort_values(by='silhouette', ascending=False).head(10)

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # First eixo Y
    ax1.bar(top10.index - 0.25, top10['silhouette'], width=0.35, label='Silhouette', color='#003f5c')
    ax1.bar(top10.index, top10['davies_bouldin'], width=0.35, label='Davies-Bouldin', color='#2f4b7c')
    ax1.set_ylabel('Silhouette / Davies-Bouldin', fontsize=14)
    ax1.set_xlabel('Configuration Index', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Second eixo Y
    ax2 = ax1.twinx()
    ax2.bar(top10.index + 0.25, top10['calinski_harabasz'], width=0.35, label='Calinski-Harabasz', color='#ffa600')
    ax2.set_ylabel('Calinski-Harabasz', fontsize=14)

    # Title
    plt.title(f'Top 10 Configurations - {prefix.strip("_").upper()}', fontsize=16)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=12)

    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(os.path.join(save_path, f"{prefix}top10_comparison.png"), dpi=300, bbox_inches='tight')
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
        df['velocity'],
        (grid_x, grid_y),
        method='cubic'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity'].min(), df['velocity'].max(), 20)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.7)
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

