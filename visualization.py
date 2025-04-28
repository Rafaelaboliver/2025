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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from folium.raster_layers import ImageOverlay


def plot_clusters_2d(df, save_path=None):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis', s=25)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("DBSCAN Clusters (2D)")
    plt.grid()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "clusters_2d.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_clusters_3d(df, save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['longitude'], df['latitude'], df['velocity'], c=df['cluster'], cmap='viridis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Velocity')
    plt.title('Clusters - 3D View')
    
    if save_path:
        plt.savefig(os.path.join(save_path, "clusters_3d.png"), dpi=300, bbox_inches='tight')
    plt.show()

def generate_cluster_contour_map(df, save_path=None, filename="clusters_contour_map.html"):
    map_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")
    
    # Interpolate values using cubic method
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 600),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 600)
    )
    # Interpolate values using cubic method
    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity'],
        (grid_x, grid_y),
        method='cubic'
    )

    # Create figure for contour    
    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity'].min(), df['velocity'].max(), 20)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.7)
    #ax.contour(grid_x, grid_y, grid_z, levels=20, linewidths=0.3,linestyles='solid')
    plt.colorbar(contour, label="Vertical Displacement (mm/year)")
    ax.axis('off')

    # Save temporary PNG
    temp_path = os.path.join(tempfile.gettempdir(), "contour_map.png")
    plt.savefig(temp_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

        # Add to Folium map
    overlay = ImageOverlay(
        image=temp_path,
        bounds=[[df['latitude'].min(), df['longitude'].min()],
                [df['latitude'].max(), df['longitude'].max()]],
        opacity=0.65
    )
    overlay.add_to(map_clusters)
    
    if save_path:
        full_path = os.path.join(save_path, filename)
    else:
        full_path = filename

    map_clusters.save(full_path)
    
    print(f"Map saved: {full_path}")
    
def generate_cluster_interactive_map(df, save_path=None):
    """
    Generate an interactive map with clustered points.

    Parameters:
    - df (DataFrame): Must have 'latitude', 'longitude', and 'cluster' columns.
    - save_path (str): Directory where the HTML will be saved. If None, saves in current directory.
    """
    # Create a map centered at the study area's center
    map_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")

    # Create a color palette for clusters
    unique_clusters = df['cluster'].unique()
    cmap = cm.get_cmap('tab10', len(unique_clusters))
    colors_dict = {cluster: mcolors.rgb2hex(cmap(i % 10)[:3]) for i, cluster in enumerate(unique_clusters)}

    # Add points to the map
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='grey' if row['cluster'] == -1 else colors_dict[row['cluster']],
            fill=True,
            fill_color='grey' if row['cluster'] == -1 else colors_dict[row['cluster']],
            fill_opacity=0.7
        ).add_to(map_clusters)

    # Save the map
    filename = os.path.join(save_path, "clusters_interactive_map.html") if save_path else "clusters_interactive_map.html"
    map_clusters.save(filename)
    print(f"Interactive map saved at: {filename}")


def plot_top10_comparison(df_results, save_path=None):
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
    plt.title('Top 10 Configurations - Silhouette, Davies-Bouldin, Calinski-Harabasz', fontsize=16)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=12)

    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(os.path.join(save_path, "top10_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()


def generate_combined_map(df, save_path=None, filename="clusters_combined_map.html"):
    """
    Generate an interactive map combining a contour interpolation layer and cluster points, excluding outliers.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'latitude', 'longitude', 'velocity', and 'cluster' columns.
    - save_path (str): Directory to save the HTML file.
    - filename (str): Output HTML filename.
    """
    # Remove outliers (-1)
    valid_clusters_df = df[df['cluster'] != -1].copy()

    # Create base map
    map_combined = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12, tiles="CartoDB Positron")

    # Grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 600),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 600)
    )

    # Interpolate using cubic
    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity'],
        (grid_x, grid_y),
        method='cubic'
    )

    # Create figure for contour with a better colormap
    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity'].min(), df['velocity'].max(), 20)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.7)  # <--- changed colormap
    plt.colorbar(contour, label="Vertical Displacement (mm/year)")
    ax.axis('off')

    # Save temporary image
    temp_path = os.path.join(tempfile.gettempdir(), "combined_contour_temp.png")
    plt.savefig(temp_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # Add contour layer
    overlay = ImageOverlay(
        image=temp_path,
        bounds=[[df['latitude'].min(), df['longitude'].min()],
                [df['latitude'].max(), df['longitude'].max()]],
        opacity=0.65
    )
    overlay.add_to(map_combined)

    # Color map for clusters
    unique_clusters = valid_clusters_df['cluster'].unique()
    cmap = cm.get_cmap('tab10', len(unique_clusters))
    colors_dict = {cluster: mcolors.rgb2hex(cmap(i)[:3]) for i, cluster in enumerate(unique_clusters)}

    # Add valid cluster points only
    for idx, row in valid_clusters_df.iterrows():
        color = colors_dict.get(row['cluster'], 'gray')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(map_combined)

    # Save final map
    output_file = os.path.join(save_path, filename)
    map_combined.save(output_file)
    print(f"✔️ Combined map (without outliers) saved: {output_file}")
