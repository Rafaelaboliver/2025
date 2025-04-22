#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:19:24 2025

@author: rafaela
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from scipy.interpolate import griddata
import tempfile
import os

def plot_clusters_2d(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis', s=25)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("DBSCAN Clusters (2D)")
    plt.grid()
    plt.show()

def plot_clusters_3d(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['longitude'], df['latitude'], df['velocity'], c=df['cluster'], cmap='viridis')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Velocity')
    plt.title('Clusters - 3D View')
    plt.show()

def generate_cluster_map(df, filename="clusters_contour_map.html"):
    map_clusters = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['longitude'].min(), df['longitude'].max(), 300),
        np.linspace(df['latitude'].min(), df['latitude'].max(), 300)
    )
    grid_z = griddata(
        (df['longitude'], df['latitude']),
        df['velocity'],
        (grid_x, grid_y),
        method='cubic'
    )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(df['velocity'].min(), df['velocity'].max(), 10)
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="bwr", alpha=0.7)
    plt.colorbar(contour, label="Vertical Velocity")
    ax.axis('off')

    temp_path = os.path.join(tempfile.gettempdir(), "contour_map.png")
    plt.savefig(temp_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    overlay = ImageOverlay(
        image=temp_path,
        bounds=[[df['latitude'].min(), df['longitude'].min()],
                [df['latitude'].max(), df['longitude'].max()]],
        opacity=0.6
    )
    overlay.add_to(map_clusters)
    map_clusters.save(filename)
    
    print(f"Map saved: {filename}")