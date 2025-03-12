# -*- coding: utf-8 -*-

"""

Created on Tue Apr 30 10:04:00 2024



@author: leoda

"""



import matplotlib.pyplot as plt

import geopandas as gpd

from sklearn.cluster import DBSCAN

import pandas as pd

from shapely.ops import unary_union

import numpy as np



def cluster_points_with_values(df, attribute_name, eps_spatial, eps_value, target_value, min_samples):

    # Extract coordinates and attribute values

    coords = df.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    attribute_values = df[attribute_name].values.reshape(-1, 1)

    # Combine spatial and attribute features

    feature_matrix = np.hstack((coords, attribute_values))

    # Cluster points using DBSCAN

    db = DBSCAN(eps=eps_spatial, min_samples=min_samples).fit(feature_matrix)

    

    # Extract cluster labels

    cluster_labels = db.labels_

    

    # Filter out noise points (label = -1) and points with attribute values far from the given threshold

    df['cluster'] = cluster_labels

    df_filtered = df[(df['cluster'] != -1) & ((df[attribute_name]) < target_value-eps_value)]

    

    return df_filtered



# Load data from shapefile

df = gpd.read_file(r'C:/Users/leoda/Desktop/PPE Master AG/Data_EGMS/canyonrestreinttest.shp')



# Specify the attribute name and threshold value for clustering

attribute_name = 'mean_veloc'  # Change this to the name of your attribute

target_value = -6  # Specify the target attribute value around which to cluster

eps_spatial = 150  # Maximum distance for spatial clustering

eps_value = -0.5 # Maximum difference in attribute value for clustering

min_samples = 2  # Minimum number of points to form a cluster



# Cluster points based on spatial and attribute proximity

clustered_df = cluster_points_with_values(df, attribute_name, eps_spatial, eps_value, target_value, min_samples)



# Visualize the clustered points

fig, ax = plt.subplots()

ax.scatter(x=clustered_df['easting'], y=clustered_df['northing'], c=clustered_df['cluster'], marker='.', cmap='tab10')



# Create convex hulls from each cluster

hulls = []

print(clustered_df['cluster'])

for clusterid, frame in clustered_df.groupby('cluster'):

    geom = unary_union(frame.geometry.tolist()).convex_hull

    hulls.append([clusterid, geom])

    

df_clusters = pd.DataFrame.from_records(data=hulls, columns=['cluster','geometry'])

gdf_clusters = gpd.GeoDataFrame(data=df_clusters, geometry=df_clusters['geometry'], crs=df.crs)



index_poly = np.where(gdf_clusters["geometry"].type=='Polygon')

index_line = np.where(gdf_clusters["geometry"].type=='Linestring')





gdf_clusters_poly = gdf_clusters.iloc[index_poly[0]]

gdf_clusters_line = gdf_clusters.iloc[index_line[0]]



# Créer une zone tampon autour de chaque polygone avec une distance de 100 mètres

gdf_clusters_poly['buffered_geometry'] = gdf_clusters_poly['geometry'].buffer(100)



# Afficher les premières lignes pour vérifier le résultat

print(gdf_clusters_poly.head())



# Spécifier le chemin de sortie pour le fichier shapefile tamponné

output_path_buffered = 'C:/Users/leoda/Desktop/PPE Master AG/Data_EGMS/tamponne.shp'



# Enregistrer le GeoDataFrame avec les zones tamponnées dans un fichier shapefile

gdf_clusters_poly.iloc[:,2].to_file(output_path_buffered)





# Save the clustered points as a new shapefile

output_path_poly = r'C:/Users/leoda/Desktop/PPE Master AG/Data_EGMS/areaf_sample_clustered.shp'

#output_path_line = r'C:/Users/leoda/Desktop/PPE Master AG/Data_EGMS/bounderyf_sample_clustered.shp'



#DataF = pd.DataFrame(clustered_df)

#DataF.to_csv('C:/Users/leoda/Desktop/PPE Master AG/Data_EGMS/bs_sample_clustered.csv')

gdf_clusters_poly.to_file(output_path_poly)

#gdf_clusters_line.to_file(output_path_line)



plt.show()

