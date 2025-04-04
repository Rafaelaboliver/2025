#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:40:31 2025

@author: rafaela
"""

import os
import pyproj
#import hdbscan
import folium
#import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt

#from IPython.display import IFrame
from sklearn.cluster import DBSCAN
from IPython.display import display
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
sys.path.append('/home/rafaela/internship/2025/')  # Add the correct directory
from functions import transform_time_series, filter_by_polygon, linear_detrend, process_rolling_mean



# 1 - Importing the data
folder_path = '/home/rafaela/internship/2025/raw_data/vertical/'
file_name = 'EGMS_L3_E37N23_100km_U_2019_2023_1.csv'
file_path = os.path.join(folder_path, file_name)
raw_file = pd.read_csv(file_path)

# 2 - Selecting the columns
filtered_file = raw_file.iloc[:, [1,2,3,5]+list(range(11,raw_file.shape[1]))]

# 3 - Coordinates conversion
#dataframe copy to avoid changes in the original dataframe
file = filtered_file.copy()

#defining the coordinates systems
etrs89_laea = pyproj.CRS("EPSG:3035")
utm_32 = pyproj.CRS('EPSG:23032')
wgs_84 = pyproj.CRS("EPSG:4326")

#Creating the 'transformer' to covert the coordinates
transformer = pyproj.Transformer.from_crs(etrs89_laea, wgs_84)

#Changing the coordinates and dropping the old ones
latitude, longitude = transformer.transform(file['northing'].to_numpy(), file['easting'].to_numpy())
file.drop(columns=['easting', 'northing'], inplace=True)

#Inserting the new latitude and longitude columns at the beginning of the dataframe
file.insert(0, 'latitude', latitude)
file.insert(1, 'longitude', longitude)

# 4 - Selecting the study area (sa)

"""

Filters a DataFrame based on geographical boundaries.

"""
sa_lat_min = 43.670
sa_lat_max = 43.700
sa_long_min = 3.410
sa_long_max = 3.480

file_sa = filter_by_polygon(file, sa_lat_min, sa_lat_max, sa_long_min, sa_long_max)

# 5 - Applying the function to transform the time series

"""
Transforms the DataFrame so that dates become the index, while preserving point metadata.

Parameters:
- file_sa (pd.DataFrame): The original DataFrame with columns ['latitude', 'longitude', 'height', 'mean_velocity', dates...].

"""
pixels_dict, metadata_dict = transform_time_series(file_sa)

# 6 - Removing the regional pattern (time series detrend)

"""
Removes a linear trend from a time series.

Parameters:
- column_name (str): Name of the column (used for naming the detrended output).
- pixel_data[column_name] (pd.Series): data to be detrended.

"""
# Dictionary to store the pixels detrended results
detrended_pixels_dic = {}

for pixel_id, pixel_data in pixels_dict.items():
       
    if pixel_data.empty:
        #print(f"⚠️  Pixel {pixel_id} está vazio, pulando detrend...")
        continue  # Pula esse pixel

    column_name = pixel_data.columns[0]
    
    #Applying the linear detrend function
    detrended_df = linear_detrend(column_name, pixel_data[column_name]) 
    detrended_pixels_dic[pixel_id] = detrended_df
    

# 7 - Applying the rolling mean function

"""
Process rolling means for all pixels, calculate the overall mean, 
and subtract the overall mean from each pixel's rolling mean.

Parameters:
    pixels_dict (dict): Dictionary with pixel time series data.
    window_size (int): Size of the rolling window (default is 2).
    return_rolling_means_dict (bool): Whether to return the rolling mean dictionary (default is False).
"""

# time series
rolling_mean_reduced, overall_mean, rolling_means_dict = process_rolling_mean(
    pixels_dict=detrended_pixels_dic, 
    window_size=2 
)

# 8 - Velocities correction (velocities detrend)

velocities_df = file_sa[['mean_velocity']].copy()
#print(velocities_df)

velocities_avg = velocities_df['mean_velocity'].mean()
velocities_detrended = velocities_df['mean_velocity'] - velocities_avg
velocities_detrended_df = pd.DataFrame({'detrended_velocity': velocities_detrended})

# 9 - Removing positive velocities

velocities_negative = velocities_detrended_df.copy()
velocities_negative = velocities_negative[velocities_negative['detrended_velocity'] < 0]


# 10 - Trend Removal Check: visualization of data before and after corrections   

for _, pixel_df in pixels_dict.items():  
    # column name of the dictionaire
    pixel_column = pixel_df.columns[0]

    # Transforming it to string 
    pixel_column_str = str(pixel_column)  

    # Veryfing if 'pixel_column_str' it is in rolling_mean_reduced
    if pixel_column_str in rolling_mean_reduced.columns:
        plt.figure(figsize=(15, 5))

        # Try to access the data of the dictionarie
        try:
            pixel_series = pixel_df.iloc[:, 0]  
        except KeyError:
            print(f"Erro: Column {pixel_column_str} not find in DataFrame dentro do dicionário.")
            continue  # Go to the next pixel if there is any Error
        
        # Plotting the original data (before the correction)
        plt.plot(pixel_series.index, pixel_series, label=f'Original {pixel_column_str}', linestyle='dashed', color='tomato', alpha=0.7, linewidth=2)

        # Plotting the adjusted data
        plt.plot(rolling_mean_reduced.index, rolling_mean_reduced[pixel_column_str], label=f'Refined {pixel_column_str}', color='saddlebrown', linewidth=2)

        # Improving the plot
        plt.legend()
        plt.title(f'Pixel Comparison {pixel_column_str}')
        plt.xlabel('Time')
        plt.ylabel('Displacement (mm)')
        plt.grid(True, linestyle='--', alpha=0.5)  
        plt.show()
    

# 11 - clustering

#filtrar o metadata com os pixels aue estao em velocities_negative:
    filtered_metadata = {key: metadata_dict[key] for key in velocities_negative.index}
# Criar DataFrame para clustering com velocidades 
clustering_data = pd.DataFrame({
    'pixel': list(filtered_metadata.keys()),
    'latitude': [filtered_metadata[key]['latitude'] for key in filtered_metadata],
    'longitude': [filtered_metadata[key]['longitude'] for key in filtered_metadata],
    'velocity': [velocities_negative.loc[key, 'detrended_velocity'] for key in filtered_metadata]  # Usando as velocidades brutas
})

#Definir o pixel como indice do DataFrame
clustering_data.set_index('pixel', inplace=True)

# Normalizar os dados antes de aplicar DBSCAN
#scaler = StandardScaler()
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data[['latitude', 'longitude', 'velocity']])

# Aplicar DBSCAN ao conjunto de dados escalado
clusterer = DBSCAN(eps=0.090, min_samples=12, metric='euclidean')
clustering_data['cluster'] = clusterer.fit_predict(clustering_data_scaled)


#----------------------------------------------------GRAFICOS-----------------------------------------------------------------

# 12 - Visualizaçao os clusters 
plt.figure(figsize=(10, 6))
sc = plt.scatter(clustering_data['longitude'], clustering_data['latitude'], 
                 c=clustering_data['cluster'], cmap='viridis', s=25)  # 'inferno' destaca bem os valores negativos
plt.colorbar(sc, label="Cluster")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Clusters detectados - DBSCAN")
plt.grid()
plt.show()

# 13 - Visualizaçao os clusters + velocidades
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(clustering_data['longitude'], clustering_data['latitude'], clustering_data['velocity'], c=clustering_data['cluster'], cmap='viridis')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Velocity')
plt.title('Clusters - 3D Visualization')
plt.show()

# 14 - Testes para verificar os clusters
plt.hist(clusterer.probabilities_, bins=20, edgecolor='black')
plt.xlabel('Probabilidade de Clusterização')
plt.ylabel('Frequência')
plt.title('Distribuição das Probabilidades de Cluster')
plt.show()

sns.histplot(clustering_data['persistence'].dropna(), bins=20, kde=True)
plt.xlabel("Persistência do Cluster")
plt.ylabel("Frequência")
plt.title("Distribuição da Persistência dos Clusters")
plt.show()

import matplotlib.cm as cm
import matplotlib.colors as colors
# Criar um mapa centralizado na média das coordenadas
map_clusters = folium.Map(location=[clustering_data['latitude'].mean(), clustering_data['longitude'].mean()], zoom_start=12)

# Criar uma paleta de cores para os clusters
unique_clusters = clustering_data['cluster'].unique()
cmap = cm.get_cmap('tab10', len(unique_clusters))
colors_dict = {cluster: colors.rgb2hex(cmap(i)[:3]) for i, cluster in enumerate(unique_clusters)}

# Adicionar pontos ao mapa
for idx, row in clustering_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='grey' if row['cluster'] == -1 else colors_dict[row['cluster']],
        fill=True,
        fill_color='grey' if row['cluster'] == -1 else colors_dict[row['cluster']],
        fill_opacity=0.6
    ).add_to(map_clusters)

# Exibir mapa
map_clusters.save("clusters_mapa2.html")

#------------------------------------------VERIFICAÇAO-QUALIDADE-CLUSTER-------------------------------------------------------

import numpy as np

# Definir faixas de parâmetros
eps_values = np.arange(0.0900, 0.150)  # Testa valores entre 0.044 e 0.050
min_samples_values = [5, 6, 7, 8, 9, 10, 11, 12]
metrics = ['euclidean', 'manhattan']

best_config = None
best_silhouette = -1

# Loop pelos parâmetros
test_results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        for metric in metrics:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = clusterer.fit_predict(clustering_data_scaled)
            
            # Contar outliers e clusters
            num_outliers = np.sum(labels == -1)
            n_clusters = len(np.unique(labels[labels != -1]))
            
            # Avaliação apenas se houver mais de um cluster
            if n_clusters > 1:
                valid_clusters = labels[labels != -1]
                valid_features = clustering_data_scaled[labels != -1]
                silhouette_avg = silhouette_score(valid_features, valid_clusters)
                db_score = davies_bouldin_score(valid_features, valid_clusters)
                
                # Salvar resultados
                test_results.append((eps, min_samples, metric, num_outliers, n_clusters, silhouette_avg, db_score))
                
                # Atualizar melhor configuração
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_config = (eps, min_samples, metric, silhouette_avg, db_score)

# Criar DataFrame com os resultados
df_results = pd.DataFrame(test_results, columns=['eps', 'min_samples', 'metric', 'outliers', 'clusters', 'silhouette', 'davies_bouldin'])

# Exibir os melhores resultados
print("Melhor configuração encontrada:")
print(f"eps={best_config[0]}, min_samples={best_config[1]}, metric={best_config[2]}")
print(f"Silhouette Score: {best_config[3]:.4f}, Davies-Bouldin Score: {best_config[4]:.4f}")


num_outliers = (clustering_data['cluster'] == -1).sum()
print(f"Número de outliers detectados pelo DBSCAN: {num_outliers}")

# Contar quantos clusters únicos existem (excluindo outliers, que são rotulados como -1)
n_clusters = len(np.unique(clustering_data['cluster'][clustering_data['cluster'] != -1]))
print(f"Número de clusters válidos (excluindo outliers): {n_clusters}")

# # # Só calcula o Silhouette Score se houver mais de um cluster válido
# # if n_clusters > 1:
# #     silhouette_avg = silhouette_score(clustering_data_scaled, clustering_data['cluster'])
# #     print(f"Silhouette Score: {silhouette_avg}")
# # else:
# #     print("Silhouette Score não pode ser calculado pois há apenas um cluster válido.")

# silhouette_avg = silhouette_score(clustering_data_scaled, clustering_data['cluster'])
# db_score = davies_bouldin_score(clustering_data_scaled, clustering_data['cluster'])
# print(f"Silhouette Score: {silhouette_avg}")
# print(f"Davies-Bouldin Score: {db_score}")

valid_clusters = clustering_data[clustering_data['cluster'] != -1]['cluster']
valid_features = clustering_data_scaled[clustering_data['cluster'] != -1]

# Apenas calcular Silhouette Score se houver pelo menos 2 clusters
if len(np.unique(valid_clusters)) > 1:
    silhouette_avg = silhouette_score(valid_features, valid_clusters)
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("Silhouette Score não pode ser calculado (menos de 2 clusters válidos).")

if len(np.unique(valid_clusters)) > 1:
    db_score = davies_bouldin_score(valid_features, valid_clusters)
    print(f"Davies-Bouldin Score: {db_score}")
else:
    print("Davies-Bouldin Score não pode ser calculado (menos de 2 clusters válidos).")
    
