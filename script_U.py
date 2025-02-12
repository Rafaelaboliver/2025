#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:40:31 2025

@author: rafaela
"""

import os
import pyproj
import hdbscan
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import IFrame

import sys
sys.path.append('/home/rafaela/internship/2025/')  # Add the correct directory
from functions import transform_time_series, filter_by_polygon, linear_detrend, process_rolling_mean



# 1 - Importing the data
folder_path = '/home/rafaela/internship/2025/raw_data/vertical'
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


# 9 - Trend Removal Check: visualization of data before and after corrections   

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
        

# 10 - clustering

# Selecionar apenas pixels com forte erosão (velocidade vertical corrigida negativa e significativa)
threshold = np.percentile(rolling_mean_reduced.mean(axis=1), 5)  # Definir como os 5% menores valores
erosion_pixels = rolling_mean_reduced[rolling_mean_reduced.mean(axis=1) < threshold]

# Criar DataFrame para clustering com velocidades 
clustering_data = pd.DataFrame({
    'latitude': [metadata_dict[key]['latitude'] for key in metadata_dict],
    'longitude': [metadata_dict[key]['longitude'] for key in metadata_dict],
    'velocity': [velocities_detrended_df.loc[key, 'detrended_velocity'] for key in velocities_detrended_df.index]  # Usando as velocidades brutas
})

# Aplicar HDBSCAN ao conjunto de dados
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
clustering_data['cluster'] = clusterer.fit_predict(clustering_data[['latitude', 'longitude', 'velocity']])

# Criar mapa interativo
if not clustering_data.empty:
    mapa = folium.Map(location=[clustering_data['latitude'].mean(), clustering_data['longitude'].mean()], zoom_start=14)

    # Adicionar os pontos ao mapa
    for _, row in clustering_data.iterrows():
        if row['velocity'] < -4:
            color = 'red'  # 🚨 Forte erosão
        elif -4 <= row['velocity'] < -2.5:
            color = 'orange'  # ⚠️ Potencial erosão
        else:
            color = 'green'  # ✅ Estável

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(mapa)

    # Salvar o mapa interativo
    mapa.save("mapa_erosao_teste.html")

else:
    print("Nenhum dado válido para visualização.")


# Visualizar os clusters 
plt.figure(figsize=(10, 6))
sc = plt.scatter(clustering_data['longitude'], clustering_data['latitude'], 
                 c=clustering_data['velocity'], cmap='inferno', s=20)  # 'inferno' destaca bem os valores negativos
plt.colorbar(sc, label="Velocidade Média (mm/ano)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mapa de Velocidade Média (Erosão)")
plt.grid()
plt.show()