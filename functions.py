#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:50:18 2025

@author: rafaela
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def filter_by_polygon(dataframe, lat_min, lat_max, long_min, long_max):
    """
    Filters a DataFrame based on geographical boundaries.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing 'latitude' and 'longitude' columns.
    - lat_min (float): Minimum latitude.
    - lat_max (float): Maximum latitude.
    - long_min (float): Minimum longitude.
    - long_max (float): Maximum longitude.

    Returns:
    - pd.DataFrame: A DataFrame containing only the points within the provided boundaries.
    """
    
    filtered_df = dataframe[
        (dataframe['latitude'] > lat_min) & (dataframe['latitude'] < lat_max) &
        (dataframe['longitude'] > long_min) & (dataframe['longitude'] < long_max)
    ]
    return filtered_df

def transform_time_series(dataframe):
    """
    Transforms the DataFrame so that dates become the index, while preserving point metadata.
    
    Parameters:
    - dataframe (pd.DataFrame): The original DataFrame with columns ['latitude', 'longitude', 'height', 'mean_velocity', dates...].
    
    Returns:
    - pixels_dict (dict): A dictionary where keys are pixel indices and values are DataFrames with dates as index and variations as values.
    - metadata_dict (dict): A dictionary where keys are pixel indices and values are metadata (latitude, longitude, mean_velocity).
    """
    
    # Dictionaries to store processed data
    pixels_dict = {}
    metadata_dict = {}
    
    for idx, row in dataframe.iterrows():  # Garante que pegamos os índices corretamente
        metadata = row[['latitude', 'longitude', 'height', 'mean_velocity']].to_dict()
        metadata["pixel"] = idx  # Agora, "value" recebe o índice real do DataFrame
        metadata_dict[idx] = metadata

        # Extraindo série temporal
        time_series = row.iloc[4:]  # Do 5º elemento em diante (colunas de datas)
        time_series.index = pd.to_datetime(time_series.index, format='%Y%m%d')
        # Adiciona prints para ver como ficou depois da conversão
        # print(f"Pixel {idx} - Índices após a conversão para datetime:")
        # print(time_series.index[:5])  # Mostra os primeiros índices para conferência
        # print(f"Pixel {idx} - time_series após a conversão:")
        # print(time_series.head())  # Mostra os primeiros valores

        pixels_dict[idx] = pd.DataFrame({"variation": time_series})  # Forma mais segura
    
    return pixels_dict, metadata_dict

def linear_detrend(column_name, indice_data):
    """
    Removes a linear trend from a time series.

    Parameters:
    - column_name (str): Name of the column (used for naming the detrended output).
    - time_series_data (pd.Series): Time series data to be detrended.

    Returns:
    - pd.DataFrame: A DataFrame with the detrended values.
    """
    
    # Prepare the data for regression
    x = np.arange(len(indice_data)).reshape(-1, 1)  
    y = indice_data.values.reshape(-1, 1)          

    # Fit the linear model
    linear_model = LinearRegression().fit(x, y)
    regression_line = linear_model.predict(x)
    
    #Calculate detrended values
    detrended_values = y.flatten() - regression_line.flatten()

    #Create a dataframe with detrended values
    detrended_df = pd.DataFrame(
        detrended_values, 
        index=indice_data.index, 
        columns=[f'{column_name}'])
    
    return detrended_df

def process_rolling_mean(pixels_dict, window_size=2):
    """
    Process rolling means for all pixels, calculate the overall mean, 
    and subtract the overall mean from each pixel's rolling mean.
    
    Parameters:
        pixels_dict (dict): Dictionary with pixel time series data.
        window_size (int): Size of the rolling window (default is 2).
        return_rolling_means_dict (bool): Whether to return the rolling mean dictionary (default is False).
    
    Returns:
        pd.DataFrame: Consolidated DataFrame with reduced rolling mean data (all pixels).
        pd.Series: Overall rolling mean (time series).
        dict (optional): Dictionary with rolling means per pixel (if return_rolling_means_dict=True).
    """
    
    if not pixels_dict:
        raise ValueError("pixels_dict cannot be empty.")

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    
    # Step 1: Calculate rolling mean for each pixel
    rolling_means_dict = {}
    rolling_means_list = []  # List to store DataFrames for efficient concatenation
    pixels_id = []
    
    # Modify the loop to include pixel_ids properly as column names
    for pixel_id, pixel_data in pixels_dict.items():
            
        if pixel_data.empty or len(pixel_data.columns) == 0:
            print(f"Skipping pixel_id {pixel_id} because pixel_data is empty!")
            continue
        column_name = pixel_data.columns[0]

        rolling_mean = pixel_data[column_name].rolling(window_size, min_periods=1).mean()

        rolling_means_dict[pixel_id] = rolling_mean.to_frame(name=f'{column_name}')
     
        # Add the DataFrame version of rolling_mean to rolling_means_list
        rolling_means_list.append(rolling_means_dict[pixel_id])
        pixels_id.append(pixel_id)
           
    # Step 2: Combine all rolling means into a single DataFrame
    all_rolling_means = pd.concat(rolling_means_list, axis=1)
    
    # Step 3: Calculate the overall rolling mean
    overall_mean = all_rolling_means.stack().mean()
    overall_mean = pd.Series(overall_mean, name= 'overall average').to_frame()  # Rename column
    
    # Step 4: Subtract the overall mean from each pixel's rolling mean
    rolling_mean_reduced = all_rolling_means.sub(overall_mean.squeeze(), axis=0)
    rolling_mean_reduced.dropna(inplace=True)  # Remove any rows with NaN values
    
    return rolling_mean_reduced, overall_mean, rolling_means_dict
    