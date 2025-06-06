#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:50:18 2025

@author: rafaela
"""

import pyproj
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression

def load_data(path):
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)

def convert_coordinates_pyproj(df, source_epsg, target_epsg):
    """
    Converts easting/northing columns to latitude/longitude using pyproj.

    Parameters:
    - df (pd.DataFrame): Must contain 'easting' and 'northing' columns
    - source_epsg (str or int): EPSG code of original CRS
    - target_epsg (str or int): EPSG code of desired output CRS

    Returns:
    - pd.DataFrame: Copy of input with 'latitude' and 'longitude' added
    """
    transformer = pyproj.Transformer.from_crs(source_epsg, target_epsg)
    latitude, longitude = transformer.transform(df["northing"].to_numpy(), df["easting"].to_numpy())

    df_converted = df.copy()
    df_converted.drop(columns=["easting", "northing"], inplace=True)
    df_converted.insert(0, "latitude", latitude)
    df_converted.insert(1, "longitude", longitude)

    return df_converted

def filter_by_polygon(dataframe, lat_min=None, lat_max=None, long_min=None, long_max=None):
    """
    Filters a DataFrame based on geographic bounding box.
    If no bounds are provided, the entire spatial extent of the DataFrame is used.

    Parameters:
    - dataframe (pd.DataFrame): Input data with 'latitude' and 'longitude' columns.
    - lat_min, lat_max, long_min, long_max (float or None): Latitude and longitude bounds.

    Returns:
    - pd.DataFrame: Subset containing only the points within the specified bounds.
    """

    if lat_min is None:
        lat_min = dataframe['latitude'].min()
    if lat_max is None:
        lat_max = dataframe['latitude'].max()
    if long_min is None:
        long_min = dataframe['longitude'].min()
    if long_max is None:
        long_max = dataframe['longitude'].max()


    filtered_df = dataframe[
        (dataframe['latitude'] >= lat_min) & (dataframe['latitude'] <= lat_max) &
        (dataframe['longitude'] >= long_min) & (dataframe['longitude'] <= long_max)
    ]
  
    return filtered_df

def transform_time_series(dataframe):
    """
    Converts a wide-format EGMS DataFrame into a dictionary of pixel-wise time series.

    Parameters:
    - dataframe (pd.DataFrame): Must contain ['latitude', 'longitude', 'height', 'mean_velocity', dates...]

    Returns:
    - pixels_dict (dict): Key = pixel index, Value = pd.DataFrame with time series
    - metadata_dict (dict): Key = pixel index, Value = dict with metadata (lat, lon, velocity)
    """
    
    # Dictionaries to store processed data
    pixels_dict = {}
    metadata_dict = {}
    
    for idx, row in dataframe.iterrows():  
        metadata = row[['latitude', 'longitude', 'height', 'mean_velocity']].to_dict()
        metadata["pixel"] = idx  
        metadata_dict[idx] = metadata

        time_series = row.iloc[4:]  
        time_series.index = pd.to_datetime(time_series.index, format='%Y%m%d')
        pixels_dict[idx] = pd.DataFrame({f"{idx}": time_series})
    
    return pixels_dict, metadata_dict

def linear_detrend(column_name, indice_data):
    """
   Removes a linear trend from a time series using linear regression.

    Parameters:
    - column_name (str): Name to assign to the resulting column.
    - time_series (pd.Series): Original time series.

    Returns:
    - pd.DataFrame: Detrended time series with the same index.
    """
    
    # Prepare the data for regression
    x = np.arange(len(indice_data)).reshape(-1, 1)  
    y = indice_data.values.reshape(-1, 1)          

    # Fit the linear model
    linear_model = LinearRegression().fit(x, y)
    predicted = linear_model.predict(x)
    
    #Calculate detrended values
    detrended_values = y.flatten() - predicted.flatten()

    #Create a dataframe with detrended values
    detrended_df = pd.DataFrame(
        detrended_values, 
        index=indice_data.index, 
        columns=[f'{column_name}'])
    
    return detrended_df

def process_rolling_mean(pixels_dict, window_size=2):
    """
    Applies rolling mean to each time series and subtracts the overall mean.

   Parameters:
   - pixels_dict (dict): Pixel-wise time series
   - window_size (int): Size of the moving window

   Returns:
   - pd.DataFrame: Rolling mean with overall trend removed
   - pd.Series: Overall average
   - dict: Individual rolling means
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

def discretize_velocity(velocities: pd.Series, print_info: bool = True) -> pd.DataFrame:
    """
    Discretizes a Series of velocities into 5 ordinal classes based on standard deviation from the mean.

    Class 1 - Very low subsidance (less negative)
    Class 2 - Low subsidance
    Class 3 - Moderate subsidance
    Class 4 - High subsidance
    Class 5 - Very high subsidance (more negative)

    Parameters:
    - velocities: pd.Series of detrended velocities (only negative values)
    - print_info: Whether to print threshold information

    Returns:
    - DataFrame with original values and corresponding class labels
    """
    mean = velocities.mean()
    std = velocities.std()

    # Define thresholds using standard deviation
    thresholds = [mean + std * 1.5,
                  mean + std * 0.5,
                  mean - std * 0.5,
                  mean - std * 1.5]

    def assign_class(v):
        if v >= thresholds[0]:
            return 1
        elif v >= thresholds[1]:
            return 2
        elif v >= thresholds[2]:
            return 3
        elif v >= thresholds[3]:
            return 4
        else:
            return 5

    df_discretized = pd.DataFrame({'detrended_velocity': velocities})
    df_discretized['velocity_class'] = df_discretized['detrended_velocity'].apply(assign_class)

    if print_info:
        print("Discretization thresholds for 5 classes:")
        print(f"  Class 1 (very low): >= {thresholds[0]:.4f}")
        print(f"  Class 2 (low):     >= {thresholds[1]:.4f} and < {thresholds[0]:.4f}")
        print(f"  Class 3 (moderate):>= {thresholds[2]:.4f} and < {thresholds[1]:.4f}")
        print(f"  Class 4 (high):    >= {thresholds[3]:.4f} and < {thresholds[2]:.4f}")
        print(f"  Class 5 (very high):< {thresholds[3]:.4f}")
        print(f"  Min velocity: {velocities.min():.4f}")
        print(f"  Max velocity: {velocities.max():.4f}")

    return df_discretized


def get_nearby_cluster_pixels(df_clustered, lat_ref, lon_ref, cluster_column='cluster', radius_meters=100, max_points=10):
    """
    Identifies pixel IDs near a reference location and belonging to a cluster (not outliers).
    """

    # Filter out points that are considered outliers (clustered points only)
    filtered = df_clustered[df_clustered[cluster_column] != -1].copy()
    
    # Function to check if a pixel is within the radius of the reference point
    def is_nearby(row):
        return geodesic((lat_ref, lon_ref), (row['latitude'], row['longitude'])).meters <= radius_meters

    # Apply the distance function
    filtered['distance'] = filtered.apply(is_nearby, axis=1)
    
    # Filter the pixels that are within the radius
    nearby = filtered[filtered['distance']].copy()

    # If no pixels are found, return an empty DataFrame
    if nearby.empty:
        print(" No cluster points found within the specified radius.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Sort the nearby points by distance to the reference location
    nearby_sorted = nearby.copy()
    nearby_sorted['dist_m'] = nearby_sorted.apply(
        lambda row: geodesic((lat_ref, lon_ref), (row['latitude'], row['longitude'])).meters,
        axis=1
    )
    nearby_sorted = nearby_sorted.sort_values(by='dist_m').head(max_points)

    # Return the sorted DataFrame
    return nearby_sorted
